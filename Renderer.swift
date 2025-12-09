import MetalKit
import AppKit
import simd

struct Uniforms {
    var viewProjectionMatrix: matrix_float4x4
    var modelMatrix: matrix_float4x4
}

struct PointVertex {
    var position: SIMD4<Float>
    var scalar: Float       // A-weighted loudness (0..1)
    var hpRatio: Float      // Harmonic-percussive ratio (0 or 1)
    var padding: SIMD2<Float> = .zero  // pad to 32 bytes for alignment
}

struct PointCloudResource {
    let buffer: MTLBuffer
    let count: Int
    let name: String
}

struct MeshVertex {
    var position: SIMD4<Float>
    var normal: SIMD3<Float>
    var colorScalar: Float
}

struct VoxelGridInfo {
    var resolution: SIMD3<UInt32>
    var _pad0: UInt32 = 0
    var origin: SIMD3<Float>
    var spacing: Float
    var pointCount: UInt32
    var _pad1: UInt32 = 0
}

struct IsoSurfaceParams {
    var isoValue: Float
    var maxVertices: UInt32
    var _pad: SIMD2<UInt32> = .zero
}

enum RenderMode {
    case song
    case trend
}

enum SurfaceRenderMode {
    case pointCloud
    case mesh
    case isosurface
}

enum MeshAlgorithm {
    case marchingTetrahedra
    case dualContouring
}

struct SongPointCloudLoader {
    /// Load a song-mode point cloud from a raw float32 binary file.
    ///
    /// The file layout must be:
    ///     [x, y, z, scalar, hp_ratio] per vertex (all Float32),
    /// where x,y,z are already normalized into clip-space-ish coordinates,
    /// scalar is A-weighted loudness in [0,1],
    /// and hp_ratio is harmonic–percussive ratio in [0,1].
    ///
    /// Returns:
    ///   - buffer: MTLBuffer containing an array of PointVertex
    ///   - count:  number of vertices in the buffer
    static func loadPointCloud(device: MTLDevice, from path: String) -> (buffer: MTLBuffer?, count: Int) {
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url) else {
            print("[SongPointCloudLoader] Failed to read point cloud at path: \(path)")
            return (nil, 0)
        }

        let floatCount = data.count / MemoryLayout<Float>.size
        if floatCount % 5 != 0 {
            print("[SongPointCloudLoader] Unexpected float count \(floatCount); expected a multiple of 5.")
            return (nil, 0)
        }

        let vertexCount = floatCount / 5
        if vertexCount == 0 {
            print("[SongPointCloudLoader] No vertices found in file: \(path)")
            return (nil, 0)
        }

        var vertices: [PointVertex] = []
        vertices.reserveCapacity(vertexCount)

        data.withUnsafeBytes { rawBuf in
            let buf = rawBuf.bindMemory(to: Float.self)
            var i = 0
            while i + 4 < floatCount {
                let x = buf[i + 0]
                let y = buf[i + 1]
                let z = buf[i + 2]
                let scalar = buf[i + 3]
                let hp = buf[i + 4]
                i += 5

                let position = SIMD4<Float>(x, y, z, 1.0)
                vertices.append(PointVertex(position: position, scalar: scalar, hpRatio: hp))
            }
        }

        guard let buffer = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<PointVertex>.stride * vertices.count,
            options: []
        ) else {
            print("[SongPointCloudLoader] Failed to create MTLBuffer.")
            return (nil, 0)
        }

        return (buffer, vertices.count)
    }
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pointPipelineState: MTLRenderPipelineState!
    var meshPipelineState: MTLRenderPipelineState!
    var depthState: MTLDepthStencilState?
    var gridClearPipeline: MTLComputePipelineState?
    var voxelSplatPipeline: MTLComputePipelineState?
    var gridNormalizePipeline: MTLComputePipelineState?
    var marchingTetraPipeline: MTLComputePipelineState?
    var dualCellPipeline: MTLComputePipelineState?
    var dualEdgeXPipeline: MTLComputePipelineState?
    var dualEdgeYPipeline: MTLComputePipelineState?
    var dualEdgeZPipeline: MTLComputePipelineState?

    private var activePointCloud: PointCloudResource?

    private var songPointCloud: PointCloudResource?
    private var trendPointCloud: PointCloudResource?

    private(set) var currentMode: RenderMode = .song
    private(set) var surfaceMode: SurfaceRenderMode = .pointCloud
    var uniformBuffer: MTLBuffer!
    var camera: Camera
    var inputController: InputController!
    private var scalarGridBuffer: MTLBuffer?
    private var gridCountsBuffer: MTLBuffer?
    private var gridMaxCountBuffer: MTLBuffer?
    private var meshVertexBuffer: MTLBuffer?
    private var meshVertexCountBuffer: MTLBuffer?
    private var dualCellVertexBuffer: MTLBuffer?
    private var dualCellMaskBuffer: MTLBuffer?
    private var meshVertexCount: Int = 0
    private var meshNeedsRebuild = false
    private let voxelResolution = SIMD3<UInt32>(64, 64, 64)
    private let gridOrigin = SIMD3<Float>(-1.0, -1.0, -1.0)
    private let maxMeshVertices: Int = 600_000
    private(set) var meshAlgorithm: MeshAlgorithm = .marchingTetrahedra

    var isovalue: Float = 0.5 {
        didSet {
            if surfaceMode == .mesh {
                meshNeedsRebuild = true
            }
        }
    }

    /// Load a song-mode point cloud from disk and replace the current vertex buffer.
    /// This can be called at runtime to switch between different songs.
    @discardableResult
    func loadSongPointCloud(at path: String) -> Bool {
        let result = SongPointCloudLoader.loadPointCloud(device: device, from: path)
        if let buffer = result.buffer, result.count > 0 {
            let name = URL(fileURLWithPath: path).lastPathComponent
            self.songPointCloud = PointCloudResource(buffer: buffer, count: result.count, name: name)
            refreshActivePointCloud()
            print("[Renderer] Loaded point cloud from: \(path) (vertices: \(result.count))")
            return true
        } else {
            print("[Renderer] Failed to load point cloud from: \(path)")
            return false
        }
    }

    /// Load the precomputed trend-mode point cloud.
    @discardableResult
    func loadTrendPointCloud(at path: String = "data/processed/trend_points.bin") -> Bool {
        let result = SongPointCloudLoader.loadPointCloud(device: device, from: path)
        if let buffer = result.buffer, result.count > 0 {
            let name = URL(fileURLWithPath: path).lastPathComponent
            self.trendPointCloud = PointCloudResource(buffer: buffer, count: result.count, name: name)
            refreshActivePointCloud()
            print("[Renderer] Loaded trend point cloud from: \(path) (vertices: \(result.count))")
            return true
        } else {
            print("[Renderer] Failed to load trend point cloud from: \(path)")
            return false
        }
    }

    /// Switch between song and trend rendering modes. Ensures buffers are loaded.
    /// Returns true if a buffer is ready for the chosen mode.
    @discardableResult
    func setMode(_ mode: RenderMode) -> Bool {
        currentMode = mode
        ensurePointCloudLoaded(for: mode)
        refreshActivePointCloud()
        invalidateDerivedSurfaces()
        return activePointCloud != nil && activePointCloud!.count > 0
    }

    /// Switch the visualization output between points or meshes
    func setSurfaceMode(_ mode: SurfaceRenderMode) {
        surfaceMode = mode
        invalidateDerivedSurfaces()
    }

    func setMeshAlgorithm(_ algorithm: MeshAlgorithm) {
        guard meshAlgorithm != algorithm else { return }
        meshAlgorithm = algorithm
        invalidateDerivedSurfaces()
    }

    private func ensurePointCloudLoaded(for mode: RenderMode) {
        switch mode {
        case .song:
            if songPointCloud == nil {
                _ = loadInitialSongPointCloudIfAvailable()
            }
        case .trend:
            if trendPointCloud == nil {
                _ = loadTrendPointCloud()
            }
        }
    }

    /// Current active point cloud name for UI display.
    func activePointCloudName() -> String? {
        switch currentMode {
        case .song:
            return songPointCloud?.name
        case .trend:
            return trendPointCloud?.name
        }
    }

    /// Attempt to find and load the first available .bin point cloud in common data folders.
    /// Falls back to a small built-in demo cloud so the renderer always has something to draw (song mode only).
    func loadInitialSongPointCloudIfAvailable() -> String? {
        if let existing = songPointCloud?.name { return existing }

        let fm = FileManager.default
        let searchRoots = [
            "data/processed",
            "data",
            FileManager.default.currentDirectoryPath
        ]

        for root in searchRoots {
            let rootURL = URL(fileURLWithPath: root)
            if let enumerator = fm.enumerator(at: rootURL, includingPropertiesForKeys: nil) {
                for case let fileURL as URL in enumerator {
                    if fileURL.pathExtension.lowercased() == "bin" {
                        if loadSongPointCloud(at: fileURL.path) {
                            return fileURL.lastPathComponent
                        }
                    }
                }
            }
        }

        // Fallback: generate a simple colored cube of points to avoid an empty scene.
        makeProceduralFallbackCloud()
        return songPointCloud?.name
    }

    /// Build a small procedural cube of colored points as a safe default.
    private func makeProceduralFallbackCloud() {
        var verts: [PointVertex] = []
        let positions: [SIMD3<Float>] = [
            SIMD3<Float>(-0.4, -0.4, -0.4),
            SIMD3<Float>( 0.4, -0.4, -0.4),
            SIMD3<Float>(-0.4,  0.4, -0.4),
            SIMD3<Float>( 0.4,  0.4, -0.4),
            SIMD3<Float>(-0.4, -0.4,  0.4),
            SIMD3<Float>( 0.4, -0.4,  0.4),
            SIMD3<Float>(-0.4,  0.4,  0.4),
            SIMD3<Float>( 0.4,  0.4,  0.4)
        ]

        for (i, pos) in positions.enumerated() {
            let scalar: Float = 1.0
            // Alternate hpRatio to give a small variation in fallback colors
            let hp: Float = (i % 2 == 0) ? 0.25 : 0.75
            verts.append(PointVertex(position: SIMD4<Float>(pos, 1.0), scalar: scalar, hpRatio: hp))
        }

        if let buffer = device.makeBuffer(
            bytes: verts,
            length: MemoryLayout<PointVertex>.stride * verts.count,
            options: []
        ) {
            songPointCloud = PointCloudResource(
                buffer: buffer,
                count: verts.count,
                name: "Procedural demo cloud"
            )
        } else {
            songPointCloud = nil
        }

        refreshActivePointCloud()
    }

    /// Ensure the active buffer matches the current mode.
    private func refreshActivePointCloud() {
        activePointCloud = (currentMode == .song) ? songPointCloud : trendPointCloud
        invalidateDerivedSurfaces()
    }

    private func invalidateDerivedSurfaces() {
        if surfaceMode == .mesh {
            meshNeedsRebuild = true
            meshVertexCount = 0
        }
    }

    private func ensureVoxelResources(pointCount: Int) {
        let cellCount = Int(voxelResolution.x * voxelResolution.y * voxelResolution.z)
        let cubeCount = Int(max((voxelResolution.x - 1) * (voxelResolution.y - 1) * (voxelResolution.z - 1), 1))
        let scalarBytes = cellCount * MemoryLayout<Float>.stride
        let countBytes = cellCount * MemoryLayout<UInt32>.stride
        let meshBytes = maxMeshVertices * MemoryLayout<MeshVertex>.stride
        let dualBytes = cubeCount * MemoryLayout<MeshVertex>.stride
        let dualMaskBytes = cubeCount * MemoryLayout<UInt32>.stride

        if scalarGridBuffer == nil || scalarGridBuffer!.length < scalarBytes {
            scalarGridBuffer = device.makeBuffer(length: scalarBytes, options: .storageModeShared)
        }
        if gridCountsBuffer == nil || gridCountsBuffer!.length < countBytes {
            gridCountsBuffer = device.makeBuffer(length: countBytes, options: .storageModeShared)
        }
        if gridMaxCountBuffer == nil {
            gridMaxCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        }
        if meshVertexBuffer == nil || meshVertexBuffer!.length < meshBytes {
            meshVertexBuffer = device.makeBuffer(length: meshBytes, options: .storageModeShared)
        }
        if meshVertexCountBuffer == nil {
            meshVertexCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        }
        if dualCellVertexBuffer == nil || dualCellVertexBuffer!.length < dualBytes {
            dualCellVertexBuffer = device.makeBuffer(length: dualBytes, options: .storageModeShared)
        }
        if dualCellMaskBuffer == nil || dualCellMaskBuffer!.length < dualMaskBytes {
            dualCellMaskBuffer = device.makeBuffer(length: dualMaskBytes, options: .storageModeShared)
        }
    }

    private func rebuildMeshIfNeeded() {
        guard meshNeedsRebuild, surfaceMode == .mesh else { return }
        guard let pointCloud = activePointCloud, pointCloud.count > 0 else { return }
        buildMesh(from: pointCloud)
    }

    private func buildMesh(from pointCloud: PointCloudResource) {
        guard let clearPipeline = gridClearPipeline,
              let splatPipeline = voxelSplatPipeline,
              let normalizePipeline = gridNormalizePipeline else {
            print("[Renderer] Compute pipelines missing; cannot build mesh.")
            meshNeedsRebuild = false
            return
        }

        if pointCloud.count == 0 {
            meshVertexCount = 0
            meshNeedsRebuild = false
            return
        }

        ensureVoxelResources(pointCount: pointCloud.count)

        let spacing = 2.0 / Float(max(voxelResolution.x - 1, 1))
        var gridInfo = VoxelGridInfo(
            resolution: voxelResolution,
            origin: gridOrigin,
            spacing: spacing,
            pointCount: UInt32(min(pointCloud.count, Int(UInt32.max)))
        )
        var isoParams = IsoSurfaceParams(
            isoValue: isovalue,
            maxVertices: UInt32(maxMeshVertices)
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let counts = gridCountsBuffer,
              let maxCount = gridMaxCountBuffer,
              let scalarGrid = scalarGridBuffer,
              let meshBuf = meshVertexBuffer,
              let meshCount = meshVertexCountBuffer else {
            return
        }

        let cellCount = Int(voxelResolution.x * voxelResolution.y * voxelResolution.z)
        let clearThreads = MTLSize(width: cellCount, height: 1, depth: 1)
        let clearTG = MTLSize(width: max(1, clearPipeline.threadExecutionWidth), height: 1, depth: 1)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(clearPipeline)
            encoder.setBuffer(counts, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBuffer(maxCount, offset: 0, index: 2)
            encoder.dispatchThreads(clearThreads, threadsPerThreadgroup: clearTG)
            encoder.endEncoding()
        }

        let splatThreads = MTLSize(width: pointCloud.count, height: 1, depth: 1)
        let splatTG = MTLSize(width: max(1, splatPipeline.threadExecutionWidth), height: 1, depth: 1)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(splatPipeline)
            encoder.setBuffer(pointCloud.buffer, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBuffer(counts, offset: 0, index: 2)
            encoder.setBuffer(maxCount, offset: 0, index: 3)
            encoder.dispatchThreads(splatThreads, threadsPerThreadgroup: splatTG)
            encoder.endEncoding()
        }

        let normalizeThreads = clearThreads
        let normTG = MTLSize(width: max(1, normalizePipeline.threadExecutionWidth), height: 1, depth: 1)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(normalizePipeline)
            encoder.setBuffer(counts, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBuffer(scalarGrid, offset: 0, index: 2)
            encoder.setBuffer(maxCount, offset: 0, index: 3)
            encoder.dispatchThreads(normalizeThreads, threadsPerThreadgroup: normTG)
            encoder.endEncoding()
        }

        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: meshCount, range: 0..<MemoryLayout<UInt32>.stride, value: 0)
            blit.endEncoding()
        }

        let cubeCount = Int((voxelResolution.x - 1) * (voxelResolution.y - 1) * (voxelResolution.z - 1))
        let mcThreads = MTLSize(width: max(cubeCount, 1), height: 1, depth: 1)
        switch meshAlgorithm {
        case .marchingTetrahedra:
            guard let marchingPipeline = marchingTetraPipeline else {
                print("[Renderer] Marching tetrahedra pipeline missing.")
                break
            }
            let mcTG = MTLSize(width: max(1, marchingPipeline.threadExecutionWidth), height: 1, depth: 1)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(marchingPipeline)
                encoder.setBuffer(scalarGrid, offset: 0, index: 0)
                encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
                encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
                encoder.setBuffer(meshBuf, offset: 0, index: 3)
                encoder.setBuffer(meshCount, offset: 0, index: 4)
                encoder.dispatchThreads(mcThreads, threadsPerThreadgroup: mcTG)
                encoder.endEncoding()
            }
        case .dualContouring:
            guard let cellPipeline = dualCellPipeline,
                  let edgeXPipe = dualEdgeXPipeline,
                  let edgeYPipe = dualEdgeYPipeline,
                  let edgeZPipe = dualEdgeZPipeline,
                  let cellVerts = dualCellVertexBuffer,
                  let cellMask = dualCellMaskBuffer else {
                print("[Renderer] Dual contouring pipelines missing; cannot build mesh.")
                break
            }

            if let mask = dualCellMaskBuffer {
                // Ensure the mask is clean before writing active flags.
                if let blit = commandBuffer.makeBlitCommandEncoder() {
                    blit.fill(buffer: mask, range: 0..<mask.length, value: 0)
                    blit.endEncoding()
                }
            }

            let cellTG = MTLSize(width: max(1, cellPipeline.threadExecutionWidth), height: 1, depth: 1)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(cellPipeline)
                encoder.setBuffer(scalarGrid, offset: 0, index: 0)
                encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
                encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
                encoder.setBuffer(cellVerts, offset: 0, index: 3)
                encoder.setBuffer(cellMask, offset: 0, index: 4)
                encoder.dispatchThreads(mcThreads, threadsPerThreadgroup: cellTG)
                encoder.endEncoding()
            }

            let edgeXCount = Int(max((voxelResolution.x - 1) * voxelResolution.y * voxelResolution.z, 1))
            let edgeXTG = MTLSize(width: max(1, edgeXPipe.threadExecutionWidth), height: 1, depth: 1)
            let edgeYCount = Int(max(voxelResolution.x * (voxelResolution.y - 1) * voxelResolution.z, 1))
            let edgeYTG = MTLSize(width: max(1, edgeYPipe.threadExecutionWidth), height: 1, depth: 1)
            let edgeZCount = Int(max(voxelResolution.x * voxelResolution.y * (voxelResolution.z - 1), 1))
            let edgeZTG = MTLSize(width: max(1, edgeZPipe.threadExecutionWidth), height: 1, depth: 1)

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(edgeXPipe)
                encoder.setBuffer(scalarGrid, offset: 0, index: 0)
                encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
                encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
                encoder.setBuffer(cellVerts, offset: 0, index: 3)
                encoder.setBuffer(cellMask, offset: 0, index: 4)
                encoder.setBuffer(meshBuf, offset: 0, index: 5)
                encoder.setBuffer(meshCount, offset: 0, index: 6)
                encoder.dispatchThreads(MTLSize(width: edgeXCount, height: 1, depth: 1), threadsPerThreadgroup: edgeXTG)
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(edgeYPipe)
                encoder.setBuffer(scalarGrid, offset: 0, index: 0)
                encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
                encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
                encoder.setBuffer(cellVerts, offset: 0, index: 3)
                encoder.setBuffer(cellMask, offset: 0, index: 4)
                encoder.setBuffer(meshBuf, offset: 0, index: 5)
                encoder.setBuffer(meshCount, offset: 0, index: 6)
                encoder.dispatchThreads(MTLSize(width: edgeYCount, height: 1, depth: 1), threadsPerThreadgroup: edgeYTG)
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(edgeZPipe)
                encoder.setBuffer(scalarGrid, offset: 0, index: 0)
                encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
                encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
                encoder.setBuffer(cellVerts, offset: 0, index: 3)
                encoder.setBuffer(cellMask, offset: 0, index: 4)
                encoder.setBuffer(meshBuf, offset: 0, index: 5)
                encoder.setBuffer(meshCount, offset: 0, index: 6)
                encoder.dispatchThreads(MTLSize(width: edgeZCount, height: 1, depth: 1), threadsPerThreadgroup: edgeZTG)
                encoder.endEncoding()
            }
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let countPtr = meshCount.contents().assumingMemoryBound(to: UInt32.self)
        meshVertexCount = Int(min(countPtr.pointee, UInt32(maxMeshVertices)))
        meshNeedsRebuild = false
    }

    init(view: MTKView) {
        self.device = view.device!
        self.commandQueue = self.device.makeCommandQueue()!
        self.camera = Camera()
        super.init()

        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.2, green: 0.2, blue: 0.25, alpha: 1.0)
        createPipelines(view: view)
        inputController = InputController(view: view, camera: camera)
    }

    private func createPipelines(view: MTKView) {
        guard let library = device.makeDefaultLibrary() else { return }
        let pointVertexFunction = library.makeFunction(name: "vertex_main")
        let pointFragmentFunction = library.makeFunction(name: "fragment_main")
        let meshVertexFunction = library.makeFunction(name: "mesh_vertex_main")
        let meshFragmentFunction = library.makeFunction(name: "mesh_fragment_main")

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: depthDescriptor)

        let pointDescriptor = MTLRenderPipelineDescriptor()
        pointDescriptor.vertexFunction = pointVertexFunction
        pointDescriptor.fragmentFunction = pointFragmentFunction
        pointDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pointDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat

        let meshDescriptor = MTLRenderPipelineDescriptor()
        meshDescriptor.vertexFunction = meshVertexFunction
        meshDescriptor.fragmentFunction = meshFragmentFunction
        meshDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        meshDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat

        do {
            pointPipelineState = try device.makeRenderPipelineState(descriptor: pointDescriptor)
            meshPipelineState = try device.makeRenderPipelineState(descriptor: meshDescriptor)
            if let clearFn = library.makeFunction(name: "clearGrid") {
                gridClearPipeline = try device.makeComputePipelineState(function: clearFn)
            }
            if let splatFn = library.makeFunction(name: "splatPointsToGrid") {
                voxelSplatPipeline = try device.makeComputePipelineState(function: splatFn)
            }
            if let normFn = library.makeFunction(name: "normalizeGrid") {
                gridNormalizePipeline = try device.makeComputePipelineState(function: normFn)
            }
            if let mcFn = library.makeFunction(name: "marchingTetrahedra") {
                marchingTetraPipeline = try device.makeComputePipelineState(function: mcFn)
            }
            if let dcCellFn = library.makeFunction(name: "dualContourCells") {
                dualCellPipeline = try device.makeComputePipelineState(function: dcCellFn)
            }
            if let edgeXFn = library.makeFunction(name: "dualContourEdgesX") {
                dualEdgeXPipeline = try device.makeComputePipelineState(function: edgeXFn)
            }
            if let edgeYFn = library.makeFunction(name: "dualContourEdgesY") {
                dualEdgeYPipeline = try device.makeComputePipelineState(function: edgeYFn)
            }
            if let edgeZFn = library.makeFunction(name: "dualContourEdgesZ") {
                dualEdgeZPipeline = try device.makeComputePipelineState(function: edgeZFn)
            }
        } catch {
            print("Failed to create pipeline state: \(error)")
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.aspectRatio = Float(size.width / size.height)
    }

    func draw(in view: MTKView) {
        inputController.update()
        rebuildMeshIfNeeded()

        var uniforms = Uniforms(
            viewProjectionMatrix: matrix_multiply(camera.projectionMatrix(), camera.viewMatrix()),
            modelMatrix: matrix_identity_float4x4
        )

        uniformBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<Uniforms>.size, options: [])

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let rpd = view.currentRenderPassDescriptor,
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd),
              let pointCloud = activePointCloud else { return }

        encoder.setDepthStencilState(depthState)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.setFrontFacing(.counterClockwise)
        let cull: MTLCullMode = (surfaceMode == .mesh && meshAlgorithm == .dualContouring) ? .none : .back
        encoder.setCullMode(cull)

        switch surfaceMode {
        case .mesh, .isosurface:
            if meshVertexCount > 0, let meshBuf = meshVertexBuffer {
                encoder.setRenderPipelineState(meshPipelineState)
                encoder.setVertexBuffer(meshBuf, offset: 0, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: meshVertexCount)
            } else {
                encoder.setRenderPipelineState(pointPipelineState)
                encoder.setVertexBuffer(pointCloud.buffer, offset: 0, index: 0)
                encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: pointCloud.count)
            }
        case .pointCloud:
            encoder.setRenderPipelineState(pointPipelineState)
            encoder.setVertexBuffer(pointCloud.buffer, offset: 0, index: 0)
            encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: pointCloud.count)
        }

        encoder.endEncoding()
        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
    }
}
