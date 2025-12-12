import MetalKit
import AppKit
import simd

struct Uniforms {
    var viewProjectionMatrix: matrix_float4x4
    var modelMatrix: matrix_float4x4
    var inverseViewProjectionMatrix: matrix_float4x4
    var cameraPosition: SIMD3<Float>
    var lightingEnabled: UInt32 = 1
    var volumeTransferMode: UInt32 = 0
    var _padUniform: UInt32 = 0
    var densityScale: Float = 1.0
    var opacityScale: Float = 1.0
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
    let boundsMin: SIMD3<Float>
    let boundsMax: SIMD3<Float>
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
    case timeline
}

enum SurfaceRenderMode {
    case pointCloud
    case mesh
    case volume
}

enum VolumeTransferMode: UInt32 {
    case classic = 0
    case ember = 1
    case glacial = 2
}

enum MeshAlgorithm {
    case marchingTetrahedra
    case dualContouring
}

struct SongPointCloudLoader {
    static func loadPointCloud(device: MTLDevice, from path: String) -> (buffer: MTLBuffer?, count: Int, boundsMin: SIMD3<Float>, boundsMax: SIMD3<Float>) {
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url) else {
            print("[SongPointCloudLoader] Failed to read point cloud at path: \(path)")
            return (nil, 0, .zero, .zero)
        }

        let floatCount = data.count / MemoryLayout<Float>.size
        if floatCount % 5 != 0 {
            print("[SongPointCloudLoader] Unexpected float count \(floatCount); expected a multiple of 5.")
            return (nil, 0, .zero, .zero)
        }

        let vertexCount = floatCount / 5
        if vertexCount == 0 {
            print("[SongPointCloudLoader] No vertices found in file: \(path)")
            return (nil, 0, .zero, .zero)
        }

        var vertices: [PointVertex] = []
        vertices.reserveCapacity(vertexCount)
        var minPos = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var maxPos = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)

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
                let pos3 = SIMD3<Float>(x, y, z)
                minPos = simd.min(minPos, pos3)
                maxPos = simd.max(maxPos, pos3)
                vertices.append(PointVertex(position: position, scalar: scalar, hpRatio: hp))
            }
        }

        guard let buffer = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<PointVertex>.stride * vertices.count,
            options: []
        ) else {
            print("[SongPointCloudLoader] Failed to create MTLBuffer.")
            return (nil, 0, .zero, .zero)
        }

        return (buffer, vertices.count, minPos, maxPos)
    }
}

class Renderer: NSObject, MTKViewDelegate {

    // Metal objects
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pointPipelineState: MTLRenderPipelineState!
    var meshPipelineState: MTLRenderPipelineState!
    var volumePipelineState: MTLRenderPipelineState!
    var depthState: MTLDepthStencilState?
    var uniformBuffer: MTLBuffer!

    // Camera and input
    var camera: Camera
    var inputController: InputController!

    // Compute pipelines
    var gridClearPipeline: MTLComputePipelineState?
    var voxelSplatPipeline: MTLComputePipelineState?
    var gridNormalizePipeline: MTLComputePipelineState?
    var marchingTetraPipeline: MTLComputePipelineState?
    var dualCellPipeline: MTLComputePipelineState?
    var dualEdgeXPipeline: MTLComputePipelineState?
    var dualEdgeYPipeline: MTLComputePipelineState?
    var dualEdgeZPipeline: MTLComputePipelineState?

    // Point cloud resources
    private var activePointCloud: PointCloudResource?
    private var songPointCloud: PointCloudResource?
    private var timelinePointCloud: PointCloudResource?

    // Rendering modes
    private(set) var currentMode: RenderMode = .song
    private(set) var surfaceMode: SurfaceRenderMode = .pointCloud

    // Rendering options
    private(set) var lightingEnabled: Bool = true
    private(set) var meshAlgorithm: MeshAlgorithm = .marchingTetrahedra
    private(set) var volumeTransferMode: VolumeTransferMode = .classic
    private(set) var densityScale: Float = 1.0
    private(set) var opacityScale: Float = 1.0

    // Volume and mesh data
    private var meshNeedsRebuild = false
    private var scalarGridBuffer: MTLBuffer?
    private var gridLoudnessBuffer: MTLBuffer?
    private var gridMaxLoudnessBuffer: MTLBuffer?
    private var gridHpAccumBuffer: MTLBuffer?
    private var gridHpCountBuffer: MTLBuffer?
    private var gridHpBuffer: MTLBuffer?
    private var meshVertexBuffer: MTLBuffer?
    private var meshVertexCountBuffer: MTLBuffer?
    private var dualCellVertexBuffer: MTLBuffer?
    private var dualCellMaskBuffer: MTLBuffer?
    private var meshVertexCount: Int = 0
    private var volumeGridDirty = true
    private let songVoxelResolution = SIMD3<UInt32>(128, 128, 128)
    private let timelineVoxelResolution = SIMD3<UInt32>(224, 224, 224)
    private var voxelResolution = SIMD3<UInt32>(128, 128, 128)
    private let maxMeshVertices: Int = 1_200_000

    private var voxelCellCount: Int {
        Int(voxelResolution.x * voxelResolution.y * voxelResolution.z)
    }

    private var voxelCubeCount: Int {
        Int((voxelResolution.x - 1) * (voxelResolution.y - 1) * (voxelResolution.z - 1))
    }

    private var edgeXCount: Int {
        Int((voxelResolution.x - 1) * voxelResolution.y * voxelResolution.z)
    }

    private var edgeYCount: Int {
        Int(voxelResolution.x * (voxelResolution.y - 1) * voxelResolution.z)
    }

    private var edgeZCount: Int {
        Int(voxelResolution.x * voxelResolution.y * (voxelResolution.z - 1))
    }

    private struct MeshBuffers {
        let loudness: MTLBuffer
        let maxLoudness: MTLBuffer
        let hpAccum: MTLBuffer
        let hpCount: MTLBuffer
        let hpGrid: MTLBuffer
        let scalarGrid: MTLBuffer
        let mesh: MTLBuffer
        let meshCount: MTLBuffer
        let dualCells: MTLBuffer?
        let dualMask: MTLBuffer?
    }

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
            self.songPointCloud = PointCloudResource(buffer: buffer,
                                                     count: result.count,
                                                     name: name,
                                                     boundsMin: result.boundsMin,
                                                     boundsMax: result.boundsMax)
            refreshActivePointCloud()
            print("[Renderer] Loaded point cloud from: \(path) (vertices: \(result.count))")
            return true
        } else {
            print("[Renderer] Failed to load point cloud from: \(path)")
            return false
        }
    }

    /// Load the chronological timeline point cloud built from all songs.
    @discardableResult
    func loadTimelinePointCloud(at path: String = "data/processed/timeline_points.bin") -> Bool {
        let result = SongPointCloudLoader.loadPointCloud(device: device, from: path)
        if let buffer = result.buffer, result.count > 0 {
            let name = URL(fileURLWithPath: path).lastPathComponent
            self.timelinePointCloud = PointCloudResource(buffer: buffer,
                                                         count: result.count,
                                                         name: name,
                                                         boundsMin: result.boundsMin,
                                                         boundsMax: result.boundsMax)
            refreshActivePointCloud()
            print("[Renderer] Loaded timeline point cloud from: \(path) (vertices: \(result.count))")
            return true
        } else {
            print("[Renderer] Failed to load timeline point cloud from: \(path)")
            return false
        }
    }

    /// Switch rendering modes (song / timeline) and ensure buffers are loaded.
    /// Returns true if a buffer is ready for the chosen mode.
    @discardableResult
    func setMode(_ mode: RenderMode) -> Bool {
        currentMode = mode
        let targetRes = (mode == .timeline) ? timelineVoxelResolution : songVoxelResolution
        if targetRes != voxelResolution {
            voxelResolution = targetRes
            volumeGridDirty = true
            if surfaceMode == .mesh { meshNeedsRebuild = true }
        }
        ensurePointCloudLoaded(for: mode)
        refreshActivePointCloud()
        invalidateDerivedSurfaces()
        if mode == .timeline {
            centerCameraOnActiveData()
        }
        return activePointCloud != nil && activePointCloud!.count > 0
    }

    /// Switch the visualization output between points or meshes
    func setSurfaceMode(_ mode: SurfaceRenderMode) {
        surfaceMode = mode
        invalidateDerivedSurfaces()
    }

    func setLightingEnabled(_ enabled: Bool) {
        lightingEnabled = enabled
    }

    func setMeshAlgorithm(_ algorithm: MeshAlgorithm) {
        guard meshAlgorithm != algorithm else { return }
        meshAlgorithm = algorithm
        invalidateDerivedSurfaces()
    }

    func setVolumeTransferMode(_ mode: VolumeTransferMode) {
        volumeTransferMode = mode
    }

    func setDensityScale(_ scale: Float) {
        densityScale = max(0.0, scale)
    }

    func setOpacityScale(_ scale: Float) {
        opacityScale = max(0.0, scale)
    }

    private func ensurePointCloudLoaded(for mode: RenderMode) {
        switch mode {
        case .song:
            if songPointCloud == nil {
                _ = loadInitialSongPointCloudIfAvailable()
            }
        case .timeline:
            if timelinePointCloud == nil {
                _ = loadTimelinePointCloud()
            }
        }
    }

    /// Current active point cloud name for UI display.
    func activePointCloudName() -> String? {
        switch currentMode {
        case .song:
            return songPointCloud?.name
        case .timeline:
            return timelinePointCloud?.name
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

        var minPos = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var maxPos = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)

        for (i, pos) in positions.enumerated() {
            let scalar: Float = 1.0
            // Alternate hpRatio to give a small variation in fallback colors
            let hp: Float = (i % 2 == 0) ? 0.25 : 0.75
            minPos = simd.min(minPos, pos)
            maxPos = simd.max(maxPos, pos)
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
                name: "Procedural demo cloud",
                boundsMin: minPos,
                boundsMax: maxPos
            )
        } else {
            songPointCloud = nil
        }

        refreshActivePointCloud()
    }

    /// Ensure the active buffer matches the current mode.
    private func refreshActivePointCloud() {
        switch currentMode {
        case .song:
            activePointCloud = songPointCloud
        case .timeline:
            activePointCloud = timelinePointCloud
        }
        invalidateDerivedSurfaces()
    }

    private func invalidateDerivedSurfaces() {
        meshNeedsRebuild = (surfaceMode == .mesh)
        if meshNeedsRebuild {
            meshVertexCount = 0
        }
        volumeGridDirty = true
    }

    private func ensureVoxelResources(pointCount: Int) {
        let cubeCount = max(voxelCubeCount, 1)
        let scalarBytes = voxelCellCount * MemoryLayout<Float>.stride
        let countBytes = voxelCellCount * MemoryLayout<UInt32>.stride
        let meshBytes = maxMeshVertices * MemoryLayout<MeshVertex>.stride
        let dualBytes = cubeCount * MemoryLayout<MeshVertex>.stride
        let dualMaskBytes = cubeCount * MemoryLayout<UInt32>.stride

        if scalarGridBuffer == nil || scalarGridBuffer!.length < scalarBytes {
            scalarGridBuffer = device.makeBuffer(length: scalarBytes, options: .storageModeShared)
        }
        if gridLoudnessBuffer == nil || gridLoudnessBuffer!.length < countBytes {
            gridLoudnessBuffer = device.makeBuffer(length: countBytes, options: .storageModeShared)
        }
        if gridMaxLoudnessBuffer == nil {
            gridMaxLoudnessBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        }
        if gridHpAccumBuffer == nil || gridHpAccumBuffer!.length < countBytes {
            gridHpAccumBuffer = device.makeBuffer(length: countBytes, options: .storageModeShared)
        }
        if gridHpCountBuffer == nil || gridHpCountBuffer!.length < countBytes {
            gridHpCountBuffer = device.makeBuffer(length: countBytes, options: .storageModeShared)
        }
        if gridHpBuffer == nil || gridHpBuffer!.length < scalarBytes {
            gridHpBuffer = device.makeBuffer(length: scalarBytes, options: .storageModeShared)
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

    private func meshBuffers() -> MeshBuffers? {
        guard let loudness = gridLoudnessBuffer,
              let maxLoudness = gridMaxLoudnessBuffer,
              let hpAccum = gridHpAccumBuffer,
              let hpCount = gridHpCountBuffer,
              let hpGrid = gridHpBuffer,
              let scalarGrid = scalarGridBuffer,
              let mesh = meshVertexBuffer,
              let meshCount = meshVertexCountBuffer else {
            return nil
        }

        return MeshBuffers(
            loudness: loudness,
            maxLoudness: maxLoudness,
            hpAccum: hpAccum,
            hpCount: hpCount,
            hpGrid: hpGrid,
            scalarGrid: scalarGrid,
            mesh: mesh,
            meshCount: meshCount,
            dualCells: dualCellVertexBuffer,
            dualMask: dualCellMaskBuffer
        )
    }

    private func dispatchCompute(_ pipeline: MTLComputePipelineState,
                                 commandBuffer: MTLCommandBuffer,
                                 threads: Int,
                                 configure: (MTLComputeCommandEncoder) -> Void) {
        guard threads > 0 else { return }
        let tg = MTLSize(width: max(1, pipeline.threadExecutionWidth), height: 1, depth: 1)
        let grid = MTLSize(width: threads, height: 1, depth: 1)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(pipeline)
            configure(encoder)
            encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    private func makeGridInfo(for pointCloud: PointCloudResource) -> VoxelGridInfo {
        let minBounds = pointCloud.boundsMin
        let maxBounds = pointCloud.boundsMax
        let center = (minBounds + maxBounds) * 0.5
        let extent = maxBounds - minBounds
        let maxExtent = max(extent.x, max(extent.y, extent.z))
        let paddedExtent = max(maxExtent, 1e-3) * 1.02  // slight pad to avoid clipping at edges
        let halfExtent = paddedExtent * 0.5
        let origin = center - SIMD3<Float>(repeating: halfExtent)
        let maxSteps = max(voxelResolution.x, max(voxelResolution.y, voxelResolution.z))
        let spacing = paddedExtent / Float(max(Int(maxSteps) - 1, 1))
        return VoxelGridInfo(
            resolution: voxelResolution,
            origin: origin,
            spacing: spacing,
            pointCount: UInt32(min(pointCloud.count, Int(UInt32.max)))
        )
    }

    /// Position the camera to frame the active point cloud (used for timeline reset).
    func centerCameraOnActiveData() {
        guard let pc = activePointCloud else { return }
        let center = (pc.boundsMin + pc.boundsMax) * 0.5
        let extent = pc.boundsMax - pc.boundsMin
        let padding: Float = 0.05
        let halfExt = (extent * (1.0 + padding)) * 0.5
        let fovRad = camera.fovDegrees.radians
        let halfFovTan = tan(fovRad * 0.5)
        let aspect = max(camera.aspectRatio, 0.1)

        let distanceFitY = halfExt.y / max(halfFovTan, 1e-4)
        let distanceFitX = halfExt.x / max(halfFovTan * aspect, 1e-4)
        let distance = max(distanceFitX, distanceFitY) * 1.05

        // Raise slightly above center so we look down and keep Z in view.
        let pos = center + SIMD3<Float>(0, halfExt.y * 0.4, distance + halfExt.z)
        camera.position = pos
        let dir = simd_normalize(center - pos)
        let yaw = atan2(dir.x, -dir.z)
        let pitch = asin(dir.y)
        camera.rotation = SIMD3<Float>(pitch, yaw, 0)
    }

    private func makeIsoParams() -> IsoSurfaceParams {
        IsoSurfaceParams(
            isoValue: isovalue,
            maxVertices: UInt32(maxMeshVertices)
        )
    }

    private func resetMeshCounters(buffers: MeshBuffers, commandBuffer: MTLCommandBuffer) {
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: buffers.meshCount, range: 0..<MemoryLayout<UInt32>.stride, value: 0)
            if let mask = buffers.dualMask {
                blit.fill(buffer: mask, range: 0..<mask.length, value: 0)
            }
            blit.fill(buffer: buffers.maxLoudness, range: 0..<MemoryLayout<UInt32>.stride, value: 0)
            blit.fill(buffer: buffers.hpAccum, range: 0..<buffers.hpAccum.length, value: 0)
            blit.fill(buffer: buffers.hpCount, range: 0..<buffers.hpCount.length, value: 0)
            blit.endEncoding()
        }
    }

    private func prepareScalarGrid(pointCloud: PointCloudResource,
                                   gridInfo: inout VoxelGridInfo,
                                   buffers: MeshBuffers,
                                   commandBuffer: MTLCommandBuffer) -> Bool {
        guard let clearPipeline = gridClearPipeline,
              let splatPipeline = voxelSplatPipeline,
              let normalizePipeline = gridNormalizePipeline else {
            print("[Renderer] Compute pipelines missing; cannot build mesh.")
            return false
        }

        dispatchCompute(clearPipeline, commandBuffer: commandBuffer, threads: voxelCellCount) { encoder in
            encoder.setBuffer(buffers.loudness, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBuffer(buffers.maxLoudness, offset: 0, index: 2)
            encoder.setBuffer(buffers.hpAccum, offset: 0, index: 3)
            encoder.setBuffer(buffers.hpCount, offset: 0, index: 4)
        }

        dispatchCompute(splatPipeline, commandBuffer: commandBuffer, threads: pointCloud.count) { encoder in
            encoder.setBuffer(pointCloud.buffer, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBuffer(buffers.loudness, offset: 0, index: 2)
            encoder.setBuffer(buffers.maxLoudness, offset: 0, index: 3)
            encoder.setBuffer(buffers.hpAccum, offset: 0, index: 4)
            encoder.setBuffer(buffers.hpCount, offset: 0, index: 5)
        }

        dispatchCompute(normalizePipeline, commandBuffer: commandBuffer, threads: voxelCellCount) { encoder in
            encoder.setBuffer(buffers.loudness, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBuffer(buffers.scalarGrid, offset: 0, index: 2)
            encoder.setBuffer(buffers.maxLoudness, offset: 0, index: 3)
            encoder.setBuffer(buffers.hpAccum, offset: 0, index: 4)
            encoder.setBuffer(buffers.hpCount, offset: 0, index: 5)
            encoder.setBuffer(buffers.hpGrid, offset: 0, index: 6)
        }

        return true
    }

    private func runMarchingTetrahedra(gridInfo: inout VoxelGridInfo,
                                       isoParams: inout IsoSurfaceParams,
                                       buffers: MeshBuffers,
                                       commandBuffer: MTLCommandBuffer) -> Bool {
        guard let marchingPipeline = marchingTetraPipeline else {
            print("[Renderer] Marching tetrahedra pipeline missing.")
            return false
        }

        // Dispatch the marching tetrahedra compute shader
        dispatchCompute(marchingPipeline, commandBuffer: commandBuffer, threads: max(voxelCubeCount, 1)) { encoder in
            encoder.setBuffer(buffers.scalarGrid, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
            encoder.setBuffer(buffers.mesh, offset: 0, index: 3)
            encoder.setBuffer(buffers.meshCount, offset: 0, index: 4)
            encoder.setBuffer(buffers.hpGrid, offset: 0, index: 5)
        }
        return true
    }

    private func runDualContouring(gridInfo: inout VoxelGridInfo,
                                   isoParams: inout IsoSurfaceParams,
                                   buffers: MeshBuffers,
                                   commandBuffer: MTLCommandBuffer) -> Bool {
        guard let cellPipeline = dualCellPipeline,
              let edgeXPipe = dualEdgeXPipeline,
              let edgeYPipe = dualEdgeYPipeline,
              let edgeZPipe = dualEdgeZPipeline,
              let cellVerts = buffers.dualCells,
              let cellMask = buffers.dualMask else {
            print("[Renderer] Dual contouring pipelines missing; cannot build mesh.")
            return false
        }

        dispatchCompute(cellPipeline, commandBuffer: commandBuffer, threads: max(voxelCubeCount, 1)) { encoder in
            encoder.setBuffer(buffers.scalarGrid, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
            encoder.setBuffer(cellVerts, offset: 0, index: 3)
            encoder.setBuffer(cellMask, offset: 0, index: 4)
            encoder.setBuffer(buffers.hpGrid, offset: 0, index: 5)
        }

        dispatchCompute(edgeXPipe, commandBuffer: commandBuffer, threads: max(edgeXCount, 1)) { encoder in
            encoder.setBuffer(buffers.scalarGrid, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
            encoder.setBuffer(cellVerts, offset: 0, index: 3)
            encoder.setBuffer(cellMask, offset: 0, index: 4)
            encoder.setBuffer(buffers.mesh, offset: 0, index: 5)
            encoder.setBuffer(buffers.meshCount, offset: 0, index: 6)
        }

        dispatchCompute(edgeYPipe, commandBuffer: commandBuffer, threads: max(edgeYCount, 1)) { encoder in
            encoder.setBuffer(buffers.scalarGrid, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
            encoder.setBuffer(cellVerts, offset: 0, index: 3)
            encoder.setBuffer(cellMask, offset: 0, index: 4)
            encoder.setBuffer(buffers.mesh, offset: 0, index: 5)
            encoder.setBuffer(buffers.meshCount, offset: 0, index: 6)
        }

        dispatchCompute(edgeZPipe, commandBuffer: commandBuffer, threads: max(edgeZCount, 1)) { encoder in
            encoder.setBuffer(buffers.scalarGrid, offset: 0, index: 0)
            encoder.setBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 1)
            encoder.setBytes(&isoParams, length: MemoryLayout<IsoSurfaceParams>.stride, index: 2)
            encoder.setBuffer(cellVerts, offset: 0, index: 3)
            encoder.setBuffer(cellMask, offset: 0, index: 4)
            encoder.setBuffer(buffers.mesh, offset: 0, index: 5)
            encoder.setBuffer(buffers.meshCount, offset: 0, index: 6)
        }

        return true
    }

    private func rebuildVolumeGridIfNeeded() {
        guard volumeGridDirty, let pointCloud = activePointCloud, pointCloud.count > 0 else { return }
        ensureVoxelResources(pointCount: pointCloud.count)
        guard let buffers = meshBuffers(),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            volumeGridDirty = false
            return
        }
        var gridInfo = makeGridInfo(for: pointCloud)
        _ = prepareScalarGrid(pointCloud: pointCloud,
                              gridInfo: &gridInfo,
                              buffers: buffers,
                              commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        volumeGridDirty = false
    }


    private func rebuildMeshIfNeeded() {
        guard meshNeedsRebuild, surfaceMode == .mesh else { return }
        guard let pointCloud = activePointCloud, pointCloud.count > 0 else { return }
        buildMesh(from: pointCloud)
    }

    private func buildMesh(from pointCloud: PointCloudResource) {
        guard pointCloud.count > 0 else {
            meshVertexCount = 0
            meshNeedsRebuild = false
            return
        }

        ensureVoxelResources(pointCount: pointCloud.count)
        guard let buffers = meshBuffers(),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            meshNeedsRebuild = false
            return
        }

        var gridInfo = makeGridInfo(for: pointCloud)
        var isoParams = makeIsoParams()

        guard prepareScalarGrid(pointCloud: pointCloud,
                                gridInfo: &gridInfo,
                                buffers: buffers,
                                commandBuffer: commandBuffer) else {
            meshNeedsRebuild = false
            return
        }

        resetMeshCounters(buffers: buffers, commandBuffer: commandBuffer)

        let algorithmRan: Bool
        switch meshAlgorithm {
        case .marchingTetrahedra:
            algorithmRan = runMarchingTetrahedra(gridInfo: &gridInfo,
                                                 isoParams: &isoParams,
                                                 buffers: buffers,
                                                 commandBuffer: commandBuffer)
        case .dualContouring:
            algorithmRan = runDualContouring(gridInfo: &gridInfo,
                                             isoParams: &isoParams,
                                             buffers: buffers,
                                             commandBuffer: commandBuffer)
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if algorithmRan {
            let countPtr = buffers.meshCount.contents().assumingMemoryBound(to: UInt32.self)
            meshVertexCount = Int(min(countPtr.pointee, UInt32(maxMeshVertices)))
        } else {
            meshVertexCount = 0
        }
        meshNeedsRebuild = false
        volumeGridDirty = false
    }

    init(view: MTKView) {
        self.device = view.device!
        self.commandQueue = self.device.makeCommandQueue()!
        self.camera = Camera()
        super.init()

        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.2, green: 0.2, blue: 0.25, alpha: 1.0)
        createPipelines(view: view)
        inputController = InputController(view: view, camera: camera, renderer: self)
    }

    private func createPipelines(view: MTKView) {
        guard let library = device.makeDefaultLibrary() else { return }
        let pointVertexFunction = library.makeFunction(name: "vertex_main")
        let pointFragmentFunction = library.makeFunction(name: "fragment_main")
        let meshVertexFunction = library.makeFunction(name: "mesh_vertex_main")
        let meshFragmentFunction = library.makeFunction(name: "mesh_fragment_main")
        let volumeVertexFunction = library.makeFunction(name: "volume_vertex_main")
        let volumeFragmentFunction = library.makeFunction(name: "volume_fragment_main")

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

        let volumeDescriptor = MTLRenderPipelineDescriptor()
        volumeDescriptor.vertexFunction = volumeVertexFunction
        volumeDescriptor.fragmentFunction = volumeFragmentFunction
        volumeDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        volumeDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat

        do {
            pointPipelineState = try device.makeRenderPipelineState(descriptor: pointDescriptor)
            meshPipelineState = try device.makeRenderPipelineState(descriptor: meshDescriptor)
            volumePipelineState = try device.makeRenderPipelineState(descriptor: volumeDescriptor)
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

        let viewMatrix = camera.viewMatrix()
        let projectionMatrix = camera.projectionMatrix()
        let viewProjection = matrix_multiply(projectionMatrix, viewMatrix)
        let inverseViewProjection = simd_inverse(viewProjection)

        var uniforms = Uniforms(
            viewProjectionMatrix: viewProjection,
            modelMatrix: matrix_identity_float4x4,
            inverseViewProjectionMatrix: inverseViewProjection,
            cameraPosition: camera.position,
            lightingEnabled: lightingEnabled ? 1 : 0,
            volumeTransferMode: volumeTransferMode.rawValue,
            densityScale: densityScale,
            opacityScale: opacityScale
        )

        uniformBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<Uniforms>.size, options: [])

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let rpd = view.currentRenderPassDescriptor,
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd),
              let pointCloud = activePointCloud else { return }

        encoder.setDepthStencilState(depthState)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.setFrontFacing(.counterClockwise)
        let cull: MTLCullMode = (surfaceMode == .mesh && meshAlgorithm == .dualContouring) || surfaceMode == .volume ? .none : .back
        encoder.setCullMode(cull)

        if surfaceMode == .volume {
            rebuildVolumeGridIfNeeded()
        }

        switch surfaceMode {
        case .mesh:
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
        case .volume:
            guard let scalarGrid = scalarGridBuffer else { break }
            encoder.setRenderPipelineState(volumePipelineState)
            var gridInfo = makeGridInfo(for: pointCloud)
            encoder.setFragmentBytes(&gridInfo, length: MemoryLayout<VoxelGridInfo>.stride, index: 2)
            encoder.setFragmentBuffer(scalarGrid, offset: 0, index: 3)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }

        encoder.endEncoding()
        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
    }
}
