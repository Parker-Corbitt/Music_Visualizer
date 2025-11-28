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
    var hpRatio: Float      // Harmonic-percussive ratio (0..1)
    var padding: SIMD2<Float> = .zero  // pad to 32 bytes for alignment
}

enum RenderMode {
    case song
    case trend
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
    var pipelineState: MTLRenderPipelineState!
    private var activeVertexBuffer: MTLBuffer?
    private var activeVertexCount: Int = 0

    private var songVertexBuffer: MTLBuffer?
    private var songVertexCount: Int = 0
    private var songPointCloudName: String?

    private var trendVertexBuffer: MTLBuffer?
    private var trendVertexCount: Int = 0
    private var trendPointCloudName: String?

    private(set) var currentMode: RenderMode = .song
    var uniformBuffer: MTLBuffer!
    var camera: Camera
    var inputController: InputController!

    /// Load a song-mode point cloud from disk and replace the current vertex buffer.
    /// This can be called at runtime to switch between different songs.
    @discardableResult
    func loadSongPointCloud(at path: String) -> Bool {
        let result = SongPointCloudLoader.loadPointCloud(device: device, from: path)
        if let buffer = result.buffer, result.count > 0 {
            self.songVertexBuffer = buffer
            self.songVertexCount = result.count
            self.songPointCloudName = URL(fileURLWithPath: path).lastPathComponent
            refreshActiveBuffersIfNeeded()
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
            self.trendVertexBuffer = buffer
            self.trendVertexCount = result.count
            self.trendPointCloudName = URL(fileURLWithPath: path).lastPathComponent
            refreshActiveBuffersIfNeeded()
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
        switch mode {
        case .song:
            if songVertexBuffer == nil {
                _ = loadInitialSongPointCloudIfAvailable()
            }
        case .trend:
            if trendVertexBuffer == nil {
                _ = loadTrendPointCloud()
            }
        }
        refreshActiveBuffersIfNeeded()
        return activeVertexBuffer != nil && activeVertexCount > 0
    }

    /// Current active point cloud name for UI display.
    func activePointCloudName() -> String? {
        switch currentMode {
        case .song:
            return songPointCloudName
        case .trend:
            return trendPointCloudName
        }
    }

    /// Attempt to find and load the first available .bin point cloud in common data folders.
    /// Falls back to a small built-in demo cloud so the renderer always has something to draw (song mode only).
    func loadInitialSongPointCloudIfAvailable() -> String? {
        if let existing = songPointCloudName { return existing }

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
        songPointCloudName = "Procedural demo cloud"
        return songPointCloudName
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
            songVertexBuffer = buffer
            songVertexCount = verts.count
        } else {
            songVertexBuffer = nil
            songVertexCount = 0
        }

        refreshActiveBuffersIfNeeded()
    }

    /// Ensure the active buffer matches the current mode.
    private func refreshActiveBuffersIfNeeded() {
        switch currentMode {
        case .song:
            activeVertexBuffer = songVertexBuffer
            activeVertexCount = songVertexCount
        case .trend:
            activeVertexBuffer = trendVertexBuffer
            activeVertexCount = trendVertexCount
        }
    }

    init(view: MTKView) {
        self.device = view.device!
        self.commandQueue = self.device.makeCommandQueue()!
        self.camera = Camera()
        super.init()

        view.clearColor = MTLClearColor(red: 0.2, green: 0.2, blue: 0.25, alpha: 1.0)
        createPipelineState(view: view)
        inputController = InputController(view: view, camera: camera)
    }

    private func createPipelineState(view: MTKView) {
        guard let library = device.makeDefaultLibrary() else { return }
        let vertexFunction = library.makeFunction(name: "vertex_main")
        let fragmentFunction = library.makeFunction(name: "fragment_main")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat

        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create pipeline state: \(error)")
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.aspectRatio = Float(size.width / size.height)
    }

    func draw(in view: MTKView) {
        inputController.update()

        var uniforms = Uniforms(
            viewProjectionMatrix: matrix_multiply(camera.projectionMatrix(), camera.viewMatrix()),
            modelMatrix: matrix_identity_float4x4
        )

        uniformBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<Uniforms>.size, options: [])

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let rpd = view.currentRenderPassDescriptor,
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd),
              let vertexBuffer = activeVertexBuffer,
              activeVertexCount > 0 else { return }

        encoder.setRenderPipelineState(pipelineState)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: activeVertexCount)
        encoder.endEncoding()
        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
    }
}
