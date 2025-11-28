import MetalKit
import AppKit
import simd

struct Uniforms {
    var viewProjectionMatrix: matrix_float4x4
    var modelMatrix: matrix_float4x4
}

struct PointVertex {
    var position: SIMD4<Float>
    var color: SIMD4<Float>
}

struct SongPointCloudLoader {
    /// Load a song-mode point cloud from a raw float32 binary file.
    ///
    /// The file layout must be:
    ///     [x, y, z, r, g, b, a] per vertex (all Float32),
    /// where x,y,z are already normalized into clip-space-ish coordinates,
    /// and r,g,b,a are color components in [0,1].
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
        if floatCount % 7 != 0 {
            print("[SongPointCloudLoader] Unexpected float count \(floatCount); expected a multiple of 7.")
            return (nil, 0)
        }

        let vertexCount = floatCount / 7
        if vertexCount == 0 {
            print("[SongPointCloudLoader] No vertices found in file: \(path)")
            return (nil, 0)
        }

        var vertices: [PointVertex] = []
        vertices.reserveCapacity(vertexCount)

        data.withUnsafeBytes { rawBuf in
            let buf = rawBuf.bindMemory(to: Float.self)
            var i = 0
            while i + 6 < floatCount {
                let x = buf[i + 0]
                let y = buf[i + 1]
                let z = buf[i + 2]
                let r = buf[i + 3]
                let g = buf[i + 4]
                let b = buf[i + 5]
                let a = buf[i + 6]
                i += 7

                let position = SIMD4<Float>(x, y, z, 1.0)
                let color    = SIMD4<Float>(r, g, b, a)
                vertices.append(PointVertex(position: position, color: color))
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
    var vertexBuffer: MTLBuffer!
    var vertexCount: Int = 0
    var currentPointCloudName: String?
    var uniformBuffer: MTLBuffer!
    var camera: Camera
    var inputController: InputController!

    /// Load a song-mode point cloud from disk and replace the current vertex buffer.
    /// This can be called at runtime to switch between different songs.
    @discardableResult
    func loadSongPointCloud(at path: String) -> Bool {
        let result = SongPointCloudLoader.loadPointCloud(device: device, from: path)
        if let buffer = result.buffer, result.count > 0 {
            self.vertexBuffer = buffer
            self.vertexCount = result.count
            self.currentPointCloudName = URL(fileURLWithPath: path).lastPathComponent
            print("[Renderer] Loaded point cloud from: \(path) (vertices: \(result.count))")
            return true
        } else {
            print("[Renderer] Failed to load point cloud from: \(path)")
            return false
        }
    }

    /// Attempt to find and load the first available .bin point cloud in common data folders.
    /// Falls back to a small built-in demo cloud so the renderer always has something to draw.
    func loadInitialPointCloudIfAvailable() -> String? {
        if let existing = currentPointCloudName { return existing }

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
        currentPointCloudName = "Procedural demo cloud"
        return currentPointCloudName
    }

    /// Build a small procedural cube of colored points as a safe default.
    private func makeProceduralFallbackCloud() {
        var verts: [PointVertex] = []
        let colors: [SIMD4<Float>] = [
            SIMD4<Float>(1, 0, 0, 1),
            SIMD4<Float>(0, 1, 0, 1),
            SIMD4<Float>(0, 0, 1, 1),
            SIMD4<Float>(1, 1, 0, 1),
            SIMD4<Float>(1, 0, 1, 1),
            SIMD4<Float>(0, 1, 1, 1)
        ]

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
            let color = colors[i % colors.count]
            verts.append(PointVertex(position: SIMD4<Float>(pos, 1.0), color: color))
        }

        if let buffer = device.makeBuffer(
            bytes: verts,
            length: MemoryLayout<PointVertex>.stride * verts.count,
            options: []
        ) {
            vertexBuffer = buffer
            vertexCount = verts.count
        } else {
            vertexBuffer = nil
            vertexCount = 0
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
              let vertexBuffer = vertexBuffer,
              vertexCount > 0 else { return }

        encoder.setRenderPipelineState(pipelineState)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: vertexCount)
        encoder.endEncoding()
        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
    }
}
