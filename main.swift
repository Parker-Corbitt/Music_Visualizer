import AppKit
import MetalKit
import Foundation
import UniformTypeIdentifiers

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

class PointCloudControlBar: NSView {
    private weak var renderer: Renderer?
    private let statusLabel = NSTextField(labelWithString: "No point cloud loaded")
    private let cameraLabel = NSTextField(labelWithString: "Camera: (0, 0, 0)")
    private lazy var renderStyleControl: NSSegmentedControl = {
        let control = NSSegmentedControl(
            labels: ["Points", "Mesh"],
            trackingMode: .selectOne,
            target: self,
            action: #selector(renderStyleChanged(_:))
        )
        control.selectedSegment = 0
        control.translatesAutoresizingMaskIntoConstraints = false
        return control
    }()
    private lazy var meshAlgorithmControl: NSSegmentedControl = {
        let control = NSSegmentedControl(
            labels: ["Tetra", "Dual"],
            trackingMode: .selectOne,
            target: self,
            action: #selector(meshAlgorithmChanged(_:))
        )
        control.selectedSegment = 0
        control.translatesAutoresizingMaskIntoConstraints = false
        control.isEnabled = false
        return control
    }()
    private let isovalueLabel = NSTextField(labelWithString: "Isovalue: 0.50")
    private lazy var isovalueSlider: NSSlider = {
        let slider = NSSlider(value: 0.5, minValue: 0.0, maxValue: 1.0, target: self, action: #selector(isovalueChanged(_:)))
        slider.isContinuous = true
        slider.isEnabled = false
        slider.allowsTickMarkValuesOnly = false
        slider.controlSize = .small
        slider.maxValue = 1.0
        slider.minValue = 0.0
        slider.numberOfTickMarks = 0
        slider.translatesAutoresizingMaskIntoConstraints = false
        return slider
    }()
    private lazy var modeControl: NSSegmentedControl = {
        let control = NSSegmentedControl(
            labels: ["Song", "Trend"],
            trackingMode: .selectOne,
            target: self,
            action: #selector(modeChanged(_:))
        )
        control.selectedSegment = 0
        control.translatesAutoresizingMaskIntoConstraints = false
        return control
    }()
    private var timer: Timer?

    init(renderer: Renderer) {
        self.renderer = renderer
        super.init(frame: .zero)
        translatesAutoresizingMaskIntoConstraints = false

        let openButton = NSButton(title: "Load Point Cloud…", target: self, action: #selector(openFile))
        openButton.bezelStyle = .rounded

        statusLabel.lineBreakMode = .byTruncatingMiddle
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        cameraLabel.lineBreakMode = .byTruncatingTail
        cameraLabel.alignment = .right
        cameraLabel.translatesAutoresizingMaskIntoConstraints = false
        isovalueLabel.alignment = .right
        isovalueLabel.translatesAutoresizingMaskIntoConstraints = false

        if let renderer = self.renderer {
            renderStyleControl.selectedSegment = segmentIndex(for: renderer.surfaceMode)
            isovalueSlider.doubleValue = Double(renderer.isovalue)
            isovalueSlider.isEnabled = renderer.surfaceMode != .pointCloud
            meshAlgorithmControl.selectedSegment = renderer.meshAlgorithm == .marchingTetrahedra ? 0 : 1
            meshAlgorithmControl.isEnabled = renderer.surfaceMode == .mesh
            updateIsovalueLabel(renderer.isovalue)
        }

        let stack = NSStackView(views: [
            modeControl,
            openButton,
            renderStyleControl,
            meshAlgorithmControl,
            isovalueLabel,
            isovalueSlider,
            statusLabel,
            NSView(),
            cameraLabel
        ])
        stack.translatesAutoresizingMaskIntoConstraints = false
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.spacing = 10
        addSubview(stack)

        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor),
            stack.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor),
            stack.topAnchor.constraint(equalTo: topAnchor),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor),
            statusLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 200),
            isovalueLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 90),
            isovalueSlider.widthAnchor.constraint(greaterThanOrEqualToConstant: 120),
            meshAlgorithmControl.widthAnchor.constraint(greaterThanOrEqualToConstant: 120),
            cameraLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 200)
        ])

        // Periodically refresh the camera position so the UI reflects user movement.
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.refreshCameraLabel()
        }
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func updateSelectedFileName(_ name: String?) {
        let modeText = (renderer?.currentMode == .trend) ? "Trend" : "Song"
        statusLabel.stringValue = "\(modeText): \(name ?? "No point cloud loaded")"
    }

    func setMode(_ mode: RenderMode) {
        modeControl.selectedSegment = (mode == .song) ? 0 : 1
        refreshStatusLabel()
    }

    private func refreshCameraLabel() {
        guard let pos = renderer?.camera.position else {
            cameraLabel.stringValue = "Camera: (n/a)"
            return
        }
        let text = String(format: "Camera: x=%.2f, y=%.2f, z=%.2f", pos.x, pos.y, pos.z)
        cameraLabel.stringValue = text
    }

    @objc private func openFile(_ sender: Any?) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if let binType = UTType(filenameExtension: "bin") {
            panel.allowedContentTypes = [binType]
        } else {
            panel.allowedContentTypes = [.data]
        }

        if panel.runModal() == .OK, let url = panel.url {
            if renderer?.loadSongPointCloud(at: url.path) == true {
                if renderer?.currentMode == .song {
                    updateSelectedFileName(url.lastPathComponent)
                }
            } else {
                NSSound.beep()
            }
        }
    }

    @objc private func modeChanged(_ sender: NSSegmentedControl) {
        let selectedMode: RenderMode = sender.selectedSegment == 0 ? .song : .trend
        guard renderer?.setMode(selectedMode) == true else {
            NSSound.beep()
            // Revert UI selection if mode change failed
            sender.selectedSegment = (renderer?.currentMode == .song) ? 0 : 1
            return
        }
        refreshStatusLabel()
    }

    @objc private func renderStyleChanged(_ sender: NSSegmentedControl) {
        guard let mode = surfaceMode(forSegment: sender.selectedSegment) else { return }
        renderer?.setSurfaceMode(mode)
        isovalueSlider.isEnabled = (mode != .pointCloud)
        meshAlgorithmControl.isEnabled = (mode != .pointCloud)
    }

    @objc private func meshAlgorithmChanged(_ sender: NSSegmentedControl) {
        let algorithm: MeshAlgorithm = (sender.selectedSegment == 0) ? .marchingTetrahedra : .dualContouring
        renderer?.setMeshAlgorithm(algorithm)
    }

    @objc private func isovalueChanged(_ sender: NSSlider) {
        let value = Float(sender.doubleValue)
        renderer?.isovalue = value
        updateIsovalueLabel(value)
    }

    private func updateIsovalueLabel(_ value: Float) {
        isovalueLabel.stringValue = String(format: "Isovalue: %.2f", value)
    }

    private func segmentIndex(for mode: SurfaceRenderMode) -> Int {
        switch mode {
        case .pointCloud: return 0
        case .mesh, .isosurface: return 1
        }
    }

    private func surfaceMode(forSegment segment: Int) -> SurfaceRenderMode? {
        switch segment {
        case 0: return .pointCloud
        case 1: return .mesh
        default: return nil
        }
    }

    private func refreshStatusLabel() {
        updateSelectedFileName(renderer?.activePointCloudName())
    }
}

let app = NSApplication.shared
let appDelegate = AppDelegate()
app.delegate = appDelegate
app.setActivationPolicy(.regular)

// Determine which screen to use: the one under the mouse cursor, or fall back to the main screen
let windowSize = NSSize(width: 800, height: 600)
let mouseLocation = NSEvent.mouseLocation
let targetScreen = NSScreen.screens.first { NSMouseInRect(mouseLocation, $0.frame, false) } ?? NSScreen.main

let screenFrame = targetScreen?.visibleFrame ?? NSScreen.main!.visibleFrame
let originX = screenFrame.origin.x + (screenFrame.size.width  - windowSize.width)  / 2
let originY = screenFrame.origin.y + (screenFrame.size.height - windowSize.height) / 2

let windowRect = NSRect(x: originX, y: originY, width: windowSize.width, height: windowSize.height)

// Create the window centered on the currently focused screen
let window = NSWindow(
    contentRect: windowRect,
    styleMask: [.titled, .closable, .resizable],
    backing: .buffered,
    defer: false
)

//Window Title
window.title = "music_visualizer"
window.makeKeyAndOrderFront(nil)

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("A fatal error as occurred")
}

let contentView = NSView(frame: window.contentView!.bounds)
contentView.autoresizingMask = [.width, .height]
window.contentView = contentView

let metalView = MTKView(frame: contentView.bounds, device: device)
metalView.translatesAutoresizingMaskIntoConstraints = false
contentView.addSubview(metalView)

let renderer = Renderer(view: metalView)
metalView.delegate = renderer

let controlBar = PointCloudControlBar(renderer: renderer)
contentView.addSubview(controlBar)

NSLayoutConstraint.activate([
    controlBar.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),
    controlBar.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 12),
    controlBar.trailingAnchor.constraint(lessThanOrEqualTo: contentView.trailingAnchor, constant: -12),

    metalView.topAnchor.constraint(equalTo: controlBar.bottomAnchor, constant: 8),
    metalView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
    metalView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
    metalView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor)
])

let initialName = renderer.loadInitialSongPointCloudIfAvailable()
renderer.setMode(.song)
controlBar.setMode(.song)
controlBar.updateSelectedFileName(renderer.activePointCloudName())

app.run()
