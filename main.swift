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
    private let debugLabel = NSTextField(labelWithString: "FPS: --  Voxels: --  Grid: --  Vertices: --")
    private lazy var renderStyleControl: NSSegmentedControl = {
        let control = NSSegmentedControl(
            labels: ["Points", "Mesh", "Volume"],
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
    private lazy var lightingButton: NSButton = {
        let button = NSButton(checkboxWithTitle: "Phong", target: self, action: #selector(lightingToggled(_:)))
        button.translatesAutoresizingMaskIntoConstraints = false
        button.state = .on
        return button
    }()
    private lazy var transferControl: NSSegmentedControl = {
        let control = NSSegmentedControl(
            labels: ["Classic", "Ember", "Glacial"],
            trackingMode: .selectOne,
            target: self,
            action: #selector(transferChanged(_:))
        )
        control.selectedSegment = 0
        control.translatesAutoresizingMaskIntoConstraints = false
        control.isEnabled = false
        return control
    }()
    private lazy var densitySlider: NSSlider = {
        let slider = NSSlider(value: 1.0, minValue: 0.1, maxValue: 3.0, target: self, action: #selector(densityChanged(_:)))
        slider.isContinuous = true
        slider.isEnabled = false
        slider.controlSize = .small
        slider.numberOfTickMarks = 0
        slider.translatesAutoresizingMaskIntoConstraints = false
        return slider
    }()
    private lazy var densityField: NSTextField = {
        let field = NSTextField(string: "1.00")
        field.alignment = .right
        field.controlSize = .small
        field.isEnabled = false
        field.isEditable = true
        field.isSelectable = true
        field.isBezeled = true
        if let cell = field.cell as? NSTextFieldCell {
            cell.sendsActionOnEndEditing = true
        }
        field.target = self
        field.action = #selector(densityFieldChanged(_:))
        field.translatesAutoresizingMaskIntoConstraints = false
        field.widthAnchor.constraint(equalToConstant: 50).isActive = true
        return field
    }()
    private lazy var opacitySlider: NSSlider = {
        let slider = NSSlider(value: 1.0, minValue: 0.05, maxValue: 1.5, target: self, action: #selector(opacityChanged(_:)))
        slider.isContinuous = true
        slider.isEnabled = false
        slider.controlSize = .small
        slider.numberOfTickMarks = 0
        slider.translatesAutoresizingMaskIntoConstraints = false
        return slider
    }()
    private lazy var opacityField: NSTextField = {
        let field = NSTextField(string: "1.00")
        field.alignment = .right
        field.controlSize = .small
        field.isEnabled = false
        field.isEditable = true
        field.isSelectable = true
        field.isBezeled = true
        if let cell = field.cell as? NSTextFieldCell {
            cell.sendsActionOnEndEditing = true
        }
        field.target = self
        field.action = #selector(opacityFieldChanged(_:))
        field.translatesAutoresizingMaskIntoConstraints = false
        field.widthAnchor.constraint(equalToConstant: 50).isActive = true
        return field
    }()
    private let densityLabel = NSTextField(labelWithString: "Density:")
    private let opacityLabel = NSTextField(labelWithString: "Opacity:")
    private let isovalueLabel = NSTextField(labelWithString: "Isovalue:")
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
    private lazy var isovalueField: NSTextField = {
        let field = NSTextField(string: "0.50")
        field.alignment = .right
        field.controlSize = .small
        field.isEnabled = false
        field.isEditable = true
        field.isSelectable = true
        field.isBezeled = true
        if let cell = field.cell as? NSTextFieldCell {
            cell.sendsActionOnEndEditing = true
        }
        field.target = self
        field.action = #selector(isovalueFieldChanged(_:))
        field.translatesAutoresizingMaskIntoConstraints = false
        field.widthAnchor.constraint(equalToConstant: 50).isActive = true
        return field
    }()
    private lazy var modeControl: NSSegmentedControl = {
        let control = NSSegmentedControl(
            labels: ["Song", "Timeline"],
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
        debugLabel.lineBreakMode = .byTruncatingTail
        debugLabel.alignment = .right
        debugLabel.translatesAutoresizingMaskIntoConstraints = false
        isovalueLabel.alignment = .right
        isovalueLabel.translatesAutoresizingMaskIntoConstraints = false

        if let renderer = self.renderer {
            renderStyleControl.selectedSegment = segmentIndex(for: renderer.surfaceMode)
            isovalueSlider.doubleValue = Double(renderer.isovalue)
            isovalueSlider.isEnabled = renderer.surfaceMode == .mesh
            meshAlgorithmControl.selectedSegment = renderer.meshAlgorithm == .marchingTetrahedra ? 0 : 1
            meshAlgorithmControl.isEnabled = renderer.surfaceMode == .mesh
            lightingButton.state = renderer.lightingEnabled ? .on : .off
            lightingButton.isEnabled = renderer.surfaceMode == .mesh
            transferControl.selectedSegment = segmentIndex(for: renderer.volumeTransferMode)
            transferControl.isEnabled = renderer.surfaceMode == .volume
            densitySlider.doubleValue = Double(renderer.densityScale)
            opacitySlider.doubleValue = Double(renderer.opacityScale)
            densitySlider.isEnabled = renderer.surfaceMode == .volume
            opacitySlider.isEnabled = renderer.surfaceMode == .volume
            updateDensityLabel(Float(densitySlider.doubleValue))
            updateOpacityLabel(Float(opacitySlider.doubleValue))
            updateIsovalueLabel(renderer.isovalue)
        }
        updateControlVisibility(for: renderer.surfaceMode)

        let mainRow = NSStackView(views: [
            modeControl,
            openButton,
            renderStyleControl,
            transferControl,
            densityLabel,
            densitySlider,
            densityField,
            opacityLabel,
            opacitySlider,
            opacityField,
            meshAlgorithmControl,
            lightingButton,
            isovalueLabel,
            isovalueSlider,
            isovalueField,
            statusLabel
        ])
        mainRow.translatesAutoresizingMaskIntoConstraints = false
        mainRow.orientation = .horizontal
        mainRow.alignment = .centerY
        mainRow.spacing = 10

        let debugRow = NSStackView(views: [debugLabel, cameraLabel])
        debugRow.translatesAutoresizingMaskIntoConstraints = false
        debugRow.orientation = .horizontal
        debugRow.alignment = .centerY
        debugRow.spacing = 10

        let column = NSStackView(views: [mainRow, debugRow])
        column.orientation = .vertical
        column.spacing = 6
        column.alignment = .leading
        column.translatesAutoresizingMaskIntoConstraints = false
        addSubview(column)

        NSLayoutConstraint.activate([
            column.leadingAnchor.constraint(equalTo: leadingAnchor),
            column.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor),
            column.topAnchor.constraint(equalTo: topAnchor),
            column.bottomAnchor.constraint(equalTo: bottomAnchor),
            statusLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 200),
            isovalueLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 90),
            isovalueSlider.widthAnchor.constraint(greaterThanOrEqualToConstant: 120),
            meshAlgorithmControl.widthAnchor.constraint(greaterThanOrEqualToConstant: 120),
            lightingButton.widthAnchor.constraint(greaterThanOrEqualToConstant: 70),
            transferControl.widthAnchor.constraint(greaterThanOrEqualToConstant: 180),
            densitySlider.widthAnchor.constraint(greaterThanOrEqualToConstant: 100),
            opacitySlider.widthAnchor.constraint(greaterThanOrEqualToConstant: 100),
            cameraLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 200),
            debugLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 260)
        ])

        // Periodically refresh the camera position so the UI reflects user movement.
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.refreshCameraLabel()
            self?.refreshDebugLabel()
        }
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func updateSelectedFileName(_ name: String?) {
        let modeText: String
        switch renderer?.currentMode {
        case .some(.timeline):
            modeText = "Timeline"
        default:
            modeText = "Song"
        }
        statusLabel.stringValue = "\(modeText): \(name ?? "No point cloud loaded")"
    }

    func setMode(_ mode: RenderMode) {
        modeControl.selectedSegment = segmentIndex(for: mode)
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

    private func refreshDebugLabel() {
        guard let stats = renderer?.debugStats() else {
            debugLabel.stringValue = "FPS: --  Voxels: --  Grid: --  Vertices: --"
            return
        }
        let fpsText = String(format: "%.1f", stats.fps)
        let voxels = stats.voxelCount
        let grid = stats.gridResolution
        let verts = stats.meshVertices
        let meshMs = stats.meshMs > 0 ? String(format: "%.1f ms", stats.meshMs) : "--"
        let volMs = stats.volumeMs > 0 ? String(format: "%.1f ms", stats.volumeMs) : "--"
        debugLabel.stringValue = "FPS: \(fpsText)  Voxels: \(voxels)  Grid: \(grid.x)x\(grid.y)x\(grid.z)  Vertices: \(verts)  Mesh: \(meshMs)  Volume: \(volMs)"
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
        guard let selectedMode = renderMode(for: sender.selectedSegment) else { return }
        guard renderer?.setMode(selectedMode) == true else {
            NSSound.beep()
            // Revert UI selection if mode change failed
            sender.selectedSegment = segmentIndex(for: renderer?.currentMode ?? .song)
            return
        }
        refreshStatusLabel()
    }

    @objc private func renderStyleChanged(_ sender: NSSegmentedControl) {
        guard let mode = surfaceMode(forSegment: sender.selectedSegment) else { return }
        renderer?.setSurfaceMode(mode)
        updateControlVisibility(for: mode)
    }

    @objc private func meshAlgorithmChanged(_ sender: NSSegmentedControl) {
        let algorithm: MeshAlgorithm = (sender.selectedSegment == 0) ? .marchingTetrahedra : .dualContouring
        renderer?.setMeshAlgorithm(algorithm)
    }

    @objc private func lightingToggled(_ sender: NSButton) {
        renderer?.setLightingEnabled(sender.state == .on)
    }

    @objc private func transferChanged(_ sender: NSSegmentedControl) {
        guard let mode = transferMode(forSegment: sender.selectedSegment) else { return }
        renderer?.setVolumeTransferMode(mode)
    }

    @objc private func densityChanged(_ sender: NSSlider) {
        let value = Float(sender.doubleValue)
        renderer?.setDensityScale(value)
        updateDensityLabel(value)
        densityField.stringValue = String(format: "%.2f", value)
    }

    @objc private func opacityChanged(_ sender: NSSlider) {
        let value = Float(sender.doubleValue)
        renderer?.setOpacityScale(value)
        updateOpacityLabel(value)
        opacityField.stringValue = String(format: "%.2f", value)
    }

    @objc private func isovalueChanged(_ sender: NSSlider) {
        let value = Float(sender.doubleValue)
        renderer?.isovalue = value
        updateIsovalueLabel(value)
        isovalueField.stringValue = String(format: "%.2f", value)
    }

    private func updateIsovalueLabel(_ value: Float) {
        isovalueLabel.stringValue = "Isovalue:"
        isovalueField.stringValue = String(format: "%.2f", value)
    }

    private func updateDensityLabel(_ value: Float) {
        densityLabel.stringValue = "Density:"
        densityField.stringValue = String(format: "%.2f", value)
    }

    private func updateOpacityLabel(_ value: Float) {
        opacityLabel.stringValue = "Opacity:"
        opacityField.stringValue = String(format: "%.2f", value)
    }

    @objc private func densityFieldChanged(_ sender: NSTextField) {
        let value = Float(sender.stringValue) ?? Float(densitySlider.doubleValue)
        let clamped = max(0.1, min(3.0, value))
        renderer?.setDensityScale(clamped)
        densitySlider.doubleValue = Double(clamped)
        updateDensityLabel(clamped)
        sender.stringValue = String(format: "%.2f", clamped)
        endEditing(sender)
    }

    @objc private func opacityFieldChanged(_ sender: NSTextField) {
        let value = Float(sender.stringValue) ?? Float(opacitySlider.doubleValue)
        let clamped = max(0.05, min(1.5, value))
        renderer?.setOpacityScale(clamped)
        opacitySlider.doubleValue = Double(clamped)
        updateOpacityLabel(clamped)
        sender.stringValue = String(format: "%.2f", clamped)
        endEditing(sender)
    }

    @objc private func isovalueFieldChanged(_ sender: NSTextField) {
        let value = Float(sender.stringValue) ?? renderer?.isovalue ?? 0.5
        let clamped = max(0.0, min(1.0, value))
        renderer?.isovalue = clamped
        isovalueSlider.doubleValue = Double(clamped)
        updateIsovalueLabel(clamped)
        sender.stringValue = String(format: "%.2f", clamped)
        endEditing(sender)
    }

    private func segmentIndex(for mode: SurfaceRenderMode) -> Int {
        switch mode {
        case .pointCloud: return 0
        case .mesh: return 1
        case .volume: return 2
        }
    }

    private func segmentIndex(for mode: VolumeTransferMode) -> Int {
        switch mode {
        case .classic: return 0
        case .ember: return 1
        case .glacial: return 2
        }
    }

    private func segmentIndex(for mode: RenderMode) -> Int {
        switch mode {
        case .song: return 0
        case .timeline: return 1
        }
    }

    private func renderMode(for segment: Int) -> RenderMode? {
        switch segment {
        case 0: return .song
        case 1: return .timeline
        default: return nil
        }
    }

    private func surfaceMode(forSegment segment: Int) -> SurfaceRenderMode? {
        switch segment {
        case 0: return .pointCloud
        case 1: return .mesh
        case 2: return .volume
        default: return nil
        }
    }

    private func transferMode(forSegment segment: Int) -> VolumeTransferMode? {
        switch segment {
        case 0: return .classic
        case 1: return .ember
        case 2: return .glacial
        default: return nil
        }
    }

    private func refreshStatusLabel() {
        updateSelectedFileName(renderer?.activePointCloudName())
    }

    private func updateControlVisibility(for mode: SurfaceRenderMode) {
        let meshVisible = (mode == .mesh)
        let volumeVisible = (mode == .volume)

        isovalueLabel.isHidden = !meshVisible
        isovalueSlider.isHidden = !meshVisible
        isovalueField.isHidden = !meshVisible
        meshAlgorithmControl.isHidden = !meshVisible
        lightingButton.isHidden = !meshVisible
        isovalueSlider.isEnabled = meshVisible
        isovalueField.isEnabled = meshVisible
        meshAlgorithmControl.isEnabled = meshVisible
        lightingButton.isEnabled = meshVisible

        transferControl.isHidden = !volumeVisible
        densityLabel.isHidden = !volumeVisible
        densitySlider.isHidden = !volumeVisible
        opacityLabel.isHidden = !volumeVisible
        opacitySlider.isHidden = !volumeVisible
        densityField.isHidden = !volumeVisible
        opacityField.isHidden = !volumeVisible
        transferControl.isEnabled = volumeVisible
        densitySlider.isEnabled = volumeVisible
        opacitySlider.isEnabled = volumeVisible
        densityField.isEnabled = volumeVisible
        opacityField.isEnabled = volumeVisible
    }

    private func endEditing(_ field: NSTextField) {
        field.window?.makeFirstResponder(nil)
    }
}

// Application Setup
let app = NSApplication.shared
let appDelegate = AppDelegate()
app.delegate = appDelegate
app.setActivationPolicy(.regular)

// Determine which screen to use: the one under the mouse cursor, or fall back to the main screen
let windowSize = NSSize(width: 800, height: 600)
let mouseLocation = NSEvent.mouseLocation
let targetScreen = NSScreen.screens.first { NSMouseInRect(mouseLocation, $0.frame, false) } ?? NSScreen.main

// Calculate centered window frame
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
window.title = "Music Visualizer"
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
