import AppKit
import MetalKit
import Foundation

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

class PointCloudControlBar: NSView {
    private weak var renderer: Renderer?
    private let statusLabel = NSTextField(labelWithString: "No point cloud loaded")
    private let cameraLabel = NSTextField(labelWithString: "Camera: (0, 0, 0)")
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

        let spacer = NSView()
        spacer.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)

        let stack = NSStackView(views: [openButton, statusLabel, spacer, cameraLabel])
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
        statusLabel.stringValue = name ?? "No point cloud loaded"
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
        panel.allowedFileTypes = ["bin"]

        if panel.runModal() == .OK, let url = panel.url {
            if renderer?.loadSongPointCloud(at: url.path) == true {
                updateSelectedFileName(url.lastPathComponent)
            } else {
                NSSound.beep()
            }
        }
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

let initialName = renderer.loadInitialPointCloudIfAvailable()
controlBar.updateSelectedFileName(initialName)

app.run()
