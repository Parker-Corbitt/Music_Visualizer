import MetalKit
import AppKit
import simd

class InputController {

    private weak var view: MTKView?
    private var camera: Camera

    private var keyboardHandler: Any?
    private var mouseHandler: Any?

    private var lastTime: TimeInterval = CACurrentMediaTime()
    private var moveSpeed: Float
    private var rotationSpeed: Float

    private var moveDirection = SIMD3<Float>(0, 0, 0)
    private var lastMousePosition: CGPoint = .zero
    private var isDragging: Bool = false
    private var pressedKeys = Set<String>()

    init(view: MTKView, camera: Camera, moveSpeed: Float = 2.0, rotationSpeed: Float = 0.005) {
        self.view = view
        self.camera = camera
        self.moveSpeed = moveSpeed
        self.rotationSpeed = rotationSpeed
        setupInputHandlers()
    }

    deinit {
        if let kh = keyboardHandler { NSEvent.removeMonitor(kh) }
        if let mh = mouseHandler { NSEvent.removeMonitor(mh) }
    }

    func update() {
        let currentTime = CACurrentMediaTime()
        let deltaTime = Float(currentTime - lastTime)
        lastTime = currentTime

        let yaw = camera.rotation.y
        let pitch = camera.rotation.x

        // Forward vector from yaw/pitch (right-handed, Y-up, looking down -Z at yaw = 0)
        let forward = normalize(SIMD3<Float>(
            sin(yaw) * cos(pitch),
            sin(pitch),
            -cos(yaw) * cos(pitch)
        ))

        let worldUp = SIMD3<Float>(0, 1, 0)
        let right = normalize(cross(forward, worldUp))
        let up = cross(right, forward)

        let movement = forward * moveDirection.z +
                       right * moveDirection.x +
                       up * moveDirection.y

        camera.position += movement * moveSpeed * deltaTime
    }

    // MARK: - Input Handling

    private func setupInputHandlers() {
        // Keyboard
        keyboardHandler = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] ev in
            guard let self = self else { return ev }
            switch ev.type {
            case .keyDown:
                self.handleKeyDown(ev)
                return nil
            case .keyUp:
                self.handleKeyUp(ev)
                return nil
            default:
                break
            }
            return ev
        }

        // Mouse (drag to rotate, scroll to zoom)
        mouseHandler = NSEvent.addLocalMonitorForEvents(
            matching: [.leftMouseDown, .leftMouseUp, .leftMouseDragged, .scrollWheel]
        ) { [weak self] ev in
            guard let self = self else { return ev }
            switch ev.type {
            case .leftMouseDown:
                self.isDragging = true
                self.lastMousePosition = ev.locationInWindow
            case .leftMouseUp:
                self.isDragging = false
            case .leftMouseDragged:
                guard self.isDragging else { break }
                let loc = ev.locationInWindow
                let dx = Float(loc.x - self.lastMousePosition.x)
                let dy = Float(loc.y - self.lastMousePosition.y)
                self.lastMousePosition = loc
                self.camera.rotation.y += dx * self.rotationSpeed
                self.camera.rotation.x += dy * self.rotationSpeed
            case .scrollWheel:
                self.camera.position.z += Float(ev.scrollingDeltaY) * 0.05
            default:
                break
            }
            return ev
        }
    }

    private func handleKeyDown(_ event: NSEvent) {
        if event.isARepeat { return }
        guard let chars = event.charactersIgnoringModifiers?.lowercased(),
              let ch = chars.first else { return }
        let key = String(ch)

        pressedKeys.insert(key)
        updateMoveDirectionFromKeys()
    }

    private func handleKeyUp(_ event: NSEvent) {
        guard let chars = event.charactersIgnoringModifiers?.lowercased(),
              let ch = chars.first else { return }
        let key = String(ch)

        pressedKeys.remove(key)
        updateMoveDirectionFromKeys()
    }

    private func updateMoveDirectionFromKeys() {
        var dir = SIMD3<Float>(0, 0, 0)

        // Forward/back
        let forward: Float = pressedKeys.contains("w") ? 1 : 0
        let back: Float    = pressedKeys.contains("s") ? -1 : 0
        dir.z = forward + back

        // Left/right
        let left: Float  = pressedKeys.contains("a") ? -1 : 0
        let right: Float = pressedKeys.contains("d") ? 1 : 0
        dir.x = left + right

        // Down/up
        let down: Float = pressedKeys.contains("q") ? -1 : 0
        let up: Float   = pressedKeys.contains("e") ? 1 : 0
        dir.y = down + up

        moveDirection = dir
    }
}