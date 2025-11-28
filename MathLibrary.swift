import simd
import Foundation

extension Float {
    var radians: Float {
        return self * .pi / 180
    }
}

extension matrix_float4x4 {
    init(translation: SIMD3<Float>) {
        self.init(
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(translation.x, translation.y, translation.z, 1)
        )
    }
    
    init(rotationX: Float) {
        let c = cos(rotationX)
        let s = sin(rotationX)
        self.init(
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, c, s, 0),
            SIMD4<Float>(0, -s, c, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
    }
    
    init(rotationY: Float) {
        let c = cos(rotationY)
        let s = sin(rotationY)
        self.init(
            SIMD4<Float>(c, 0, -s, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(s, 0, c, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
    }
    
    init(rotationZ: Float) {
        let c = cos(rotationZ)
        let s = sin(rotationZ)
        self.init(
            SIMD4<Float>(c, s, 0, 0),
            SIMD4<Float>(-s, c, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
    }
    
    init(rotation: SIMD3<Float>) {
        let rx = matrix_float4x4(rotationX: rotation.x)
        let ry = matrix_float4x4(rotationY: rotation.y)
        let rz = matrix_float4x4(rotationZ: rotation.z)
        // Note: rotation order Y * X * Z (matches common camera conventions)
        self = ry * rx * rz
    }
    
    init(perspectiveWithFov fovyRadians: Float, aspectRatio: Float, nearZ: Float, farZ: Float) {
        let yScale = 1 / tan(fovyRadians * 0.5)
        let xScale = yScale / aspectRatio
        let zRange = farZ - nearZ
        let zScale = -(farZ + nearZ) / zRange
        let wz = -2 * farZ * nearZ / zRange

        self.init(
            SIMD4<Float>( xScale,  0,      0,   0),
            SIMD4<Float>( 0,     yScale,   0,   0),
            SIMD4<Float>( 0,       0,  zScale, -1),
            SIMD4<Float>( 0,       0,   wz,    0)
        )
    }
}