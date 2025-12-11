import simd

class Camera {
    var position: SIMD3<Float> = SIMD3<Float>(0, 2, 5)
    var rotation: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    
    var aspectRatio: Float = 1.0
    var fovDegrees: Float = 65
    var nearZ: Float = 0.1
    var farZ: Float = 100
    
    func viewMatrix() -> matrix_float4x4 {
        let rotationMatrix = matrix_float4x4(rotation: rotation)
        let translationMatrix = matrix_float4x4(translation: -position)
        return rotationMatrix * translationMatrix
    }
    
    func projectionMatrix() -> matrix_float4x4 {
        return matrix_float4x4(perspectiveWithFov: fovDegrees.radians,
                             aspectRatio: aspectRatio,
                             nearZ: nearZ,
                             farZ: farZ)
    }

    func reset() {
        position = SIMD3<Float>(0, 2, 5)
        rotation = SIMD3<Float>(0, 0, 0)
    }
}
