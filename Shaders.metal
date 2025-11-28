#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float4 position;
    float scalar;   // A-weighted loudness (0..1)
    float hpRatio;  // Harmonic-percussive ratio (0..1)
    float2 padding; // align to 32 bytes
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float pointSize [[point_size]];
};

struct Uniforms {
    float4x4 viewProjectionMatrix;
    float4x4 modelMatrix;
};

vertex VertexOut vertex_main(const device VertexIn *vertices [[buffer(0)]],
                             constant Uniforms &uniforms [[buffer(1)]],
                             uint vid [[vertex_id]]) {
    VertexOut out;
    float4 worldPos = uniforms.modelMatrix * vertices[vid].position;
    out.position = uniforms.viewProjectionMatrix * worldPos;

    // Color is driven solely by hpRatio: red = harmonic, blue = percussive.
    float hp = clamp(vertices[vid].hpRatio, 0.0, 1.0);
    float r = hp;
    float g = 0.15;
    float b = 1.0 - hp;
    float alpha = mix(0.4, 1.0, clamp(vertices[vid].scalar, 0.0, 1.0));
    out.color = float4(r, g, b, alpha);

    // Use scalar to modestly influence point size so the field reflects loudness.
    float baseSize = 2.5;
    float sizeBoost = 6.0 * clamp(vertices[vid].scalar, 0.0, 1.0);
    out.pointSize = baseSize + sizeBoost;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}
