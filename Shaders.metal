#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float4 position;
    float4 color;
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
    out.color = vertices[vid].color;
    out.pointSize = 4.0; // Small but visible point size for all vertices
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}
