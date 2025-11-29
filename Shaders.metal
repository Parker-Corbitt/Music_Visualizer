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

struct MeshVertex {
    float4 position;
    float3 normal;
    float colorScalar;
};

struct VoxelGridInfo {
    uint3 resolution;
    uint _pad0;
    float3 origin;
    float spacing;
    uint pointCount;
    uint _pad1;
};

struct MarchingCubesParams {
    float isoValue;
    uint maxVertices;
    uint2 _pad;
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

// MARK: - Mesh rendering (lit)

struct MeshVOut {
    float4 position [[position]];
    float3 normal;
    float colorScalar;
};

vertex MeshVOut mesh_vertex_main(const device MeshVertex *vertices [[buffer(0)]],
                                 constant Uniforms &uniforms [[buffer(1)]],
                                 uint vid [[vertex_id]]) {
    MeshVOut out;
    float4 worldPos = uniforms.modelMatrix * vertices[vid].position;
    out.position = uniforms.viewProjectionMatrix * worldPos;
    out.normal = (uniforms.modelMatrix * float4(vertices[vid].normal, 0.0)).xyz;
    out.colorScalar = vertices[vid].colorScalar;
    return out;
}

fragment float4 mesh_fragment_main(MeshVOut in [[stage_in]]) {
    float3 lightDir = normalize(float3(0.4, 0.7, 1.0));
    float3 n = normalize(in.normal);
    float ndotl = clamp(dot(n, lightDir), 0.0, 1.0);
    float3 base = mix(float3(0.15, 0.25, 0.65), float3(0.95, 0.35, 0.2), clamp(in.colorScalar, 0.0, 1.0));
    float3 color = base * (0.25 + 0.75 * ndotl);
    return float4(color, 1.0);
}

// MARK: - Voxelization + marching tetrahedra

inline uint flatten3D(uint3 coord, uint3 res) {
    return (coord.z * res.y + coord.y) * res.x + coord.x;
}

inline float3 interpolateVertex(float3 p0, float3 p1, float v0, float v1, float iso) {
    float denom = (v1 - v0);
    float t = (abs(denom) < 1e-6) ? 0.5 : clamp((iso - v0) / denom, 0.0, 1.0);
    return mix(p0, p1, t);
}

kernel void clearGrid(device atomic_uint *counts [[buffer(0)]],
                      constant VoxelGridInfo &info [[buffer(1)]],
                      device atomic_uint *maxCount [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    uint totalCells = info.resolution.x * info.resolution.y * info.resolution.z;
    if (gid < totalCells) {
        atomic_store_explicit(&counts[gid], 0, memory_order_relaxed);
    }
    if (gid == 0) {
        atomic_store_explicit(maxCount, 0, memory_order_relaxed);
    }
}

kernel void splatPointsToGrid(const device VertexIn *points [[buffer(0)]],
                              constant VoxelGridInfo &info [[buffer(1)]],
                              device atomic_uint *counts [[buffer(2)]],
                              device atomic_uint *maxCount [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= info.pointCount) { return; }

    float3 pos = points[gid].position.xyz;
    float3 local = (pos - info.origin) / info.spacing;
    int3 cell = int3(floor(local + 0.5));
    if (cell.x < 0 || cell.y < 0 || cell.z < 0) { return; }
    uint3 ucell = uint3(cell);
    if (ucell.x >= info.resolution.x || ucell.y >= info.resolution.y || ucell.z >= info.resolution.z) {
        return;
    }

    uint idx = flatten3D(ucell, info.resolution);
    uint newCount = atomic_fetch_add_explicit(&counts[idx], 1u, memory_order_relaxed) + 1u;
    atomic_fetch_max_explicit(maxCount, newCount, memory_order_relaxed);
}

kernel void normalizeGrid(const device atomic_uint *counts [[buffer(0)]],
                          constant VoxelGridInfo &info [[buffer(1)]],
                          device float *scalarGrid [[buffer(2)]],
                          const device atomic_uint *maxCount [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    uint totalCells = info.resolution.x * info.resolution.y * info.resolution.z;
    if (gid >= totalCells) { return; }

    uint count = atomic_load_explicit(&counts[gid], memory_order_relaxed);
    uint maxVal = max(atomic_load_explicit(maxCount, memory_order_relaxed), 1u);
    scalarGrid[gid] = float(count) / float(maxVal);
}

constant ushort4 tetrahedra[6] = {
    ushort4(0, 5, 1, 6),
    ushort4(0, 1, 2, 6),
    ushort4(0, 2, 3, 6),
    ushort4(0, 3, 7, 6),
    ushort4(0, 7, 4, 6),
    ushort4(0, 4, 5, 6)
};

// Edges within a tetrahedron: (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)
constant short tetraTriTable[16][7] = {
    {-1, -1, -1, -1, -1, -1, -1},
    {0, 2, 3, -1, -1, -1, -1},
    {0, 1, 4, -1, -1, -1, -1},
    {1, 4, 2, 2, 4, 3, -1},
    {1, 2, 5, -1, -1, -1, -1},
    {0, 3, 5, 0, 5, 1, -1},
    {0, 2, 5, 0, 5, 4, -1},
    {5, 4, 3, -1, -1, -1, -1},
    {5, 4, 3, -1, -1, -1, -1},
    {0, 2, 5, 0, 5, 4, -1},
    {0, 3, 5, 0, 5, 1, -1},
    {1, 2, 5, -1, -1, -1, -1},
    {1, 4, 2, 2, 4, 3, -1},
    {0, 1, 4, -1, -1, -1, -1},
    {0, 2, 3, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1}
};

kernel void marchingTetrahedra(const device float *scalarGrid [[buffer(0)]],
                               constant VoxelGridInfo &info [[buffer(1)]],
                               constant MarchingCubesParams &params [[buffer(2)]],
                               device MeshVertex *meshVerts [[buffer(3)]],
                               device atomic_uint *vertexCounter [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    uint3 res = info.resolution;
    if (res.x < 2 || res.y < 2 || res.z < 2) { return; }

    uint cellsX = res.x - 1;
    uint cellsY = res.y - 1;
    uint cellsZ = res.z - 1;
    uint totalCells = cellsX * cellsY * cellsZ;
    if (gid >= totalCells) { return; }

    uint xy = cellsX * cellsY;
    uint cz = gid / xy;
    uint rem = gid - cz * xy;
    uint cy = rem / cellsX;
    uint cx = rem - cy * cellsX;

    // Corner indices for this cube
    uint3 base = uint3(cx, cy, cz);
    uint cornerIdx[8] = {
        flatten3D(base + uint3(0, 0, 0), res),
        flatten3D(base + uint3(1, 0, 0), res),
        flatten3D(base + uint3(0, 1, 0), res),
        flatten3D(base + uint3(1, 1, 0), res),
        flatten3D(base + uint3(0, 0, 1), res),
        flatten3D(base + uint3(1, 0, 1), res),
        flatten3D(base + uint3(0, 1, 1), res),
        flatten3D(base + uint3(1, 1, 1), res)
    };

    float values[8];
    float3 positions[8];
    for (uint i = 0; i < 8; ++i) {
        values[i] = scalarGrid[cornerIdx[i]];
        uint3 offs = uint3((i & 1) ? 1u : 0u, (i & 2) ? 1u : 0u, (i & 4) ? 1u : 0u);
        float3 corner = float3(base + offs);
        positions[i] = info.origin + corner * info.spacing;
    }

    for (uint t = 0; t < 6; ++t) {
        ushort4 tet = tetrahedra[t];
        float v0 = values[tet[0]];
        float v1 = values[tet[1]];
        float v2 = values[tet[2]];
        float v3 = values[tet[3]];

        uint mask = (v0 > params.isoValue) |
                    ((v1 > params.isoValue) << 1) |
                    ((v2 > params.isoValue) << 2) |
                    ((v3 > params.isoValue) << 3);
        constant short *tri = tetraTriTable[mask];
        if (tri[0] < 0) { continue; }

        float3 p0 = positions[tet[0]];
        float3 p1 = positions[tet[1]];
        float3 p2 = positions[tet[2]];
        float3 p3 = positions[tet[3]];

        float3 edgeVerts[6];
        edgeVerts[0] = interpolateVertex(p0, p1, v0, v1, params.isoValue);
        edgeVerts[1] = interpolateVertex(p1, p2, v1, v2, params.isoValue);
        edgeVerts[2] = interpolateVertex(p2, p0, v2, v0, params.isoValue);
        edgeVerts[3] = interpolateVertex(p0, p3, v0, v3, params.isoValue);
        edgeVerts[4] = interpolateVertex(p1, p3, v1, v3, params.isoValue);
        edgeVerts[5] = interpolateVertex(p2, p3, v2, v3, params.isoValue);

        float colorValue = clamp((v0 + v1 + v2 + v3) * 0.25, 0.0, 1.0);

        for (uint i = 0; i < 7; i += 3) {
            if (tri[i] < 0 || tri[i + 1] < 0 || tri[i + 2] < 0) { break; }

            float3 a = edgeVerts[tri[i]];
            float3 b = edgeVerts[tri[i + 1]];
            float3 c = edgeVerts[tri[i + 2]];
            float3 normal = normalize(cross(b - a, c - a));
            if (all(isfinite(normal)) == false || length(normal) < 1e-6) {
                normal = float3(0.0, 1.0, 0.0);
            }

            uint baseIndex = atomic_fetch_add_explicit(vertexCounter, 3u, memory_order_relaxed);
            if (baseIndex + 2u >= params.maxVertices) { return; }

            meshVerts[baseIndex + 0] = {float4(a, 1.0), normal, colorValue};
            meshVerts[baseIndex + 1] = {float4(b, 1.0), normal, colorValue};
            meshVerts[baseIndex + 2] = {float4(c, 1.0), normal, colorValue};
        }
    }
}
