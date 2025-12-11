#include <metal_stdlib>
using namespace metal;

// MARK: - Point rendering

// Input structure for point rendering
struct VertexIn {
    float4 position;
    float scalar;   // A-weighted loudness (0..1)
    float hpRatio;  // Harmonic-percussive ratio (0..1)
    float2 padding; // align to 32 bytes
};

// Output structure for point rendering
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

struct IsoSurfaceParams {
    float isoValue;
    uint maxVertices;
    uint2 _pad;
};

// Uniforms shared across shaders
struct Uniforms {
    float4x4 viewProjectionMatrix;
    float4x4 modelMatrix;
    float4x4 inverseViewProjectionMatrix;
    float3 cameraPosition;
    uint lightingEnabled;
};


vertex VertexOut vertex_main(const device VertexIn *vertices [[buffer(0)]],
                             constant Uniforms &uniforms [[buffer(1)]],
                             uint vid [[vertex_id]]) {
    VertexOut out;
    float4 worldPos = uniforms.modelMatrix * vertices[vid].position;
    out.position = uniforms.viewProjectionMatrix * worldPos;

    // Map hpRatio into a 5-stop gradient (0..0.2..1.0) from red to blue.
    const float3 palette[5] = {
        float3(0.70, 0.09, 0.17),  // red
        float3(0.94, 0.54, 0.38),  // orange
        float3(0.97, 0.97, 0.97),  // white 
        float3(0.40, 0.66, 0.81),  // light blue
        float3(0.13, 0.40, 0.67)   // deep blue
    };
    float hp = clamp(vertices[vid].hpRatio, 0.0, 1.0);
    float alpha = mix(0.4, 1.0, clamp(vertices[vid].scalar, 0.0, 1.0));
    uint band = uint(round(hp * 4.0));
    band = clamp(band, 0u, 4u);
    float3 baseColor = palette[band];
    out.color = float4(baseColor, alpha);

    // Set point size
    float baseSize = 2.5;
    out.pointSize = baseSize;
    
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}

// MARK: - Mesh rendering

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

fragment float4 mesh_fragment_main(MeshVOut in [[stage_in]],
                                   constant Uniforms &uniforms [[buffer(1)]]) {
    float3 lightDir = normalize(float3(0.4, 0.7, 1.0));
    float3 n = normalize(in.normal);
    float ndotl = clamp(dot(n, lightDir), 0.0, 1.0);
    float3 base = mix(float3(0.15, 0.25, 0.65), float3(0.95, 0.35, 0.2), clamp(in.colorScalar, 0.0, 1.0));
    if (uniforms.lightingEnabled == 0u) {
        return float4(base, 1.0);
    }
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

inline uint flattenCell3D(uint3 coord, uint3 res) {
    return (coord.z * res.y + coord.y) * res.x + coord.x;
}

kernel void clearGrid(device atomic_uint *loudnessAccum [[buffer(0)]],
                      constant VoxelGridInfo &info [[buffer(1)]],
                      device atomic_uint *maxAccum [[buffer(2)]],
                      device atomic_uint *hpAccum [[buffer(3)]],
                      device atomic_uint *hpCount [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    uint totalCells = info.resolution.x * info.resolution.y * info.resolution.z;
    if (gid < totalCells) {
        atomic_store_explicit(&loudnessAccum[gid], 0, memory_order_relaxed);
        atomic_store_explicit(&hpAccum[gid], 0, memory_order_relaxed);
        atomic_store_explicit(&hpCount[gid], 0, memory_order_relaxed);
    }
    if (gid == 0) {
        atomic_store_explicit(maxAccum, 0, memory_order_relaxed);
    }
}

kernel void splatPointsToGrid(const device VertexIn *points [[buffer(0)]],
                              constant VoxelGridInfo &info [[buffer(1)]],
                              device atomic_uint *loudnessAccum [[buffer(2)]],
                              device atomic_uint *maxAccum [[buffer(3)]],
                              device atomic_uint *hpAccum [[buffer(4)]],
                              device atomic_uint *hpCount [[buffer(5)]],
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
    float loudness = clamp(points[gid].scalar, 0.0, 1.0);
    const float scale = 65535.0;
    uint contribution = (uint)round(loudness * scale);
    uint newAccum = atomic_fetch_add_explicit(&loudnessAccum[idx], contribution, memory_order_relaxed) + contribution;
    atomic_fetch_max_explicit(maxAccum, newAccum, memory_order_relaxed);

    float hp = clamp(points[gid].hpRatio, 0.0, 1.0);
    uint hpContribution = (uint)round(hp * scale);
    atomic_fetch_add_explicit(&hpAccum[idx], hpContribution, memory_order_relaxed);
    atomic_fetch_add_explicit(&hpCount[idx], 1u, memory_order_relaxed);
}

kernel void normalizeGrid(const device atomic_uint *loudnessAccum [[buffer(0)]],
                          constant VoxelGridInfo &info [[buffer(1)]],
                          device float *scalarGrid [[buffer(2)]],
                          const device atomic_uint *maxAccum [[buffer(3)]],
                          const device atomic_uint *hpAccum [[buffer(4)]],
                          const device atomic_uint *hpCount [[buffer(5)]],
                          device float *hpGrid [[buffer(6)]],
                          uint gid [[thread_position_in_grid]]) {
    uint totalCells = info.resolution.x * info.resolution.y * info.resolution.z;
    if (gid >= totalCells) { return; }

    uint accum = atomic_load_explicit(&loudnessAccum[gid], memory_order_relaxed);
    uint maxVal = max(atomic_load_explicit(maxAccum, memory_order_relaxed), 1u);
    scalarGrid[gid] = float(accum) / float(maxVal);

    uint count = atomic_load_explicit(&hpCount[gid], memory_order_relaxed);
    uint hpSum = atomic_load_explicit(&hpAccum[gid], memory_order_relaxed);
    float hpAvg = (count > 0) ? (float(hpSum) / float(count) / 65535.0) : 0.0;
    hpGrid[gid] = clamp(hpAvg, 0.0, 1.0);
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
                               constant IsoSurfaceParams &params [[buffer(2)]],
                               device MeshVertex *meshVerts [[buffer(3)]],
                               device atomic_uint *vertexCounter [[buffer(4)]],
                               const device float *hpGrid [[buffer(5)]],
                               uint gid [[thread_position_in_grid]]) {
    uint3 res = info.resolution;
    if (res.x < 2 || res.y < 2 || res.z < 2) { return; }

    // Number of cells in each dimension
    uint cellsX = res.x - 1;
    uint cellsY = res.y - 1;
    uint cellsZ = res.z - 1;
    uint totalCells = cellsX * cellsY * cellsZ;
    if (gid >= totalCells) { return; }

    // 3D cell coordinates
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

    // Sample scalar values and positions at cube corners
    float values[8];
    float3 positions[8];
    for (uint i = 0; i < 8; ++i) {
        values[i] = scalarGrid[cornerIdx[i]];
        uint3 offs = uint3((i & 1) ? 1u : 0u, (i & 2) ? 1u : 0u, (i & 4) ? 1u : 0u);
        float3 corner = float3(base + offs);
        positions[i] = info.origin + corner * info.spacing;
    }

    // Process each of the 6 tetrahedra within the cube
    for (uint t = 0; t < 6; ++t) {
        ushort4 tet = tetrahedra[t];
        float v0 = values[tet[0]];
        float v1 = values[tet[1]];
        float v2 = values[tet[2]];
        float v3 = values[tet[3]];

        // Determine the case index
        uint mask = (v0 > params.isoValue) |
                    ((v1 > params.isoValue) << 1) |
                    ((v2 > params.isoValue) << 2) |
                    ((v3 > params.isoValue) << 3);
        constant short *tri = tetraTriTable[mask];
        if (tri[0] < 0) { continue; }

        // Interpolate edge vertices
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

        // Average hpRatio from tetrahedron corners for color
        float hp0 = hpGrid[cornerIdx[tet[0]]];
        float hp1 = hpGrid[cornerIdx[tet[1]]];
        float hp2 = hpGrid[cornerIdx[tet[2]]];
        float hp3 = hpGrid[cornerIdx[tet[3]]];
        float colorValue = clamp((hp0 + hp1 + hp2 + hp3) * 0.25, 0.0, 1.0);

        // Emit triangles
        for (uint i = 0; i < 7; i += 3) {
            if (tri[i] < 0 || tri[i + 1] < 0 || tri[i + 2] < 0) { break; }

            float3 a = edgeVerts[tri[i]];
            float3 b = edgeVerts[tri[i + 1]];
            float3 c = edgeVerts[tri[i + 2]];
            float3 normal = normalize(cross(b - a, c - a));
            if (all(isfinite(normal)) == false || length(normal) < 1e-6) {
                normal = float3(0.0, 1.0, 0.0);
            }

            // Reserve space for 3 vertices
            uint baseIndex = atomic_fetch_add_explicit(vertexCounter, 3u, memory_order_relaxed);
            if (baseIndex + 2u >= params.maxVertices) { return; }

            // Write out the triangle vertices
            meshVerts[baseIndex + 0] = {float4(a, 1.0), normal, colorValue};
            meshVerts[baseIndex + 1] = {float4(b, 1.0), normal, colorValue};
            meshVerts[baseIndex + 2] = {float4(c, 1.0), normal, colorValue};
        }
    }
}

// MARK: - Dual Contouring

inline float3 estimateCellNormal(const float values[8]) {
    float gx = (values[1] + values[3] + values[5] + values[7]) - (values[0] + values[2] + values[4] + values[6]);
    float gy = (values[2] + values[3] + values[6] + values[7]) - (values[0] + values[1] + values[4] + values[5]);
    float gz = (values[4] + values[5] + values[6] + values[7]) - (values[0] + values[1] + values[2] + values[3]);
    float3 n = float3(gx, gy, gz);
    if (all(isfinite(n)) && length(n) > 1e-6) {
        return normalize(n);
    }
    return float3(0.0, 1.0, 0.0);
}

constant ushort2 cubeEdges[12] = {
    ushort2(0, 1), ushort2(1, 3), ushort2(3, 2), ushort2(2, 0),
    ushort2(4, 5), ushort2(5, 7), ushort2(7, 6), ushort2(6, 4),
    ushort2(0, 4), ushort2(1, 5), ushort2(3, 7), ushort2(2, 6)
};

kernel void dualContourCells(const device float *scalarGrid [[buffer(0)]],
                             constant VoxelGridInfo &info [[buffer(1)]],
                             constant IsoSurfaceParams &params [[buffer(2)]],
                             device MeshVertex *cellVertices [[buffer(3)]],
                             device uint *cellMask [[buffer(4)]],
                             const device float *hpGrid [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    uint3 res = info.resolution;
    if (res.x < 2 || res.y < 2 || res.z < 2) { return; }

    // Number of cells in each dimension
    uint cellsX = res.x - 1;
    uint cellsY = res.y - 1;
    uint cellsZ = res.z - 1;
    uint cellCount = cellsX * cellsY * cellsZ;
    if (gid >= cellCount) { return; }

    // 3D cell coordinates
    uint cx = gid % cellsX;
    uint rem = gid / cellsX;
    uint cy = rem % cellsY;
    uint cz = rem / cellsY;

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

    // Sample scalar values and positions at cube corners
    float values[8];
    float3 positions[8];
    float minV = 0.0;
    float maxV = 0.0;
    for (uint i = 0; i < 8; ++i) {
        values[i] = scalarGrid[cornerIdx[i]];
        uint3 offs = uint3((i & 1) ? 1u : 0u, (i & 2) ? 1u : 0u, (i & 4) ? 1u : 0u);
        positions[i] = info.origin + float3(base + offs) * info.spacing;
        if (i == 0) {
            minV = maxV = values[i];
        } else {
            minV = min(minV, values[i]);
            maxV = max(maxV, values[i]);
        }
    }

    // Check if the isovalue is within the range of this cell
    if (!(minV < params.isoValue && maxV > params.isoValue)) {
        cellMask[gid] = 0u;
        return;
    }

    // Accumulate intersection points on edges
    float3 accum = float3(0.0);
    uint intersections = 0;
    for (uint e = 0; e < 12; ++e) {
        ushort2 edge = cubeEdges[e];
        float v0 = values[edge.x];
        float v1 = values[edge.y];
        float d0 = v0 - params.isoValue;
        float d1 = v1 - params.isoValue;
        if (d0 * d1 > 0.0) { continue; }
        float3 p = interpolateVertex(positions[edge.x], positions[edge.y], v0, v1, params.isoValue);
        accum += p;
        intersections += 1u;
    }

    // If no intersections found, mark cell as empty
    if (intersections == 0) {
        cellMask[gid] = 0u;
        return;
    }

    // Store the dual vertex for this cell
    cellMask[gid] = 1u;
    float3 pos = accum / float(intersections);
    float3 normal = estimateCellNormal(values);
    float hpSum = 0.0;
    for (uint i = 0; i < 8; ++i) {
        hpSum += hpGrid[cornerIdx[i]];
    }
    float colorValue = clamp(hpSum / 8.0, 0.0, 1.0);

    cellVertices[gid] = {float4(pos, 1.0), normal, colorValue};
}

kernel void dualContourEdgesX(const device float *scalarGrid [[buffer(0)]],
                              constant VoxelGridInfo &info [[buffer(1)]],
                              constant IsoSurfaceParams &params [[buffer(2)]],
                              const device MeshVertex *cellVertices [[buffer(3)]],
                              const device uint *cellMask [[buffer(4)]],
                              device MeshVertex *meshVerts [[buffer(5)]],
                              device atomic_uint *vertexCounter [[buffer(6)]],
                              uint gid [[thread_position_in_grid]]) {
    uint3 res = info.resolution;
    if (res.x < 2 || res.y < 2 || res.z < 2) { return; }

    uint strideX = res.x - 1;
    uint edgesX = strideX * res.y * res.z;
    if (gid >= edgesX) { return; }

    uint z = gid / (strideX * res.y);
    uint rem = gid - z * strideX * res.y;
    uint y = rem / strideX;
    uint x = rem - y * strideX;

    if (y == 0 || z == 0 || y >= res.y - 1 || z >= res.z - 1) { return; }

    float v0 = scalarGrid[flatten3D(uint3(x, y, z), res)];
    float v1 = scalarGrid[flatten3D(uint3(x + 1, y, z), res)];
    float d0 = v0 - params.isoValue;
    float d1 = v1 - params.isoValue;
    if (d0 * d1 >= 0.0) { return; }

    uint3 cellRes = uint3(res.x - 1, res.y - 1, res.z - 1);
    uint c0 = flattenCell3D(uint3(x,     y - 1, z - 1), cellRes);
    uint c1 = flattenCell3D(uint3(x,     y,     z - 1), cellRes);
    uint c2 = flattenCell3D(uint3(x,     y - 1, z    ), cellRes);
    uint c3 = flattenCell3D(uint3(x,     y,     z    ), cellRes);

    if (cellMask[c0] == 0 || cellMask[c1] == 0 || cellMask[c2] == 0 || cellMask[c3] == 0) { return; }

    MeshVertex vtx0 = cellVertices[c0];
    MeshVertex vtx1 = cellVertices[c1];
    MeshVertex vtx2 = cellVertices[c2];
    MeshVertex vtx3 = cellVertices[c3];

    uint base = atomic_fetch_add_explicit(vertexCounter, 6u, memory_order_relaxed);
    if (base + 5u >= params.maxVertices) { return; }

    meshVerts[base + 0] = vtx0;
    meshVerts[base + 1] = vtx1;
    meshVerts[base + 2] = vtx2;
    meshVerts[base + 3] = vtx2;
    meshVerts[base + 4] = vtx1;
    meshVerts[base + 5] = vtx3;
}

kernel void dualContourEdgesY(const device float *scalarGrid [[buffer(0)]],
                              constant VoxelGridInfo &info [[buffer(1)]],
                              constant IsoSurfaceParams &params [[buffer(2)]],
                              const device MeshVertex *cellVertices [[buffer(3)]],
                              const device uint *cellMask [[buffer(4)]],
                              device MeshVertex *meshVerts [[buffer(5)]],
                              device atomic_uint *vertexCounter [[buffer(6)]],
                              uint gid [[thread_position_in_grid]]) {
    uint3 res = info.resolution;
    if (res.x < 2 || res.y < 2 || res.z < 2) { return; }

    uint strideY = res.y - 1;
    uint edgesY = res.x * strideY * res.z;
    if (gid >= edgesY) { return; }

    uint z = gid / (res.x * strideY);
    uint rem = gid - z * res.x * strideY;
    uint x = rem / strideY;
    uint y = rem - x * strideY;

    if (x == 0 || z == 0 || x >= res.x - 1 || z >= res.z - 1) { return; }

    float v0 = scalarGrid[flatten3D(uint3(x, y, z), res)];
    float v1 = scalarGrid[flatten3D(uint3(x, y + 1, z), res)];
    float d0 = v0 - params.isoValue;
    float d1 = v1 - params.isoValue;
    if (d0 * d1 >= 0.0) { return; }

    uint3 cellRes = uint3(res.x - 1, res.y - 1, res.z - 1);
    uint c0 = flattenCell3D(uint3(x - 1, y,     z - 1), cellRes);
    uint c1 = flattenCell3D(uint3(x,     y,     z - 1), cellRes);
    uint c2 = flattenCell3D(uint3(x - 1, y,     z    ), cellRes);
    uint c3 = flattenCell3D(uint3(x,     y,     z    ), cellRes);

    if (cellMask[c0] == 0 || cellMask[c1] == 0 || cellMask[c2] == 0 || cellMask[c3] == 0) { return; }

    MeshVertex vtx0 = cellVertices[c0];
    MeshVertex vtx1 = cellVertices[c1];
    MeshVertex vtx2 = cellVertices[c2];
    MeshVertex vtx3 = cellVertices[c3];

    uint base = atomic_fetch_add_explicit(vertexCounter, 6u, memory_order_relaxed);
    if (base + 5u >= params.maxVertices) { return; }

    meshVerts[base + 0] = vtx0;
    meshVerts[base + 1] = vtx1;
    meshVerts[base + 2] = vtx2;
    meshVerts[base + 3] = vtx2;
    meshVerts[base + 4] = vtx1;
    meshVerts[base + 5] = vtx3;
}

kernel void dualContourEdgesZ(const device float *scalarGrid [[buffer(0)]],
                              constant VoxelGridInfo &info [[buffer(1)]],
                              constant IsoSurfaceParams &params [[buffer(2)]],
                              const device MeshVertex *cellVertices [[buffer(3)]],
                              const device uint *cellMask [[buffer(4)]],
                              device MeshVertex *meshVerts [[buffer(5)]],
                              device atomic_uint *vertexCounter [[buffer(6)]],
                              uint gid [[thread_position_in_grid]]) {
    uint3 res = info.resolution;
    if (res.x < 2 || res.y < 2 || res.z < 2) { return; }

    uint strideZ = res.z - 1;
    uint edgesZ = res.x * res.y * strideZ;
    if (gid >= edgesZ) { return; }

    uint z = gid / (res.x * res.y);
    uint rem = gid - z * res.x * res.y;
    uint y = rem / res.x;
    uint x = rem - y * res.x;

    if (x == 0 || y == 0 || x >= res.x - 1 || y >= res.y - 1) { return; }

    float v0 = scalarGrid[flatten3D(uint3(x, y, z), res)];
    float v1 = scalarGrid[flatten3D(uint3(x, y, z + 1), res)];
    float d0 = v0 - params.isoValue;
    float d1 = v1 - params.isoValue;
    if (d0 * d1 >= 0.0) { return; }

    uint3 cellRes = uint3(res.x - 1, res.y - 1, res.z - 1);
    uint c0 = flattenCell3D(uint3(x - 1, y - 1, z), cellRes);
    uint c1 = flattenCell3D(uint3(x,     y - 1, z), cellRes);
    uint c2 = flattenCell3D(uint3(x - 1, y,     z), cellRes);
    uint c3 = flattenCell3D(uint3(x,     y,     z), cellRes);

    if (cellMask[c0] == 0 || cellMask[c1] == 0 || cellMask[c2] == 0 || cellMask[c3] == 0) { return; }

    MeshVertex vtx0 = cellVertices[c0];
    MeshVertex vtx1 = cellVertices[c1];
    MeshVertex vtx2 = cellVertices[c2];
    MeshVertex vtx3 = cellVertices[c3];

    uint base = atomic_fetch_add_explicit(vertexCounter, 6u, memory_order_relaxed);
    if (base + 5u >= params.maxVertices) { return; }

    meshVerts[base + 0] = vtx0;
    meshVerts[base + 1] = vtx1;
    meshVerts[base + 2] = vtx2;
    meshVerts[base + 3] = vtx2;
    meshVerts[base + 4] = vtx1;
    meshVerts[base + 5] = vtx3;
}

// MARK: - Volume Ray Marching

struct VolumeVOut {
    float4 position [[position]];
    float2 uv;
};

vertex VolumeVOut volume_vertex_main(uint vid [[vertex_id]]) {
    const float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VolumeVOut out;
    float2 pos = positions[vid];
    out.position = float4(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    return out;
}

inline bool intersectAABB(float3 rayOrigin, float3 rayDir, float3 bmin, float3 bmax, thread float &tmin, thread float &tmax) {
    float3 invDir = 1.0 / rayDir;
    float3 t0s = (bmin - rayOrigin) * invDir;
    float3 t1s = (bmax - rayOrigin) * invDir;
    float3 tsmaller = min(t0s, t1s);
    float3 tbigger = max(t0s, t1s);
    tmin = max(max(tsmaller.x, tsmaller.y), max(tsmaller.z, 0.0));
    tmax = min(min(tbigger.x, tbigger.y), tbigger.z);
    return tmax >= tmin;
}

inline float sampleScalarGrid(const device float *grid,
                              constant VoxelGridInfo &info,
                              float3 worldPos) {
    float3 local = (worldPos - info.origin) / info.spacing;
    float3 maxCoord = float3(info.resolution) - 1.001;
    if (any(local < float3(0.0)) || any(local > maxCoord)) {
        return 0.0;
    }

    float3 clamped = clamp(local, float3(0.0), maxCoord);
    uint3 base = uint3(clamp(floor(clamped), float3(0.0), float3(info.resolution - 2)));
    float3 frac = clamped - float3(base);

    uint idx000 = flatten3D(base + uint3(0, 0, 0), info.resolution);
    uint idx100 = flatten3D(base + uint3(1, 0, 0), info.resolution);
    uint idx010 = flatten3D(base + uint3(0, 1, 0), info.resolution);
    uint idx110 = flatten3D(base + uint3(1, 1, 0), info.resolution);
    uint idx001 = flatten3D(base + uint3(0, 0, 1), info.resolution);
    uint idx101 = flatten3D(base + uint3(1, 0, 1), info.resolution);
    uint idx011 = flatten3D(base + uint3(0, 1, 1), info.resolution);
    uint idx111 = flatten3D(base + uint3(1, 1, 1), info.resolution);

    float c000 = grid[idx000];
    float c100 = grid[idx100];
    float c010 = grid[idx010];
    float c110 = grid[idx110];
    float c001 = grid[idx001];
    float c101 = grid[idx101];
    float c011 = grid[idx011];
    float c111 = grid[idx111];

    float c00 = mix(c000, c100, frac.x);
    float c10 = mix(c010, c110, frac.x);
    float c01 = mix(c001, c101, frac.x);
    float c11 = mix(c011, c111, frac.x);
    float c0 = mix(c00, c10, frac.y);
    float c1 = mix(c01, c11, frac.y);
    return mix(c0, c1, frac.z);
}

fragment float4 volume_fragment_main(VolumeVOut in [[stage_in]],
                                     constant Uniforms &uniforms [[buffer(1)]],
                                     constant VoxelGridInfo &info [[buffer(2)]],
                                     const device float *scalarGrid [[buffer(3)]]) {
    float2 ndc = float2(in.uv * 2.0 - 1.0);
    float4 clipNear = float4(ndc, -1.0, 1.0);
    float4 clipFar = float4(ndc, 1.0, 1.0);
    float4 worldNear = uniforms.inverseViewProjectionMatrix * clipNear;
    float4 worldFar = uniforms.inverseViewProjectionMatrix * clipFar;
    worldNear /= worldNear.w;
    worldFar /= worldFar.w;

    float3 rayOrigin = worldNear.xyz;
    float3 rayDir = normalize(worldFar.xyz - worldNear.xyz);

    float3 boxMin = info.origin;
    float3 boxMax = info.origin + (float3(info.resolution - 1) * info.spacing);
    float tmin, tmax;
    if (!intersectAABB(rayOrigin, rayDir, boxMin, boxMax, tmin, tmax)) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    const int kSteps = 96;
    float dt = (tmax - tmin) / float(kSteps);
    float t = tmin;

    float3 accumColor = float3(0.0);
    float accumAlpha = 0.0;
    const float3 colorLow = float3(0.1, 0.2, 0.4);
    const float3 colorHigh = float3(0.9, 0.45, 0.15);

    for (int i = 0; i < kSteps && accumAlpha < 0.98; ++i) {
        float3 samplePos = rayOrigin + rayDir * (t + 0.5 * dt);
        float density = clamp(sampleScalarGrid(scalarGrid, info, samplePos), 0.0, 1.0);
        float opacity = density * 0.15;
        float3 color = mix(colorLow, colorHigh, density);

        float oneMinusA = 1.0 - accumAlpha;
        accumColor += color * opacity * oneMinusA;
        accumAlpha += opacity * oneMinusA;
        t += dt;
    }

    return float4(accumColor, clamp(accumAlpha, 0.0, 1.0));
}
