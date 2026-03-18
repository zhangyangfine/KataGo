#include <metal_stdlib>
using namespace metal;

// Porter-Duff "over" operator
static float4 over(float4 dst, float4 src) {
    float out_a = src.a + dst.a * (1.0 - src.a);
    if (out_a < 1e-6) return float4(0);
    float3 out_rgb = (src.rgb * src.a + dst.rgb * dst.a * (1.0 - src.a)) / out_a;
    return float4(out_rgb, out_a);
}

// Core stone algorithm in stone-local UV [0,1]²
static float4 drawStone(float2 uv, float3 stoneRGB, float stoneLength) {
    const float r_stone    = 0.5;
    const float div4_ratio = (1.0/4.0) / 0.95;   // ≈ 0.2632
    const float div8_ratio = (1.0/8.0) / 0.95;   // ≈ 0.1316
    const float div16_ratio = (1.0/16.0) / 0.95; // ≈ 0.0658
    const float div2_ratio  = (1.0/2.0) / 0.95;  // ≈ 0.5263

    float2 center = float2(0.5, 0.5);

    // Layer 1: Stone base (solid filled circle)
    float d_stone = length(uv - center);
    float aa = 0.5 / stoneLength;   // ≈ 1-pt antialiasing in UV space
    float base_alpha = smoothstep(r_stone + aa, r_stone - aa, d_stone);
    float4 base_color = float4(stoneRGB, base_alpha);

    // Layer 2a: Radial gradient light highlight
    // .offset(-div8_ratio) shifts visual center; symmetric .padding() only shrinks radius, not center
    float2 grad_center = center + float2(-div8_ratio, -div8_ratio);
    float grad_radius  = r_stone - div4_ratio;
    float d_grad = length(uv - grad_center);
    float t = clamp(d_grad / div4_ratio, 0.0, 1.0);
    float3 grad_rgb = mix(float3(1, 1, 1), stoneRGB, t);   // white → stoneColor
    float grad_alpha = d_grad <= grad_radius ? 1.0 : 0.0;
    float4 light_color = float4(grad_rgb, grad_alpha);

    // Layer 2b: Blurred overlay mask (suppresses excess highlight)
    float mask_radius = div2_ratio / 2.0;
    float d_mask  = d_stone;
    float sdf_mask = mask_radius - d_mask;
    float mask_alpha = smoothstep(-div16_ratio, div16_ratio, sdf_mask);
    float4 mask_color = float4(stoneRGB, mask_alpha);

    // Composite: transparent → base → light → mask
    float4 result = base_color;
    result = over(result, light_color);
    result = over(result, mask_color);
    return result;
}

// SwiftUI colorEffect entry point
// stoneLength:   stone bounding-box size in points (same unit as `position`)
// stoneRGB:      linear RGB stone color
[[stitchable]] half4 stone(float2 position,
                           half4 color,
                           float stoneLength,
                           float3 stoneRGB)
{
    float2 uv = position / stoneLength;
    float4 c = drawStone(uv, stoneRGB, stoneLength);
    c.rgb *= c.a;   // premultiply alpha so SwiftUI composites correctly
    return half4(c);
}
