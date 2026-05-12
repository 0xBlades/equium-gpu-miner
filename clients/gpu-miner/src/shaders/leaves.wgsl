// Equihash 96,5 leaf-generation kernel — compact G-function edition.
//
// One invocation = one BLAKE2b-512 call → 5 leaves × 12 bytes each
// (n=96, k=5, digest_length=60, indices_per_hash=5).
//
// Root cause of previous segfault (NVIDIA driver 550.x):
//   libnvidia-glvkspirv.so crashed when compiling a SPIR-V function that
//   had ~900 inlined operations (the original fully-unrolled main()).
//
// Fix: extract blake2b_g() as a separate WGSL function.  naga emits it as
// a proper SPIR-V OpFunction, so main() becomes 96 call sites instead of
// 768 inlined instructions.  Byte output is identical — all 12 tests pass.
//
// SIGMA permutations are still written as literal block[N] at every call
// site (no dynamic array indexing), which was the original reason for
// unrolling.  The function boundary alone is enough to fix the driver bug.

// ---- u64 = vec2<u32> arithmetic ----

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry;
    return vec2<u32>(lo, hi);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn u64_rotr(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 32u) {
        return vec2<u32>(a.y, a.x);
    }
    if (n < 32u) {
        let lo = (a.x >> n) | (a.y << (32u - n));
        let hi = (a.y >> n) | (a.x << (32u - n));
        return vec2<u32>(lo, hi);
    }
    let m = n - 32u;
    let swapped = vec2<u32>(a.y, a.x);
    let lo = (swapped.x >> m) | (swapped.y << (32u - m));
    let hi = (swapped.y >> m) | (swapped.x << (32u - m));
    return vec2<u32>(lo, hi);
}

// BLAKE2b initialization vector.
const IV0: vec2<u32> = vec2<u32>(0xf3bcc908u, 0x6a09e667u);
const IV1: vec2<u32> = vec2<u32>(0x84caa73bu, 0xbb67ae85u);
const IV2: vec2<u32> = vec2<u32>(0xfe94f82bu, 0x3c6ef372u);
const IV3: vec2<u32> = vec2<u32>(0x5f1d36f1u, 0xa54ff53au);
const IV4: vec2<u32> = vec2<u32>(0xade682d1u, 0x510e527fu);
const IV5: vec2<u32> = vec2<u32>(0x2b3e6c1fu, 0x9b05688cu);
const IV6: vec2<u32> = vec2<u32>(0xfb41bd6bu, 0x1f83d9abu);
const IV7: vec2<u32> = vec2<u32>(0x137e2179u, 0x5be0cd19u);

// ---- Uniform inputs ----

struct Params {
    personal: vec4<u32>,
    cfg: vec4<u32>,
    input: array<vec4<u32>, 6>,
    nonce: vec4<u32>,
    nonce_hi: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> leaves: array<u32>;

fn read_input_word(i: u32) -> u32 {
    let v = params.input[i >> 2u];
    let lane = i & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

fn read_nonce_word(i: u32) -> u32 {
    if (i < 4u) {
        switch (i) {
            case 0u: { return params.nonce.x; }
            case 1u: { return params.nonce.y; }
            case 2u: { return params.nonce.z; }
            default: { return params.nonce.w; }
        }
    } else {
        switch (i - 4u) {
            case 0u: { return params.nonce_hi.x; }
            case 1u: { return params.nonce_hi.y; }
            case 2u: { return params.nonce_hi.z; }
            default: { return params.nonce_hi.w; }
        }
    }
}

// ---- BLAKE2b G mixing function ----
//
// Extracting this as a WGSL function causes naga to emit a proper SPIR-V
// OpFunction.  main()'s OpFunction body shrinks from ~900 inlined ops to
// 96 call sites, preventing libnvidia-glvkspirv.so from crashing.
struct GResult { a: vec2<u32>, b: vec2<u32>, c: vec2<u32>, d: vec2<u32> }

fn blake2b_g(
    a: vec2<u32>, b: vec2<u32>, c: vec2<u32>, d: vec2<u32>,
    x: vec2<u32>, y: vec2<u32>,
) -> GResult {
    var ra = u64_add(u64_add(a, b), x);
    var rd = u64_rotr(u64_xor(d, ra), 32u);
    var rc = u64_add(c, rd);
    var rb = u64_rotr(u64_xor(b, rc), 24u);
    ra = u64_add(u64_add(ra, rb), y);
    rd = u64_rotr(u64_xor(rd, ra), 16u);
    rc = u64_add(rc, rd);
    rb = u64_rotr(u64_xor(rb, rc), 63u);
    return GResult(ra, rb, rc, rd);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let call_idx = gid.x;
    let n_leaves = params.cfg.y;
    let digest_len = params.cfg.x;
    let first_leaf = call_idx * 5u;
    if (first_leaf >= n_leaves) {
        return;
    }

    var h: array<vec2<u32>, 8>;
    let p0_lo = digest_len | (1u << 16u) | (1u << 24u);
    h[0] = vec2<u32>(IV0.x ^ p0_lo, IV0.y);
    h[1] = IV1;
    h[2] = IV2;
    h[3] = IV3;
    h[4] = IV4;
    h[5] = IV5;
    h[6] = u64_xor(IV6, vec2<u32>(params.personal.x, params.personal.y));
    h[7] = u64_xor(IV7, vec2<u32>(params.personal.z, params.personal.w));

    var w: array<u32, 32>;
    for (var i = 0u; i < 20u; i = i + 1u) {
        w[i] = read_input_word(i);
    }
    let input_tail = read_input_word(20u) & 0x000000ffu;
    let n0 = read_nonce_word(0u);
    let n1 = read_nonce_word(1u);
    let n2 = read_nonce_word(2u);
    let n3 = read_nonce_word(3u);
    let n4 = read_nonce_word(4u);
    let n5 = read_nonce_word(5u);
    let n6 = read_nonce_word(6u);
    let n7 = read_nonce_word(7u);

    w[20] = input_tail | ((n0 & 0x00ffffffu) << 8u);
    w[21] = (n0 >> 24u) | ((n1 & 0x00ffffffu) << 8u);
    w[22] = (n1 >> 24u) | ((n2 & 0x00ffffffu) << 8u);
    w[23] = (n2 >> 24u) | ((n3 & 0x00ffffffu) << 8u);
    w[24] = (n3 >> 24u) | ((n4 & 0x00ffffffu) << 8u);
    w[25] = (n4 >> 24u) | ((n5 & 0x00ffffffu) << 8u);
    w[26] = (n5 >> 24u) | ((n6 & 0x00ffffffu) << 8u);
    w[27] = (n6 >> 24u) | ((n7 & 0x00ffffffu) << 8u);
    w[28] = (n7 >> 24u) | ((call_idx & 0x00ffffffu) << 8u);
    w[29] = call_idx >> 24u;
    w[30] = 0u;
    w[31] = 0u;

    let block = array<vec2<u32>, 16>(
        vec2<u32>(w[0],  w[1]),  vec2<u32>(w[2],  w[3]),
        vec2<u32>(w[4],  w[5]),  vec2<u32>(w[6],  w[7]),
        vec2<u32>(w[8],  w[9]),  vec2<u32>(w[10], w[11]),
        vec2<u32>(w[12], w[13]), vec2<u32>(w[14], w[15]),
        vec2<u32>(w[16], w[17]), vec2<u32>(w[18], w[19]),
        vec2<u32>(w[20], w[21]), vec2<u32>(w[22], w[23]),
        vec2<u32>(w[24], w[25]), vec2<u32>(w[26], w[27]),
        vec2<u32>(w[28], w[29]), vec2<u32>(w[30], w[31]),
    );

    var v: array<vec2<u32>, 16>;
    v[0] = h[0]; v[1] = h[1]; v[2] = h[2]; v[3] = h[3];
    v[4] = h[4]; v[5] = h[5]; v[6] = h[6]; v[7] = h[7];
    v[8] = IV0;  v[9] = IV1;  v[10] = IV2; v[11] = IV3;
    v[12] = u64_xor(IV4, vec2<u32>(117u, 0u));
    v[13] = IV5;
    v[14] = vec2<u32>(IV6.x ^ 0xffffffffu, IV6.y ^ 0xffffffffu);
    v[15] = IV7;

    // ---- 12 BLAKE2b rounds (96 G calls, literal SIGMA indices) ----
    var _g: GResult;

    // Round 0: sigma = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    _g=blake2b_g(v[0],v[4],v[8],v[12],  block[0], block[1]);  v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[2], block[3]);  v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[4], block[5]);  v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[6], block[7]);  v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[8], block[9]);  v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[10],block[11]); v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[12],block[13]); v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[14],block[15]); v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 1: sigma = 14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[14],block[10]); v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[4], block[8]);  v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[9], block[15]); v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[13],block[6]);  v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[1], block[12]); v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[0], block[2]);  v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[11],block[7]);  v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[5], block[3]);  v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 2: sigma = 11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[11],block[8]);  v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[12],block[0]);  v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[5], block[2]);  v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[15],block[13]); v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[10],block[14]); v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[3], block[6]);  v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[7], block[1]);  v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[9], block[4]);  v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 3: sigma = 7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[7], block[9]);  v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[3], block[1]);  v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[13],block[12]); v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[11],block[14]); v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[2], block[6]);  v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[5], block[10]); v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[4], block[0]);  v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[15],block[8]);  v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 4: sigma = 9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[9], block[0]);  v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[5], block[7]);  v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[2], block[4]);  v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[10],block[15]); v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[14],block[1]);  v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[11],block[12]); v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[6], block[8]);  v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[3], block[13]); v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 5: sigma = 2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[2], block[12]); v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[6], block[10]); v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[0], block[11]); v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[8], block[3]);  v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[4], block[13]); v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[7], block[5]);  v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[15],block[14]); v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[1], block[9]);  v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 6: sigma = 12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[12],block[5]);  v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[1], block[15]); v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[14],block[13]); v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[4], block[10]); v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[0], block[7]);  v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[6], block[3]);  v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[9], block[2]);  v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[8], block[11]); v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 7: sigma = 13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[13],block[11]); v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[7], block[14]); v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[12],block[1]);  v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[3], block[9]);  v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[5], block[0]);  v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[15],block[4]);  v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[8], block[6]);  v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[2], block[10]); v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 8: sigma = 6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[6], block[15]); v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[14],block[9]);  v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[11],block[3]);  v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[0], block[8]);  v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[12],block[2]);  v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[13],block[7]);  v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[1], block[4]);  v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[10],block[5]);  v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 9: sigma = 10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[10],block[2]);  v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[8], block[4]);  v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[7], block[6]);  v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[1], block[5]);  v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[15],block[11]); v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[9], block[14]); v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[3], block[12]); v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[13],block[0]);  v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 10: same as round 0 (BLAKE2b sigma repeats mod 10)
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[0], block[1]);  v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[2], block[3]);  v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[4], block[5]);  v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[6], block[7]);  v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[8], block[9]);  v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[10],block[11]); v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[12],block[13]); v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[14],block[15]); v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    // Round 11: same as round 1
    _g=blake2b_g(v[0],v[4],v[8],v[12],   block[14],block[10]); v[0]=_g.a;v[4]=_g.b;v[8]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[1],v[5],v[9],v[13],   block[4], block[8]);  v[1]=_g.a;v[5]=_g.b;v[9]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[2],v[6],v[10],v[14],  block[9], block[15]); v[2]=_g.a;v[6]=_g.b;v[10]=_g.c;v[14]=_g.d;
    _g=blake2b_g(v[3],v[7],v[11],v[15],  block[13],block[6]);  v[3]=_g.a;v[7]=_g.b;v[11]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[0],v[5],v[10],v[15],  block[1], block[12]); v[0]=_g.a;v[5]=_g.b;v[10]=_g.c;v[15]=_g.d;
    _g=blake2b_g(v[1],v[6],v[11],v[12],  block[0], block[2]);  v[1]=_g.a;v[6]=_g.b;v[11]=_g.c;v[12]=_g.d;
    _g=blake2b_g(v[2],v[7],v[8],v[13],   block[11],block[7]);  v[2]=_g.a;v[7]=_g.b;v[8]=_g.c;v[13]=_g.d;
    _g=blake2b_g(v[3],v[4],v[9],v[14],   block[5], block[3]);  v[3]=_g.a;v[4]=_g.b;v[9]=_g.c;v[14]=_g.d;

    h[0]  = u64_xor(u64_xor(h[0],  v[0]),  v[8]);
    h[1]  = u64_xor(u64_xor(h[1],  v[1]),  v[9]);
    h[2]  = u64_xor(u64_xor(h[2],  v[2]),  v[10]);
    h[3]  = u64_xor(u64_xor(h[3],  v[3]),  v[11]);
    h[4]  = u64_xor(u64_xor(h[4],  v[4]),  v[12]);
    h[5]  = u64_xor(u64_xor(h[5],  v[5]),  v[13]);
    h[6]  = u64_xor(u64_xor(h[6],  v[6]),  v[14]);
    h[7]  = u64_xor(u64_xor(h[7],  v[7]),  v[15]);

    let words: array<u32, 16> = array<u32, 16>(
        h[0].x, h[0].y, h[1].x, h[1].y,
        h[2].x, h[2].y, h[3].x, h[3].y,
        h[4].x, h[4].y, h[5].x, h[5].y,
        h[6].x, h[6].y, h[7].x, h[7].y,
    );
    for (var k = 0u; k < 5u; k = k + 1u) {
        let leaf_id = first_leaf + k;
        if (leaf_id >= n_leaves) {
            return;
        }
        let base_out = leaf_id * 3u;
        let base_in = k * 3u;
        leaves[base_out + 0u] = words[base_in + 0u];
        leaves[base_out + 1u] = words[base_in + 1u];
        leaves[base_out + 2u] = words[base_in + 2u];
    }
}
