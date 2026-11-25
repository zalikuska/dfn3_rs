//! Hot kernels: packed GEMM/GEMV and GRU/tanh activations.
//! AVX2+FMA (plus F16C for mixed precision) is selected at construction.
//! aarch64 always uses NEON (part of the baseline; added in this copy, see NOTICE).
//! Other CPUs use the scalar path.

use crate::weights::{Data, Mat};

#[derive(Clone, Copy)]
pub(crate) struct Simd {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    avx2: bool,
}

impl Simd {
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        let avx2 = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        Self { avx2 }
    }

    /// `out[r][..n] = relu?(a[r] · w + bias + (acc ? out[r] : 0))` for `r < m`.
    /// `m` must be 1 or a multiple of 4; all `w.n` padded columns are written.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        self,
        m: usize,
        a: &[f32],
        lda: usize,
        w: &Mat,
        bias: Option<&[f32]>,
        acc: bool,
        relu: bool,
        out: &mut [f32],
        ldo: usize,
    ) {
        assert!(m == 1 || m.is_multiple_of(4));
        assert!(a.len() >= (m - 1) * lda + w.k && out.len() >= (m - 1) * ldo + w.n);
        assert!(w.w.len() == w.k * w.n && bias.is_none_or(|b| b.len() >= w.n));
        macro_rules! run {
            ($weights:expr, $expand:expr, $half:literal) => {{
                let weights = $weights;
                #[cfg(target_arch = "x86_64")]
                if self.avx2 {
                    // SAFETY: CPU support checked at construction; bounds asserted above.
                    return unsafe {
                        x86::gemm::<$half>(
                            m,
                            a,
                            lda,
                            w.k,
                            w.n,
                            weights.as_ptr().cast(),
                            bias,
                            acc,
                            relu,
                            out,
                            ldo,
                        )
                    };
                }
                #[cfg(target_arch = "aarch64")]
                // SAFETY: NEON is part of the aarch64 baseline; bounds asserted above.
                return unsafe {
                    neon::gemm::<$half>(
                        m,
                        a,
                        lda,
                        w.k,
                        w.n,
                        weights.as_ptr().cast(),
                        bias,
                        acc,
                        relu,
                        out,
                        ldo,
                    )
                };
                #[cfg(not(target_arch = "aarch64"))]
                for r in 0..m {
                    for n in 0..w.n {
                        let col = &weights[n / 16 * w.k * 16 + n % 16..];
                        let mut sum = 0.0;
                        for i in 0..w.k {
                            sum += a[r * lda + i] * $expand(col[i * 16]);
                        }
                        let o = &mut out[r * ldo + n];
                        sum += bias.map_or(0.0, |b| b[n]) + if acc { *o } else { 0.0 };
                        *o = if relu { sum.max(0.0) } else { sum };
                    }
                }
            }};
        }
        match w.w {
            Data::F32(weights) => run!(weights, std::convert::identity, false),
            Data::F16(weights) => run!(weights, crate::weights::expand, true),
        }
    }

    /// GRU update with reset applied after the recurrent projection.
    /// Packed pre-activations `gi = W x + b_w`, `gh = R h + b_r` use
    /// z (update), r (reset), n (candidate) order.
    pub fn gru_gates(self, gi: &[f32; 768], gh: &[f32; 768], h: &mut [f32; 256]) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: CPU support checked at construction; fixed sizes.
            return unsafe { x86::gru_gates(gi, gh, h) };
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline; fixed sizes.
        return unsafe { neon::gru_gates(gi, gh, h) };
        #[cfg(not(target_arch = "aarch64"))]
        for j in 0..256 {
            let z = sigmoid(gi[j] + gh[j]);
            let r = sigmoid(gi[256 + j] + gh[256 + j]);
            let n = (gi[512 + j] + r * gh[512 + j]).tanh();
            h[j] = n + z * (h[j] - n);
        }
    }

    pub fn tanh(self, x: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: CPU support checked at construction.
            return unsafe { x86::tanh_slice(x) };
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline.
        return unsafe { neon::tanh_slice(x) };
        #[cfg(not(target_arch = "aarch64"))]
        x.iter_mut().for_each(|v| *v = v.tanh());
    }
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    /// One register tile: `R` rows x `T` 16-column tiles (R * T * 2 <= 8 accumulators).
    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile<const R: usize, const T: usize, const HALF: bool>(
        k: usize,
        a: *const f32,
        lda: usize,
        w: *const u8,
        bias: *const f32,
        acc: bool,
        relu: bool,
        out: *mut f32,
        ldo: usize,
    ) {
        unsafe {
            let mut sum = [[[_mm256_setzero_ps(); 2]; T]; R];
            for i in 0..k {
                for t in 0..T {
                    let p = w.add((t * k + i) * 16 * if HALF { 2 } else { 4 });
                    let v = if HALF {
                        [
                            _mm256_cvtph_ps(_mm_load_si128(p.cast())),
                            _mm256_cvtph_ps(_mm_load_si128(p.add(16).cast())),
                        ]
                    } else {
                        [_mm256_load_ps(p.cast()), _mm256_load_ps(p.add(32).cast())]
                    };
                    for (r, row) in sum.iter_mut().enumerate() {
                        let x = _mm256_set1_ps(*a.add(r * lda + i));
                        row[t][0] = _mm256_fmadd_ps(x, v[0], row[t][0]);
                        row[t][1] = _mm256_fmadd_ps(x, v[1], row[t][1]);
                    }
                }
            }
            for (r, row) in sum.iter().enumerate() {
                for (t, s) in row.iter().enumerate() {
                    for (h, &v) in s.iter().enumerate() {
                        let o = out.add(r * ldo + t * 16 + h * 8);
                        let mut v = v;
                        if !bias.is_null() {
                            v = _mm256_add_ps(v, _mm256_loadu_ps(bias.add(t * 16 + h * 8)));
                        }
                        if acc {
                            v = _mm256_add_ps(v, _mm256_loadu_ps(o));
                        }
                        if relu {
                            v = _mm256_max_ps(v, _mm256_setzero_ps());
                        }
                        _mm256_storeu_ps(o, v);
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm<const HALF: bool>(
        m: usize,
        a: &[f32],
        lda: usize,
        k: usize,
        n: usize,
        wp: *const u8,
        bias: Option<&[f32]>,
        acc: bool,
        relu: bool,
        out: &mut [f32],
        ldo: usize,
    ) {
        let tiles = n / 16;
        let bytes = if HALF { 2 } else { 4 };
        let (a, o) = (a.as_ptr(), out.as_mut_ptr());
        let bias = |t: usize| bias.map_or(std::ptr::null(), |b| b[t * 16..].as_ptr());
        unsafe {
            if m == 1 {
                let mut t = 0;
                while t + 4 <= tiles {
                    tile::<1, 4, HALF>(
                        k,
                        a,
                        0,
                        wp.add(t * k * 16 * bytes),
                        bias(t),
                        acc,
                        relu,
                        o.add(t * 16),
                        0,
                    );
                    t += 4;
                }
                while t + 2 <= tiles {
                    tile::<1, 2, HALF>(
                        k,
                        a,
                        0,
                        wp.add(t * k * 16 * bytes),
                        bias(t),
                        acc,
                        relu,
                        o.add(t * 16),
                        0,
                    );
                    t += 2;
                }
                if t < tiles {
                    tile::<1, 1, HALF>(
                        k,
                        a,
                        0,
                        wp.add(t * k * 16 * bytes),
                        bias(t),
                        acc,
                        relu,
                        o.add(t * 16),
                        0,
                    );
                }
            } else {
                for t in 0..tiles {
                    for r in (0..m).step_by(4) {
                        let (a, o) = (a.add(r * lda), o.add(r * ldo + t * 16));
                        tile::<4, 1, HALF>(
                            k,
                            a,
                            lda,
                            wp.add(t * k * 16 * bytes),
                            bias(t),
                            acc,
                            relu,
                            o,
                            ldo,
                        );
                    }
                }
            }
        }
    }

    /// Cephes-style exp, valid after clamping to the finite f32 range.
    #[target_feature(enable = "avx2,fma")]
    fn exp(x: __m256) -> __m256 {
        let x = _mm256_min_ps(
            _mm256_max_ps(x, _mm256_set1_ps(-87.0)),
            _mm256_set1_ps(88.0),
        );
        let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
        );
        let f = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_4), x);
        let f = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), f);
        let mut p = _mm256_set1_ps(1.987_569_1e-4);
        for c in [
            1.398_199_9e-3,
            8.333_452e-3,
            4.166_579_6e-2,
            1.666_666_5e-1,
            0.5,
        ] {
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(c));
        }
        let y = _mm256_fmadd_ps(
            p,
            _mm256_mul_ps(f, f),
            _mm256_add_ps(f, _mm256_set1_ps(1.0)),
        );
        let scale = _mm256_slli_epi32::<23>(_mm256_add_epi32(
            _mm256_cvtps_epi32(n),
            _mm256_set1_epi32(127),
        ));
        _mm256_mul_ps(y, _mm256_castsi256_ps(scale))
    }

    #[target_feature(enable = "avx2,fma")]
    fn sigmoid(x: __m256) -> __m256 {
        let one = _mm256_set1_ps(1.0);
        _mm256_div_ps(
            one,
            _mm256_add_ps(one, exp(_mm256_sub_ps(_mm256_setzero_ps(), x))),
        )
    }

    #[target_feature(enable = "avx2,fma")]
    fn tanh(x: __m256) -> __m256 {
        let two = _mm256_set1_ps(2.0);
        _mm256_fmsub_ps(two, sigmoid(_mm256_mul_ps(two, x)), _mm256_set1_ps(1.0))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn gru_gates(gi: &[f32; 768], gh: &[f32; 768], h: &mut [f32; 256]) {
        let (gi, gh, hp) = (gi.as_ptr(), gh.as_ptr(), h.as_mut_ptr());
        for j in (0..256).step_by(8) {
            unsafe {
                let ld = |p: *const f32, o: usize| _mm256_loadu_ps(p.add(o + j));
                let z = sigmoid(_mm256_add_ps(ld(gi, 0), ld(gh, 0)));
                let r = sigmoid(_mm256_add_ps(ld(gi, 256), ld(gh, 256)));
                let n = tanh(_mm256_fmadd_ps(r, ld(gh, 512), ld(gi, 512)));
                let old = _mm256_loadu_ps(hp.add(j));
                _mm256_storeu_ps(hp.add(j), _mm256_fmadd_ps(z, _mm256_sub_ps(old, n), n));
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn tanh_slice(x: &mut [f32]) {
        let (chunks, tail) = x.as_chunks_mut::<8>();
        for c in chunks {
            unsafe { _mm256_storeu_ps(c.as_mut_ptr(), tanh(_mm256_loadu_ps(c.as_ptr()))) };
        }
        tail.iter_mut().for_each(|v| *v = v.tanh());
    }
}

/// NEON counterparts of the x86 kernels (this copy only, see NOTICE): the same packed
/// `[n / 16][k][16]` layout and tiling, 4-lane vectors instead of 8.
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)] // some NEON intrinsics are safe to call in newer toolchains
mod neon {
    use core::arch::aarch64::*;

    /// Eight FP16 weights as two f32 vectors. FCVTL/FCVTL2 (half to single) are in the
    /// ARMv8-A baseline, but the float16 intrinsics aren't stable in Rust yet.
    #[inline(always)]
    unsafe fn load_half(p: *const u16) -> (float32x4_t, float32x4_t) {
        unsafe {
            let raw = vld1q_u16(p);
            let lo: float32x4_t;
            let hi: float32x4_t;
            // `out`, not `lateout`: `lo` is written before the second read of `raw`.
            core::arch::asm!(
                "fcvtl {lo:v}.4s, {raw:v}.4h",
                "fcvtl2 {hi:v}.4s, {raw:v}.8h",
                raw = in(vreg) raw,
                lo = out(vreg) lo,
                hi = out(vreg) hi,
                options(pure, nomem, nostack, preserves_flags),
            );
            (lo, hi)
        }
    }

    /// Sixteen packed weights of row `i` of 16-column tile `t`.
    #[inline(always)]
    unsafe fn load16<const HALF: bool>(w: *const u8, k: usize, t: usize, i: usize) -> [float32x4_t; 4] {
        unsafe {
            if HALF {
                let p = w.cast::<u16>().add((t * k + i) * 16);
                let (a, b) = load_half(p);
                let (c, d) = load_half(p.add(8));
                [a, b, c, d]
            } else {
                let p = w.cast::<f32>().add((t * k + i) * 16);
                [vld1q_f32(p), vld1q_f32(p.add(4)), vld1q_f32(p.add(8)), vld1q_f32(p.add(12))]
            }
        }
    }

    /// One register tile: `R` rows x `T` 16-column tiles (R * T * 4 <= 16 accumulators
    /// of the 32 vector registers).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile<const R: usize, const T: usize, const HALF: bool>(
        k: usize,
        a: *const f32,
        lda: usize,
        w: *const u8,
        bias: *const f32,
        acc: bool,
        relu: bool,
        out: *mut f32,
        ldo: usize,
    ) {
        unsafe {
            let mut sum = [[[vdupq_n_f32(0.0); 4]; T]; R];
            for i in 0..k {
                for t in 0..T {
                    let v = load16::<HALF>(w, k, t, i);
                    for (r, row) in sum.iter_mut().enumerate() {
                        let x = *a.add(r * lda + i);
                        for (s, &v) in row[t].iter_mut().zip(&v) {
                            *s = vfmaq_n_f32(*s, v, x);
                        }
                    }
                }
            }
            for (r, row) in sum.iter().enumerate() {
                for (t, s) in row.iter().enumerate() {
                    for (q, &v) in s.iter().enumerate() {
                        let o = out.add(r * ldo + t * 16 + q * 4);
                        let mut v = v;
                        if !bias.is_null() {
                            v = vaddq_f32(v, vld1q_f32(bias.add(t * 16 + q * 4)));
                        }
                        if acc {
                            v = vaddq_f32(v, vld1q_f32(o));
                        }
                        if relu {
                            v = vmaxq_f32(v, vdupq_n_f32(0.0));
                        }
                        vst1q_f32(o, v);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm<const HALF: bool>(
        m: usize,
        a: &[f32],
        lda: usize,
        k: usize,
        n: usize,
        wp: *const u8,
        bias: Option<&[f32]>,
        acc: bool,
        relu: bool,
        out: &mut [f32],
        ldo: usize,
    ) {
        let tiles = n / 16;
        let bytes = if HALF { 2 } else { 4 };
        let (a, o) = (a.as_ptr(), out.as_mut_ptr());
        let bias = |t: usize| bias.map_or(std::ptr::null(), |b| b[t * 16..].as_ptr());
        unsafe {
            let w = |t: usize| wp.add(t * k * 16 * bytes);
            if m == 1 {
                let mut t = 0;
                while t + 4 <= tiles {
                    tile::<1, 4, HALF>(k, a, 0, w(t), bias(t), acc, relu, o.add(t * 16), 0);
                    t += 4;
                }
                while t + 2 <= tiles {
                    tile::<1, 2, HALF>(k, a, 0, w(t), bias(t), acc, relu, o.add(t * 16), 0);
                    t += 2;
                }
                if t < tiles {
                    tile::<1, 1, HALF>(k, a, 0, w(t), bias(t), acc, relu, o.add(t * 16), 0);
                }
            } else {
                for t in 0..tiles {
                    for r in (0..m).step_by(4) {
                        let (a, o) = (a.add(r * lda), o.add(r * ldo + t * 16));
                        tile::<4, 1, HALF>(k, a, lda, w(t), bias(t), acc, relu, o, ldo);
                    }
                }
            }
        }
    }

    /// Cephes-style exp, valid after clamping to the finite f32 range (as the x86 one).
    #[inline(always)]
    fn exp(x: float32x4_t) -> float32x4_t {
        unsafe {
            let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0)), vdupq_n_f32(88.0));
            let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
            let f = vfmsq_f32(x, n, vdupq_n_f32(0.693_359_4));
            let f = vfmsq_f32(f, n, vdupq_n_f32(-2.121_944_4e-4));
            let mut p = vdupq_n_f32(1.987_569_1e-4);
            for c in [
                1.398_199_9e-3,
                8.333_452e-3,
                4.166_579_6e-2,
                1.666_666_5e-1,
                0.5,
            ] {
                p = vfmaq_f32(vdupq_n_f32(c), p, f);
            }
            let y = vfmaq_f32(vaddq_f32(f, vdupq_n_f32(1.0)), p, vmulq_f32(f, f));
            let scale = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
            vmulq_f32(y, vreinterpretq_f32_s32(scale))
        }
    }

    #[inline(always)]
    fn sigmoid(x: float32x4_t) -> float32x4_t {
        unsafe {
            let one = vdupq_n_f32(1.0);
            vdivq_f32(one, vaddq_f32(one, exp(vnegq_f32(x))))
        }
    }

    #[inline(always)]
    fn tanh(x: float32x4_t) -> float32x4_t {
        unsafe {
            let two = vdupq_n_f32(2.0);
            vfmaq_f32(vdupq_n_f32(-1.0), two, sigmoid(vmulq_f32(two, x)))
        }
    }

    pub(super) unsafe fn gru_gates(gi: &[f32; 768], gh: &[f32; 768], h: &mut [f32; 256]) {
        let (gi, gh, hp) = (gi.as_ptr(), gh.as_ptr(), h.as_mut_ptr());
        for j in (0..256).step_by(4) {
            unsafe {
                let ld = |p: *const f32, o: usize| vld1q_f32(p.add(o + j));
                let z = sigmoid(vaddq_f32(ld(gi, 0), ld(gh, 0)));
                let r = sigmoid(vaddq_f32(ld(gi, 256), ld(gh, 256)));
                let n = tanh(vfmaq_f32(ld(gi, 512), r, ld(gh, 512)));
                let old = vld1q_f32(hp.add(j));
                vst1q_f32(hp.add(j), vfmaq_f32(n, z, vsubq_f32(old, n)));
            }
        }
    }

    pub(super) unsafe fn tanh_slice(x: &mut [f32]) {
        let (chunks, tail) = x.as_chunks_mut::<4>();
        for c in chunks {
            unsafe { vst1q_f32(c.as_mut_ptr(), tanh(vld1q_f32(c.as_ptr()))) };
        }
        tail.iter_mut().for_each(|v| *v = v.tanh());
    }
}
