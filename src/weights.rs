//! Shared DFN3 weights: FP32 encoder and FP16 decoder matrices.

#[cfg(not(target_endian = "little"))]
compile_error!("dfn3_rs weights require little-endian storage");

/// Size of the packed model, including tensor alignment padding.
pub const WEIGHTS_SIZE: usize = 5_260_224;

#[cfg(feature = "embedded-model")]
#[repr(C, align(64))]
struct Aligned<T: ?Sized>(T);

// Put the bytes themselves, rather than a pointer, in a discoverable PE section.
#[cfg(feature = "embedded-model")]
#[cfg_attr(target_os = "windows", unsafe(link_section = ".dfn3"))]
static BLOB: Aligned<[u8; WEIGHTS_SIZE]> = Aligned(*include_bytes!("../weights/dfn3.bin"));

/// A validated view of packed weights, shared by any number of denoisers.
/// External storage must remain alive for the lifetime of the process.
#[derive(Clone, Copy, Debug)]
pub struct Model {
    bytes: &'static [u8],
}

impl Model {
    /// Borrow the library's embedded model without allocating.
    #[cfg(feature = "embedded-model")]
    pub fn embedded() -> Self {
        Self { bytes: &BLOB.0 }
    }

    /// Borrow external weights without allocating or copying.
    /// The bytes must use this library's packed tensor layout and little-endian
    /// representation. Length and 64-byte base alignment are checked here.
    pub fn from_bytes(bytes: &'static [u8]) -> Result<Self, ModelError> {
        if bytes.len() != WEIGHTS_SIZE {
            return Err(ModelError::Size);
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(64) {
            return Err(ModelError::Alignment);
        }
        Ok(Self { bytes })
    }
}

/// External weights have the wrong size or base alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    Size,
    Alignment,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Size => "weight blob size mismatch",
            Self::Alignment => "weight blob must be aligned to 64 bytes",
        })
    }
}

impl std::error::Error for ModelError {}

#[derive(Clone, Copy)]
pub(crate) enum Data {
    F32(&'static [f32]),
    F16(&'static [u16]),
}

impl Data {
    pub fn len(self) -> usize {
        match self {
            Self::F32(w) => w.len(),
            Self::F16(w) => w.len(),
        }
    }

    fn slice(self, start: usize, len: usize) -> Self {
        match self {
            Self::F32(w) => Self::F32(&w[start..][..len]),
            Self::F16(w) => Self::F16(&w[start..][..len]),
        }
    }
}

// aarch64 converts FP16 in the NEON kernel (src/simd.rs).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn expand(w: u16) -> f32 {
    let sign = ((w & 0x8000) as u32) << 16;
    let exp = (w >> 10) & 31;
    let mantissa = (w & 1023) as u32;
    let value = match exp {
        0 => (mantissa as f32 * (1.0 / 16_777_216.0)).to_bits(),
        31 => 0x7f80_0000 | (mantissa << 13),
        _ => ((exp as u32 + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | value)
}

/// Matrix packed as `[n / 16][k][16]`.
/// `n` is padded to a multiple of 16.
#[derive(Clone, Copy)]
pub(crate) struct Mat {
    pub k: usize,
    pub n: usize,
    pub w: Data,
}

/// `groups` independent matrices stored back to back (block-diagonal linear).
#[derive(Clone, Copy)]
pub(crate) struct Grouped {
    pub groups: usize,
    pub mat: Mat,
}

impl Grouped {
    pub fn group(&self, g: usize) -> Mat {
        let len = self.mat.k * self.mat.n;
        Mat {
            w: self.mat.w.slice(g * len, len),
            ..self.mat
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Gru {
    pub w: Mat,
    pub r: Mat,
    pub bias_w: &'static [f32],
    pub bias_r: &'static [f32],
}

/// Sequential reader; every tensor starts on a 64-byte boundary.
pub(crate) struct Reader {
    bytes: &'static [u8],
    pub decoder: bool,
}

impl Reader {
    pub fn new(model: Model) -> Self {
        Self {
            bytes: model.bytes,
            decoder: false,
        }
    }

    fn tensor<T>(&mut self, len: usize) -> &'static [T] {
        let size = len * std::mem::size_of::<T>();
        let (head, rest) = self.bytes.split_at(size.next_multiple_of(64));
        self.bytes = rest;
        // SAFETY: validated tensors start at 64-byte boundaries; T is f32 or u16,
        // both accept every bit pattern. The exporter writes little-endian data.
        unsafe { std::slice::from_raw_parts(head.as_ptr().cast(), len) }
    }

    /// Returns a padded FP32 tensor, including padding used by SIMD bias loads.
    pub fn take(&mut self, len: usize) -> &'static [f32] {
        self.tensor(len.next_multiple_of(16))
    }

    fn matrix(&mut self, len: usize) -> Data {
        if self.decoder {
            return Data::F16(self.tensor(len));
        }
        Data::F32(self.tensor(len))
    }

    pub fn mat(&mut self, k: usize, n: usize) -> Mat {
        let n = n.next_multiple_of(16);
        Mat {
            k,
            n,
            w: self.matrix(k * n),
        }
    }

    pub fn grouped(&mut self, groups: usize, k: usize, n: usize) -> Grouped {
        let n = n.next_multiple_of(16);
        let w = self.matrix(groups * k * n);
        Grouped {
            groups,
            mat: Mat { k, n, w },
        }
    }

    pub fn gru(&mut self, inputs: usize) -> Gru {
        Gru {
            w: self.mat(inputs, 768),
            r: self.mat(256, 768),
            bias_w: self.take(768),
            bias_r: self.take(768),
        }
    }

    pub fn finish(self) {
        assert!(self.bytes.is_empty(), "weight blob layout mismatch");
    }
}
