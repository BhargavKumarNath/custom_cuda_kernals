//! Mirrors the `KernelDType` enum in csrc/includes/common.cuh. Values are
//! part of the Rust <-> C++ ABI contract — do not renumber without updating
//! both sides.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    F8E4M3 = 3,
    F8E5M2 = 4,
}

impl KernelDType {
    pub fn from_torch_dtype_str(s: &str) -> Option<Self> {
        match s {
            "torch.float32" => Some(Self::F32),
            "torch.float16" => Some(Self::F16),
            "torch.bfloat16" => Some(Self::BF16),
            "torch.float8_e4m3fn" => Some(Self::F8E4M3),
            "torch.float8_e5m2" => Some(Self::F8E5M2),
            _ => None,
        }
    }
}
