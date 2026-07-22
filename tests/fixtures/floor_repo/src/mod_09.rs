pub fn feature_09(value: u32) -> u32 {
    let scaled = value.wrapping_mul(12).wrapping_add(63);
    scaled ^ 0x09
}
