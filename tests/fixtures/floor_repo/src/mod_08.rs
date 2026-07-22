pub fn feature_08(value: u32) -> u32 {
    let scaled = value.wrapping_mul(11).wrapping_add(56);
    scaled ^ 0x08
}
