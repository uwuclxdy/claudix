pub fn feature_07(value: u32) -> u32 {
    let scaled = value.wrapping_mul(10).wrapping_add(49);
    scaled ^ 0x07
}
