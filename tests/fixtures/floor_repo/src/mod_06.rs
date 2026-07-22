pub fn feature_06(value: u32) -> u32 {
    let scaled = value.wrapping_mul(9).wrapping_add(42);
    scaled ^ 0x06
}
