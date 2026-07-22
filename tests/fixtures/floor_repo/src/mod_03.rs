pub fn feature_03(value: u32) -> u32 {
    let scaled = value.wrapping_mul(6).wrapping_add(21);
    scaled ^ 0x03
}
