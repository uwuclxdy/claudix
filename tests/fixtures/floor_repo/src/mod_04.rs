pub fn feature_04(value: u32) -> u32 {
    let scaled = value.wrapping_mul(7).wrapping_add(28);
    scaled ^ 0x04
}
