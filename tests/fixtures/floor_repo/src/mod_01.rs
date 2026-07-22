pub fn feature_01(value: u32) -> u32 {
    let scaled = value.wrapping_mul(4).wrapping_add(7);
    scaled ^ 0x01
}
