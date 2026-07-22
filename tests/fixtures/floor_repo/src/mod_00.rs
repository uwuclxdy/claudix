pub fn feature_00(value: u32) -> u32 {
    let scaled = value.wrapping_mul(3).wrapping_add(0);
    scaled ^ 0x00
}
