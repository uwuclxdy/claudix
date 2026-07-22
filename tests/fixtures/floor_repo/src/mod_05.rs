pub fn feature_05(value: u32) -> u32 {
    let scaled = value.wrapping_mul(8).wrapping_add(35);
    scaled ^ 0x05
}
