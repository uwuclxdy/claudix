pub fn feature_02(value: u32) -> u32 {
    let scaled = value.wrapping_mul(5).wrapping_add(14);
    scaled ^ 0x02
}
