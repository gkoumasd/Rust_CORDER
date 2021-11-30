#[doc = " Try to read the features using IsProcessorFeaturePresent."]
pub(crate) fn detect_features() -> cache::Initializer {
    type DWORD = u32;
    type BOOL = i32;
    const FALSE: BOOL = 0;
    const PF_ARM_NEON_INSTRUCTIONS_AVAILABLE: u32 = 19;
    const PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE: u32 = 30;
    const PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE: u32 = 31;
    extern "system" {
        pub fn IsProcessorFeaturePresent(ProcessorFeature: DWORD) -> BOOL;
    }
    let mut value = cache::Initializer::default();
    {
        let mut enable_feature = |f, enable| {
            if enable {
                value.set(f as u32);
            }
        };
        {
            enable_feature(
                Feature::asimd,
                IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::crc,
                IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::crypto,
                IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::pmull,
                IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
        }
    }
    value
}
