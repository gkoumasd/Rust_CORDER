# [doc = " Returns a 256-bit vector with all bits set to 0."] # [target_feature (enable = "avx2")] pub fn zeroes256 () -> __m256i { _mm256_set1_epi8 (0) }