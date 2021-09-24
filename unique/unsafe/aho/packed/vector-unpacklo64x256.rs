# [doc = " Unpack the low 128-bits of `a` and `b`, and return them as 4 64-bit"] # [doc = " integers."] # [doc = ""] # [doc = " More precisely, if a = a4 a3 a2 a1 and b = b4 b3 b2 b1, where each element"] # [doc = " is a 64-bit integer and a1/b1 correspond to the least significant 64 bits,"] # [doc = " then the return value is `b2 b1 a2 a1`."] # [target_feature (enable = "avx2")] pub fn unpacklo64x256 (a : __m256i , b : __m256i) -> [u64 ; 4] { let lo = _mm256_castsi256_si128 (a) ; let hi = _mm256_castsi256_si128 (b) ; [_mm_cvtsi128_si64 (lo) as u64 , _mm_cvtsi128_si64 (_mm_srli_si128 (lo , 8)) as u64 , _mm_cvtsi128_si64 (hi) as u64 , _mm_cvtsi128_si64 (_mm_srli_si128 (hi , 8)) as u64 ,] }