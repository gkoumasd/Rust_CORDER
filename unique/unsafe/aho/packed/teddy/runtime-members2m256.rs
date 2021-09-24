# [doc = " Return candidates for Slim 256-bit Teddy, where `chunk` corresponds"] # [doc = " to a 32-byte window of the haystack (where the least significant byte"] # [doc = " corresponds to the start of the window), and the masks correspond to a"] # [doc = " low/high mask for the first and second bytes of all patterns that are being"] # [doc = " searched. The vectors returned correspond to candidates for the first and"] # [doc = " second bytes in the patterns represented by the masks."] # [doc = ""] # [doc = " Note that this can also be used for Fat Teddy, where the high 128 bits in"] # [doc = " `chunk` is the same as the low 128 bits, which corresponds to a 16 byte"] # [doc = " window in the haystack."] # [target_feature (enable = "avx2")] fn members2m256 (chunk : __m256i , mask1 : Mask256 , mask2 : Mask256 ,) -> (__m256i , __m256i) { let lomask = _mm256_set1_epi8 (0xF) ; let hlo = _mm256_and_si256 (chunk , lomask) ; let hhi = _mm256_and_si256 (_mm256_srli_epi16 (chunk , 4) , lomask) ; let res0 = _mm256_and_si256 (_mm256_shuffle_epi8 (mask1 . lo , hlo) , _mm256_shuffle_epi8 (mask1 . hi , hhi) ,) ; let res1 = _mm256_and_si256 (_mm256_shuffle_epi8 (mask2 . lo , hlo) , _mm256_shuffle_epi8 (mask2 . hi , hhi) ,) ; (res0 , res1) }