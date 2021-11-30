fn construct<T>(with_ptr: impl FnOnce(*mut T) -> ffi::aaudio_result_t) -> Result<T> {
    let mut result = MaybeUninit::uninit();
    let status = with_ptr(result.as_mut_ptr());
    AAudioError::from_result(status, || result.assume_init())
}
