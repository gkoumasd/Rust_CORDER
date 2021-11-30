#[doc = " Returns the ABI with the given name (if any)."]
pub fn lookup(name: &str) -> Option<Abi> {
    AbiDatas
        .iter()
        .find(|abi_data| name == abi_data.name)
        .map(|&x| x.abi)
}
