#[doc = " Read the RData from the given Decoder"]
#[allow(clippy::many_single_char_names)]
pub fn read(decoder: &mut BinDecoder<'_>) -> ProtoResult<Ipv6Addr> {
    let a: u16 = decoder.read_u16()?.unverified();
    let b: u16 = decoder.read_u16()?.unverified();
    let c: u16 = decoder.read_u16()?.unverified();
    let d: u16 = decoder.read_u16()?.unverified();
    let e: u16 = decoder.read_u16()?.unverified();
    let f: u16 = decoder.read_u16()?.unverified();
    let g: u16 = decoder.read_u16()?.unverified();
    let h: u16 = decoder.read_u16()?.unverified();
    Ok(Ipv6Addr::new(a, b, c, d, e, f, g, h))
}
