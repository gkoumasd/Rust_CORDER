fn for_slice<F>()
where
    for<'a> [&'a F]: Eq,
{
}
