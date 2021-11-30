fn range_borders(r: IntRange<'_>) -> impl Iterator<Item = Border> {
    let (lo, hi) = r.range.into_inner();
    let from = Border::JustBefore(lo);
    let to = match hi.checked_add(1) {
        Some(m) => Border::JustBefore(m),
        None => Border::AfterMax,
    };
    vec![from, to].into_iter()
}
