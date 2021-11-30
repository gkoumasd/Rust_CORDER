pub fn foo<S: Iterator>() -> String
where
    <S as Iterator>::Item: Eq,
{
    "".to_owned()
}
