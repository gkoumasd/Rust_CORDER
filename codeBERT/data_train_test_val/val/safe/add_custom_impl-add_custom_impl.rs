pub(crate) fn add_custom_impl(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let attr = ctx.find_node_at_offset::<ast::Attr>()?;
    let input = attr.token_tree()?;
    let attr_name = attr
        .syntax()
        .descendants_with_tokens()
        .filter(|t| t.kind() == IDENT)
        .find_map(|i| i.into_token())
        .filter(|t| *t.text() == "derive")?
        .text()
        .clone();
    let trait_token = ctx
        .token_at_offset()
        .find(|t| t.kind() == IDENT && *t.text() != attr_name)?;
    let annotated = attr
        .syntax()
        .siblings(Direction::Next)
        .find_map(ast::Name::cast)?;
    let annotated_name = annotated.syntax().text().to_string();
    let start_offset = annotated.syntax().parent()?.text_range().end();
    let label = format!(
        "Add custom impl `{}` for `{}`",
        trait_token.text().as_str(),
        annotated_name
    );
    let target = attr.syntax().text_range();
    acc.add(
        AssistId("add_custom_impl", AssistKind::Refactor),
        label,
        target,
        |builder| {
            let new_attr_input = input
                .syntax()
                .descendants_with_tokens()
                .filter(|t| t.kind() == IDENT)
                .filter_map(|t| t.into_token().map(|t| t.text().clone()))
                .filter(|t| t != trait_token.text())
                .collect::<Vec<SmolStr>>();
            let has_more_derives = !new_attr_input.is_empty();
            if has_more_derives {
                let new_attr_input = format!("({})", new_attr_input.iter().format(", "));
                builder.replace(input.syntax().text_range(), new_attr_input);
            } else {
                let attr_range = attr.syntax().text_range();
                builder.delete(attr_range);
                let line_break_range = attr
                    .syntax()
                    .next_sibling_or_token()
                    .filter(|t| t.kind() == WHITESPACE)
                    .map(|t| t.text_range())
                    .unwrap_or_else(|| TextRange::new(TextSize::from(0), TextSize::from(0)));
                builder.delete(line_break_range);
            }
            match ctx.config.snippet_cap {
                Some(cap) => {
                    builder.insert_snippet(
                        cap,
                        start_offset,
                        format!(
                            "\n\nimpl {} for {} {{\n    $0\n}}",
                            trait_token, annotated_name
                        ),
                    );
                }
                None => {
                    builder.insert(
                        start_offset,
                        format!("\n\nimpl {} for {} {{\n\n}}", trait_token, annotated_name),
                    );
                }
            }
        },
    )
}
