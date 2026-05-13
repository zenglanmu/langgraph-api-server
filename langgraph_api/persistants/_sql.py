def split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_dollar_quote = False
    for line in sql.split("\n"):
        if line.count("$$") % 2 == 1:
            in_dollar_quote = not in_dollar_quote
        current.append(line)
        if not in_dollar_quote and line.rstrip().endswith(";"):
            stmt = "\n".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
    if current:
        remaining = "\n".join(current).strip()
        if remaining:
            statements.append(remaining)
    return statements
