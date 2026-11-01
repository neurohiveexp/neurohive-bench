def paginate(items, page, per_page=10):
    """Вернуть элементы страницы с номером page (нумерация страниц с 1).
    Если page < 1 или per_page < 1 — ValueError.
    Для страницы за пределами списка вернуть пустой список."""
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be >= 1")
    start = (page - 1) * per_page
    end = start + per_page
    return list(items[start:end])


def total_pages(n_items, per_page=10):
    """Сколько страниц нужно, чтобы показать n_items элементов.
    0 элементов — 0 страниц. Если n_items < 0 или per_page < 1 — ValueError."""
    if n_items < 0 or per_page < 1:
        raise ValueError("n_items must be >= 0 and per_page >= 1")
    return (n_items + per_page - 1) // per_page
