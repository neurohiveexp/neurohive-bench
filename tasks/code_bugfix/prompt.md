В этом коде есть ошибки. Исправь обе функции так, чтобы они работали как описано в докстрингах. Сигнатуры не меняй. В ответе пришли исправленный код обеих функций целиком в одном блоке ```python.

```python
def paginate(items, page, per_page=10):
    """Вернуть элементы страницы с номером page (нумерация страниц с 1).
    Если page < 1 или per_page < 1 — ValueError.
    Для страницы за пределами списка вернуть пустой список."""
    start = page * per_page
    end = start + per_page - 1
    return items[start:end]


def total_pages(n_items, per_page=10):
    """Сколько страниц нужно, чтобы показать n_items элементов.
    0 элементов — 0 страниц. Если n_items < 0 или per_page < 1 — ValueError."""
    return n_items // per_page
```
