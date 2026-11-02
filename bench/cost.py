"""Request cost in USD. Same rules as the code_cost task, so the article math and the bench agree."""


def request_cost(input_tokens, output_tokens, cached_tokens, price):
    """Prices in `price` are USD per 1M tokens: input, cached, output,
    optional long_threshold / long_input_mult / long_output_mult."""
    if input_tokens < 0 or output_tokens < 0 or cached_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if cached_tokens > input_tokens:
        raise ValueError("cached_tokens cannot exceed input_tokens")
    input_price = price["input"]
    output_price = price["output"]
    threshold = price.get("long_threshold")
    if threshold is not None and input_tokens > threshold:
        input_price *= price["long_input_mult"]
        output_price *= price["long_output_mult"]
    cost = ((input_tokens - cached_tokens) * input_price
            + cached_tokens * price["cached"]
            + output_tokens * output_price) / 1_000_000
    return round(cost, 6)
