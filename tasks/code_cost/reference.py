def request_cost(input_tokens, output_tokens, cached_tokens, price):
    """Cost of one API request in USD. Prices in `price` are USD per 1M tokens."""
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

    uncached = input_tokens - cached_tokens
    cost = (
        uncached * input_price
        + cached_tokens * price["cached"]
        + output_tokens * output_price
    ) / 1_000_000
    return round(cost, 6)
