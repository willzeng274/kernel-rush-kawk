"""Rank-preserving, request-local proposal insertion; never accepts output."""


def validate_ranked(predictions, ranked, batch, width):
    """Validate the complete result before any request/table mutation."""
    if width != 8:
        raise ValueError("ranked proposals require a W8 verification plan")
    def sequence(value, length):
        return isinstance(value, (list, tuple)) and len(value) == length
    if not sequence(predictions, batch) or not sequence(ranked, batch):
        raise ValueError("one ranked result per batch member required")
    for ys, pairs in zip(predictions, ranked):
        if not sequence(ys, width) or not sequence(pairs, width):
            raise ValueError("one ranked pair per verifier node required")
        for greedy, pair in zip(ys, pairs):
            if (type(greedy) is not int or not 0 <= greedy < 151936
                    or not sequence(pair, 2)
                    or any(type(x) is not int or not 0 <= x < 151936 for x in pair)
                    or pair[0] != greedy or pair[0] == pair[1]):
                raise ValueError("invalid ranked pair or authoritative argmax mismatch")


def record_ranked(table, context, pair):
    """Insert an already validated whole pair, top-one first, last row wins."""
    values = (pair[0], pair[1])
    for size in (4, 2, 1):
        if len(context) >= size:
            key = tuple(context[-size:])
            table.predicted.pop(key, None)
            table.predicted[key] = values
            if len(table.predicted) > table.capacity:
                table.predicted.popitem(last=False)
