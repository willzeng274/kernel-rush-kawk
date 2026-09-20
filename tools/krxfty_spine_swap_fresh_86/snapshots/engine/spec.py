"""Draft generation for speculative decoding: n-gram lookup over the sequence.

No extra model and no weights: the draft for the next K tokens is whatever
followed the most recent earlier occurrence of the current suffix (longest
n-gram first). Verification is exact, so a wrong draft costs only the extra
rows in the verify pass, never a wrong token.
"""

from __future__ import annotations


class NGramDrafter:
    def __init__(self, tokens: list[int], K: int, max_n: int = 4, min_n: int = 1):
        self.tokens: list[int] = []
        self.K, self.max_n, self.min_n = K, max_n, min_n
        self.index: list[dict[tuple[int, ...], int]] = [dict() for _ in range(max_n + 1)]
        self.extend(tokens)

    def extend(self, new_tokens: list[int]) -> None:
        toks = self.tokens
        for tok in new_tokens:
            i = len(toks)
            toks.append(tok)
            for n in range(self.min_n, self.max_n + 1):
                if i >= n:
                    self.index[n][tuple(toks[i - n:i])] = i

    def draft(self) -> list[int]:
        toks = self.tokens
        L = len(toks)
        for n in range(self.max_n, self.min_n - 1, -1):
            if L < n:
                continue
            p = self.index[n].get(tuple(toks[L - n:]))
            if p is not None:
                out = toks[p:p + self.K]
                if len(out) < self.K:
                    out = out + [out[-1] if out else toks[-1]] * (self.K - len(out))
                return out
        return [toks[-1]] * self.K

    def draft_or_none(self) -> list[int] | None:
        """Continuation after the longest suffix match of length >= min_n, or None."""
        toks = self.tokens
        L = len(toks)
        for n in range(self.max_n, self.min_n - 1, -1):
            if L < n:
                continue
            p = self.index[n].get(tuple(toks[L - n:]))
            if p is not None:
                out = toks[p:p + self.K]
                return out if out else None
        return None
