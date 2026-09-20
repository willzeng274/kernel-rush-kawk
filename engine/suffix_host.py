"""Request-local accepted-history SAM; fixed complete three-token support."""
class SAM:
    """Full accepted history; earliest end per state; complete D-token drafts.

    Standard incremental SAM construction, independently implemented here.
    Looking up after append is safe: start at the suffix link of the terminal
    state, then require the earliest endpoint to leave D observed successors.
    No hypothetical verifier rows enter this structure.
    """
    __slots__ = ("tokens", "length", "link", "first", "edges", "last", "d", "frozen")

    def __init__(self, tokens=(), d=3):
        self.tokens = []
        self.length, self.link, self.first, self.edges = [0], [-1], [-1], [{}]
        self.frozen = False
        self.last, self.d = 0, d
        self.append(tokens)

    def append(self, tokens):
        self._live()
        for token in tokens:
            c = int(token)
            pos = len(self.tokens)
            self.tokens.append(c)
            cur = len(self.length)
            self.length.append(self.length[self.last] + 1)
            self.link.append(0)
            self.first.append(pos)
            self.edges.append({})
            p = self.last
            while p >= 0 and c not in self.edges[p]:
                self.edges[p][c] = cur
                p = self.link[p]
            if p >= 0:
                q = self.edges[p][c]
                if self.length[p] + 1 == self.length[q]:
                    self.link[cur] = q
                else:
                    clone = len(self.length)
                    self.length.append(self.length[p] + 1)
                    self.link.append(self.link[q])
                    self.first.append(self.first[q])
                    self.edges.append(self.edges[q].copy())
                    while p >= 0 and self.edges[p].get(c) == q:
                        self.edges[p][c] = clone
                        p = self.link[p]
                    self.link[q] = self.link[cur] = clone
            self.last = cur

    def match_state(self):
        self._live()
        s = self.link[self.last] if self.last else 0
        while s > 0 and self.first[s] + self.d >= len(self.tokens):
            s = self.link[s]
        return max(s, 0)

    def match(self):
        s = self.match_state()
        if s <= 0:
            return 0, -1, ()
        end = self.first[s] + 1
        return self.length[s], end, tuple(self.tokens[end:end + self.d])

    def draft(self):
        return self.match()[2]

    def supports_at_longest(self, draft):
        s = self.match_state()
        if not s or len(draft) != self.d:
            return False
        for token in draft:
            s = self.edges[s].get(token, -1)
            if s < 0:
                return False
        return True

    def coherent_override(self, old_draft):
        # Inherited minimum bigram scope; no new weak one-token fallback.
        # This is a concrete design policy, not a calibrated acceptance model.
        if self.match()[0] < 2 or self.supports_at_longest(old_draft):
            return tuple(old_draft)
        return self.draft()

    def _live(self):
        if self.frozen:
            raise ValueError("frozen proposal index cannot be used")

    def freeze(self):
        self.frozen = True

    def choose(self, old):
        s = self.match_state()
        if not s or self.length[s] < 2:
            return tuple(old), self.length[s]
        following = s
        for token in old:
            following = self.edges[following].get(token, -1)
            if following < 0:
                end = self.first[s] + 1
                return tuple(self.tokens[end:end + 3]), self.length[s]
        return tuple(old), self.length[s]
