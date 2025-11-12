# graph_io.py (string labels, no regex)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Iterable, FrozenSet


@dataclass
class Graph:
    n: int
    m: int
    directed: bool
    s: str
    t: str
    vertices: Set[str]
    red: Set[str]
    adj: Dict[str, List[str]] = field(default_factory=dict)

    def add_edge(self, u: str, v: str) -> None:
        if u not in self.vertices or v not in self.vertices:
            raise ValueError(f"Edge references unknown vertex: {u} -> {v}")
        self.adj.setdefault(u, []).append(v)
        if not self.directed:
            self.adj.setdefault(v, []).append(u)

    def neighbors(self, u: str) -> List[str]:
        return self.adj.get(u, [])

    def degree(self, u: str) -> int:
        return len(self.neighbors(u))

    def validate(self) -> None:
        # start/end must exist
        if self.s not in self.vertices or self.t not in self.vertices:
            raise ValueError("Start or end vertex not in vertex set.")
        # vertex count must match N
        if len(self.vertices) != self.n:
            raise ValueError(f"N mismatch: header {self.n}, parsed {len(self.vertices)}.")
        # red must be subset of vertices
        if any(v not in self.vertices for v in self.red):
            raise ValueError("Red set contains unknown vertices.")
        # edge counts
        if self.directed:
            counted = sum(len(vs) for vs in self.adj.values())
            if counted != self.m:
                raise ValueError(f"M mismatch: header {self.m}, parsed {counted} directed edges.")
        else:
            counted_twice = sum(len(vs) for vs in self.adj.values())
            if counted_twice != 2 * self.m:
                raise ValueError(f"M mismatch: header {self.m}, parsed {counted_twice//2} undirected edges.")

    def __repr__(self) -> str:
        kind = "Directed" if self.directed else "Undirected"
        return f"<Graph {kind} n={self.n} m={self.m} s={self.s} t={self.t} red={len(self.red)}>"


class GraphParser:
    """
    Plain-split parser (no regex).
    Format:
      Line 1: N M R                (three integers)
      Line 2: s t                  (two strings, no spaces inside a label)
      Next N lines: v [*]          (star optional; 'A*' or 'A *' both OK)
      Next M lines: u -> v   OR   u -- v   (never mixed)
    Notes:
      - All vertex labels are parsed as strings.
      - Inline comments starting with '#' are ignored.
    """

    @staticmethod
    def _clean_lines(text: str) -> List[str]:
        out: List[str] = []
        for raw in text.splitlines():
            line = raw.split('#', 1)[0].strip()
            if line:
                out.append(line)
        return out

    @staticmethod
    def _parse_three_ints(line: str) -> Tuple[int, int, int]:
        parts = line.split()
        if len(parts) != 3:
            raise ValueError("Header must be exactly: N M R")
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            raise ValueError("Header contains non-integer values.")

    @staticmethod
    def _parse_two_labels(line: str) -> Tuple[str, str]:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("Second line must be exactly two labels: 's t'.")
        return parts[0], parts[1]

    @staticmethod
    def _parse_vertex_line(line: str) -> Tuple[str, bool]:
        """
        Accepts:
          'X'
          'X *'
          'X*'
        where X is a single token (no spaces).
        """
        line = line.strip()
        is_red = False
        if line.endswith('*'):
            is_red = True
            line = line[:-1].rstrip()
        parts = line.split()
        if len(parts) != 1:
            # e.g. "A *" becomes ["A"] after star-trim; if more, it's malformed
            if len(parts) == 2 and parts[1] == '*':
                # tolerate 'X *' variant
                return parts[0], True
            raise ValueError(f"Malformed vertex line: {line!r}")
        return parts[0], is_red

    @staticmethod
    def _split_edge(line: str) -> Tuple[str, str, str]:
        """
        Returns (style, u, v) where style in {'dir','und'} for '->'/'--'.
        Labels must be single tokens (no spaces). No regex used.
        """
        has_dir = '->' in line
        has_und = '--' in line
        if has_dir and has_und:
            raise ValueError(f"Edge mixes '->' and '--': {line!r}")
        if not (has_dir or has_und):
            raise ValueError(f"Edge must contain '->' or '--': {line!r}")

        if has_dir:
            parts = line.split('->')
            style = 'dir'
        else:
            parts = line.split('--')
            style = 'und'

        if len(parts) != 2:
            raise ValueError(f"Malformed edge line: {line!r}")

        u = parts[0].strip()
        v = parts[1].strip()
        # enforce single-token labels (no spaces inside)
        if not u or not v or (' ' in u) or (' ' in v):
            raise ValueError(f"Edge endpoints must be single tokens: {line!r}")
        return style, u, v

    @classmethod
    def from_string(cls, text: str) -> Graph:
        lines = cls._clean_lines(text)
        if len(lines) < 2:
            raise ValueError("Input too short: need header and 's t' line.")

        # 1) Header
        N, M, R = cls._parse_three_ints(lines[0])

        # 2) Start/End (string labels)
        s, t = cls._parse_two_labels(lines[1])

        # 3) N vertex lines
        if len(lines) < 2 + N:
            raise ValueError(f"Expected {N} vertex lines, got {len(lines)-2}.")
        vertex_lines = lines[2:2+N]

        vertices: Set[str] = set()
        red: Set[str] = set()
        for idx, row in enumerate(vertex_lines, 1):
            v, is_red = cls._parse_vertex_line(row)
            if v in vertices:
                raise ValueError(f"Duplicate vertex id: {v}")
            vertices.add(v)
            if is_red:
                red.add(v)

        # 4) M edge lines
        if len(lines) != 2 + N + M:
            raise ValueError(f"Expected {M} edge lines, got {len(lines)-(2+N)}.")
        edge_lines = lines[2+N:]

        style_seen: str | None = None
        dir_edges: List[Tuple[str, str]] = []
        und_edges: Set[FrozenSet[str]] = set()

        for row in edge_lines:
            style, u, v = cls._split_edge(row)
            if style_seen is None:
                style_seen = style
            elif style != style_seen:
                raise ValueError("Mixed edge styles detected across lines.")
            if u not in vertices or v not in vertices:
                raise ValueError(f"Edge references unknown vertex: {u} ? {v}")
            if style == 'dir':
                dir_edges.append((u, v))
            else:
                e = frozenset((u, v))
                if u == v:
                    # allow self-loop in undirected? If not desired, forbid here.
                    pass
                if e in und_edges:
                    raise ValueError(f"Duplicate undirected edge {u} -- {v}")
                und_edges.add(e)

        directed = (style_seen == 'dir')

        g = Graph(
            n=N, m=M, directed=directed, s=s, t=t,
            vertices=vertices, red=red, adj={}
        )

        if directed:
            seen: Set[Tuple[str, str]] = set()
            for u, v in dir_edges:
                if (u, v) in seen:
                    raise ValueError(f"Duplicate directed edge {u} -> {v}")
                seen.add((u, v))
                g.add_edge(u, v)
        else:
            for e in und_edges:
                u, v = tuple(e)
                g.add_edge(u, v)

        # sanity checks
        if len(red) != R:
            raise ValueError(f"Red-count mismatch: header R={R}, parsed {len(red)}.")
        g.validate()
        return g

    @classmethod
    def from_file(cls, path: str, encoding: str = "utf-8") -> Graph:
        with open(path, "r", encoding=encoding) as f:
            return cls.from_string(f.read())


# --- tiny demo ---
if __name__ == "__main__":
    sample = """\
100 8 47
start ender
ovule
ender *
topos *
merit
metes
cease
ethic *
smite
yummy *
bonks *
brook
waled *
libra *
blaze
tamer
verso *
wooed
hadst *
liras *
moods
amply *
goony *
sited *
crock
joked
coots *
beady *
fonts *
awash *
crook
brags
magus *
redip *
older
loins
showy
newly
nudes *
nuder *
quads *
algin *
carob *
pulpy
sneer
spark
typal *
rinds
cuber *
conch
route
waded
groks *
wowee *
gusty
whole
lavas
waged
socks
slain
cooed
slots
razor
faddy *
start
stamp
serfs
fight
dicot *
paint
leaky
darns
pairs
fuzes *
mires
rials *
polly *
mussy *
coati *
purge
vroom *
purls *
petit *
rally
titer *
lemon
wetly *
umbra
brack *
polis *
badly
kopek *
weave
waist
gimps *
curve
heres
ascot *
oaten *
redux *
usual
brook -- crook
waled -- waged
waled -- waded
wooed -- cooed
liras -- rials
crock -- crook
nudes -- nuder
waded -- waged
"""
    g = GraphParser.from_string(sample)
    print(g)
    print("Directed:", g.directed)
    print("s,t:", g.s, g.t)
    print("Red:", sorted(g.red))
    print("Neighbors of waded:", g.neighbors("waded"))
