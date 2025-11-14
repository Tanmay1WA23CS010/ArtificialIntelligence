# fol_unify.py
# A simple First-Order Logic unifier with parser and occurs-check.

import re
from typing import List, Dict, Optional, Union, Tuple

# ---------------------------
# Term classes
# ---------------------------

class Term:
    def apply_subst(self, subst: Dict["Variable", "Term"]) -> "Term":
        raise NotImplementedError()

    def vars(self) -> set:
        """Return set of Variable objects occurring in this term."""
        raise NotImplementedError()

    def __repr__(self):
        return self.__str__()

class Variable(Term):
    def __init__(self, name: str):
        self.name = name

    def apply_subst(self, subst: Dict["Variable", Term]) -> Term:
        # If variable is in substitution map, apply substitution (and then apply recursively)
        for v in subst:
            # use name equality for lookup convenience
            if v.name == self.name:
                return subst[v].apply_subst(subst)
        return self

    def vars(self) -> set:
        return {self}

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name

    def __hash__(self):
        return hash(("Var", self.name))

    def __str__(self):
        return self.name

class Constant(Term):
    def __init__(self, name: str):
        self.name = name

    def apply_subst(self, subst: Dict[Variable, Term]) -> Term:
        return self  # constants unaffected

    def vars(self) -> set:
        return set()

    def __eq__(self, other):
        return isinstance(other, Constant) and self.name == other.name

    def __hash__(self):
        return hash(("Const", self.name))

    def __str__(self):
        return self.name

class Function(Term):
    def __init__(self, name: str, args: List[Term]):
        self.name = name
        self.args = args

    def apply_subst(self, subst: Dict[Variable, Term]) -> Term:
        return Function(self.name, [arg.apply_subst(subst) for arg in self.args])

    def vars(self) -> set:
        s = set()
        for a in self.args:
            s |= a.vars()
        return s

    def __eq__(self, other):
        return isinstance(other, Function) and self.name == other.name and self.args == other.args

    def __hash__(self):
        return hash(("Func", self.name, tuple(self.args)))

    def __str__(self):
        if len(self.args) == 0:
            return self.name
        return f"{self.name}({', '.join(map(str, self.args))})"


# ---------------------------
# Substitution utilities
# ---------------------------

Subst = Dict[Variable, Term]

def compose_subst(s1: Subst, s2: Subst) -> Subst:
    """
    Return composition s = s1 ∘ s2 meaning apply s2 then s1.
    Implemented so that each term in s1 has s2 applied, and we keep s2 entries that aren't overridden.
    """
    new = {}
    # Apply s2 to all terms in s1
    for v, t in s1.items():
        new_v = v
        new_t = t.apply_subst(s2)
        new[new_v] = new_t
    # Add entries from s2 that are not in s1
    for v, t in s2.items():
        if v not in new:
            new[v] = t
    return new

def apply_subst_to_term(term: Term, subst: Subst) -> Term:
    return term.apply_subst(subst)


# ---------------------------
# Occurs-check
# ---------------------------

def occurs_check(var: Variable, term: Term, subst: Subst) -> bool:
    """
    Check whether variable `var` occurs in `term` after applying current substitution `subst`.
    Returns True if occurs -> then cannot bind var to term (would create a cyclic substitution).
    """
    term_applied = term.apply_subst(subst)
    return var in term_applied.vars()


# ---------------------------
# Unification algorithm (Robinson's algorithm with occurs-check)
# ---------------------------

def unify(t1: Term, t2: Term, subst: Optional[Subst] = None) -> Optional[Subst]:
    """
    Attempt to unify terms t1 and t2 given initial substitution subst.
    Returns the most general unifier (a substitution dict Variable -> Term) or None on failure.
    """
    if subst is None:
        subst = {}

    # Worklist of pairs
    pairs: List[Tuple[Term, Term]] = [(t1, t2)]
    current_subst: Subst = dict(subst)  # copy

    while pairs:
        s, t = pairs.pop(0)
        # apply current substitution
        s = s.apply_subst(current_subst)
        t = t.apply_subst(current_subst)

        # print("Debug pair:", s, t)  # uncomment for step debugging

        if s == t:
            continue

        if isinstance(s, Variable):
            if occurs_check(s, t, current_subst):
                return None  # failure due to occurs-check
            # add substitution s -> t
            current_subst = compose_subst({s: t}, current_subst)
            continue

        if isinstance(t, Variable):
            if occurs_check(t, s, current_subst):
                return None
            current_subst = compose_subst({t: s}, current_subst)
            continue

        # Both are functions or constants
        if isinstance(s, Constant) and isinstance(t, Constant):
            # different constants can't be unified
            if s.name != t.name:
                return None
            else:
                continue

        if isinstance(s, Function) and isinstance(t, Function):
            if s.name != t.name or len(s.args) != len(t.args):
                return None
            # push pairwise arguments
            pairs = [(sa, ta) for sa, ta in zip(s.args, t.args)] + pairs
            continue

        # other cases not unifyable
        return None

    return current_subst


# ---------------------------
# Simple parser for terms and predicates
# ---------------------------

TOKEN_REGEX = r"\s*([A-Za-z_][A-Za-z_0-9]*|\(|\)|,)\s*"

def tokenize(s: str) -> List[str]:
    tokens = re.findall(TOKEN_REGEX, s)
    return [t for t in tokens if t.strip() != ""]

def parse_term_from_tokens(tokens: List[str], pos: int = 0) -> Tuple[Term, int]:
    """
    Parse a term starting at tokens[pos].
    Returns (term, new_pos)
    Grammar (simple):
      term ::= ID | ID '(' term (',' term)* ')'
    We decide variable vs constant/function name by identifier's first character:
      - If starts with lowercase letter -> Variable
      - Else -> Constant or Function (if followed by '(')
    """
    if pos >= len(tokens):
        raise ValueError("Unexpected end of tokens")

    token = tokens[pos]
    if re.match(r"[A-Za-z_][A-Za-z_0-9]*", token) is None:
        raise ValueError(f"Expected identifier at pos {pos}, found {token}")

    name = token
    pos += 1
    # function / predicate with args?
    if pos < len(tokens) and tokens[pos] == "(":
        pos += 1  # skip '('
        args = []
        # handle empty-arg functions (rare)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1
            term = Function(name, [])
            return term, pos

        while True:
            arg, pos = parse_term_from_tokens(tokens, pos)
            args.append(arg)
            if pos >= len(tokens):
                raise ValueError("Missing closing ')' in function")
            if tokens[pos] == ",":
                pos += 1
                continue
            elif tokens[pos] == ")":
                pos += 1
                break
            else:
                raise ValueError(f"Unexpected token {tokens[pos]} in args")
        term = Function(name, args)
        return term, pos
    else:
        # no args -> variable or constant
        if name[0].islower():
            return Variable(name), pos
        else:
            return Constant(name), pos

def parse_term(s: str) -> Term:
    tokens = tokenize(s)
    term, pos = parse_term_from_tokens(tokens, 0)
    if pos != len(tokens):
        raise ValueError("Extra tokens after parsing: " + " ".join(tokens[pos:]))
    return term

# ---------------------------
# Utility to pretty-print substitution
# ---------------------------

def subst_to_str(subst: Optional[Subst]) -> str:
    if subst is None:
        return "Fail"
    if not subst:
        return "{}"
    items = []
    for v, t in subst.items():
        items.append(f"{v} -> {t}")
    return "{ " + ", ".join(items) + " }"


# ---------------------------
# Demonstration / Tests
# ---------------------------

if __name__ == "__main__":
    examples = [
        ("Eats(x, Apple)", "Eats(Riya, y)"),
        ("p(f(a), g(Y))", "p(X, X)"),           # should fail (example in prompt)
        ("Knows(John, x)", "Knows(x, Elisabeth)"),  # should fail
        ("P(x, h(y))", "P(a, f(z))"),           # should fail because h != f
        ("f(x, x)", "f(a, b)"),                 # fail: x must be both a and b
        ("parent(John, Mary)", "parent(John, Mary)"), # trivial unify
        ("q(X, g(Y))", "q(f(a), g(b))"),        # Y->b, X->f(a)
        ("r(X)", "r(f(X))"),                    # occurs-check failure
    ]

    for a, b in examples:
        t1 = parse_term(a)
        t2 = parse_term(b)
        print("------------------------------------------------")
        print("Unify:", a, "  WITH  ", b)
        result = unify(t1, t2)
        print("Result:", subst_to_str(result))
