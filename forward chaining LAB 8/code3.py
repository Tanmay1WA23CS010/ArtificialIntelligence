# fol_forward_chaining.py
# Forward chaining for FOL definite clauses with unification and standardizing variables

import re
import itertools
import math
from typing import List, Dict, Tuple, Optional, Set

# -----------------------
# Term classes + parser
# -----------------------

class Term:
    def apply(self, subst: Dict[str, "Term"]) -> "Term":
        raise NotImplementedError()
    def vars(self) -> Set[str]:
        raise NotImplementedError()

class Var(Term):
    def __init__(self, name: str):
        self.name = name
    def apply(self, subst):
        if self.name in subst:
            return subst[self.name].apply(subst)
        return self
    def vars(self):
        return {self.name}
    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name
    def __hash__(self):
        return hash(("Var", self.name))
    def __str__(self):
        return self.name

class Const(Term):
    def __init__(self, name: str):
        self.name = name
    def apply(self, subst):
        return self
    def vars(self):
        return set()
    def __eq__(self, other):
        return isinstance(other, Const) and self.name == other.name
    def __hash__(self):
        return hash(("Const", self.name))
    def __str__(self):
        return self.name

class Func(Term):
    def __init__(self, name: str, args: List[Term]):
        self.name = name
        self.args = args
    def apply(self, subst):
        return Func(self.name, [a.apply(subst) for a in self.args])
    def vars(self):
        s = set()
        for a in self.args:
            s |= a.vars()
        return s
    def __eq__(self, other):
        return isinstance(other, Func) and self.name == other.name and self.args == other.args
    def __hash__(self):
        return hash(("Func", self.name, tuple(self.args)))
    def __str__(self):
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(map(str, self.args))})"

TOKEN_RE = r"\s*([A-Za-z_][A-Za-z_0-9]*|\(|\)|,|\->)\s*"

def tokenize(s: str):
    toks = re.findall(TOKEN_RE, s)
    return [t for t in toks if t.strip()]

def parse_term_from_tokens(tokens, pos=0) -> Tuple[Term,int]:
    if pos >= len(tokens):
        raise ValueError("unexpected end")
    tok = tokens[pos]
    if not re.match(r"[A-Za-z_][A-Za-z_0-9]*", tok):
        raise ValueError("expected identifier")
    name = tok
    pos += 1
    if pos < len(tokens) and tokens[pos] == "(":
        pos += 1
        args = []
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1
            return Func(name, []), pos
        while True:
            arg, pos = parse_term_from_tokens(tokens, pos)
            args.append(arg)
            if pos >= len(tokens):
                raise ValueError("missing ')'")
            if tokens[pos] == ",":
                pos += 1
                continue
            if tokens[pos] == ")":
                pos += 1
                break
            raise ValueError("unexpected token " + tokens[pos])
        return Func(name, args), pos
    else:
        # variable if starts with lowercase, constant/function symbol if uppercase
        if name[0].islower():
            return Var(name), pos
        else:
            return Const(name), pos

def parse_atom(s: str) -> Func:
    tokens = tokenize(s)
    term, pos = parse_term_from_tokens(tokens, 0)
    if pos != len(tokens):
        raise ValueError("extra tokens")
    if not isinstance(term, Func):
        # treat zero-arg predicate as Func(name, [])
        return Func(str(term), [])
    return term

# -----------------------
# Unification (with occurs-check)
# -----------------------

def occurs_check(varname: str, term: Term, subst: Dict[str, Term]) -> bool:
    t = term.apply(subst)
    return varname in t.vars()

def unify_terms(t1: Term, t2: Term, subst: Dict[str, Term]) -> Optional[Dict[str, Term]]:
    # returns updated subst or None if failure
    # apply current substitution first
    t1 = t1.apply(subst)
    t2 = t2.apply(subst)
    if isinstance(t1, Var):
        if t1.name == t2.__dict__.get("name", None) and isinstance(t2, Var):
            return subst
        if occurs_check(t1.name, t2, subst):
            return None
        new = dict(subst)
        new[t1.name] = t2
        return new
    if isinstance(t2, Var):
        return unify_terms(t2, t1, subst)
    if isinstance(t1, Const) and isinstance(t2, Const):
        if t1.name == t2.name:
            return subst
        return None
    if isinstance(t1, Func) and isinstance(t2, Func):
        if t1.name != t2.name or len(t1.args) != len(t2.args):
            return None
        s = dict(subst)
        for a,b in zip(t1.args, t2.args):
            s = unify_terms(a, b, s)
            if s is None:
                return None
        return s
    return None

def unify_atoms(a: Func, b: Func, subst: Dict[str, Term]) -> Optional[Dict[str, Term]]:
    # predicate names and arity must match
    if a.name != b.name or len(a.args) != len(b.args):
        return None
    s = dict(subst)
    for x,y in zip(a.args, b.args):
        s = unify_terms(x, y, s)
        if s is None:
            return None
    return s

# -----------------------
# Clauses / KB
# -----------------------

class Rule:
    def __init__(self, antecedents: List[Func], consequent: Func):
        self.antecedents = antecedents
        self.consequent = consequent
    def __str__(self):
        if self.antecedents:
            return f"{' & '.join(map(str,self.antecedents))} -> {self.consequent}"
        else:
            return str(self.consequent)

# Standardize variables apart by renaming variables in rule to fresh names
_counter = 0
def fresh_var_name(base: str) -> str:
    global _counter
    _counter += 1
    return f"{base}_{_counter}"

def standardize_apart(rule: Rule) -> Rule:
    # collect variables in the rule
    varset = set()
    for a in rule.antecedents + [rule.consequent]:
        varset |= a.vars()
    mapping = {}
    for v in varset:
        mapping[v] = Var(fresh_var_name(v))
    # apply mapping to terms
    def remap_term(t: Term):
        if isinstance(t, Var):
            return mapping[t.name]
        if isinstance(t, Const):
            return t
        return Func(t.name, [remap_term(arg) for arg in t.args])
    antecedents = [remap_term(a) for a in rule.antecedents]
    consequent = remap_term(rule.consequent)
    return Rule(antecedents, consequent)

# -----------------------
# Forward chaining algorithm (FOL-FC-ASK)
# -----------------------

def fol_fc_ask(kb_facts: List[Func], kb_rules: List[Rule], query: Func) -> Optional[Dict[str, Term]]:
    """
    kb_facts: list of ground atoms (Func) -- constants only expected
    kb_rules: list of Rule objects (with Vars)
    query: atom to prove (may contain constants/vars)
    Returns substitution (dict var->Term) that proves query or None.
    """
    # KB as set for fast membership (use string repr for simplicity)
    known = list(kb_facts)  # allow duplicates semantics via list
    known_set = set(str(f) for f in known)
    print("Initial facts:")
    for f in known:
        print("  ", f)
    iteration = 0
    while True:
        iteration += 1
        new_facts = []
        # For each rule, standardized apart
        for rule in kb_rules:
            std_rule = standardize_apart(rule)
            n = len(std_rule.antecedents)
            # if no antecedents (a fact rule), just try consequent
            if n == 0:
                g = std_rule.consequent
                ground_g = g.apply({})  # no subst
                if str(ground_g) not in known_set:
                    new_facts.append(ground_g)
                continue
            # try to find substitutions that make each antecedent unify with some known fact
            # we will try all combinations of known facts of length n (with repetition allowed)
            for facts_combo in itertools.product(known, repeat=n):
                s = {}
                failed = False
                for pat, fact in zip(std_rule.antecedents, facts_combo):
                    s = unify_atoms(pat, fact, s)
                    if s is None:
                        failed = True
                        break
                if failed:
                    continue
                # s is a substitution making all antecedents match these facts
                # produce the instantiated consequent
                instantiated_consequent = std_rule.consequent.apply(s)
                if str(instantiated_consequent) not in known_set:
                    new_facts.append(instantiated_consequent)
        # deduplicate new_facts
        added_any = False
        for nf in new_facts:
            if str(nf) not in known_set:
                known.append(nf)
                known_set.add(str(nf))
                added_any = True
                print(f"[Iter {iteration}] Inferred: {nf}")
                # check whether query is satisfied by this new fact (allow query with variables or constants)
                s = {}
                unify_result = unify_atoms(query, nf, s)
                if unify_result is not None:
                    print("Query unified with inferred fact:", nf)
                    return unify_result
        # Also check existing facts for query (in case present at start)
        for f in known:
            res = unify_atoms(query, f, {})
            if res is not None:
                print("Query matches known fact:", f)
                return res
        if not added_any:
            print("No new facts inferred; stopping.")
            return None

# -----------------------
# Build the KB for the Robert example
# -----------------------

def build_robert_kb():
    facts = []
    rules = []
    # Given facts:
    # Existential instantiation: we assume T1 is a fresh constant for the missile that A owns
    # Owns(A, T1)
    # Missile(T1)
    facts.append(parse_atom("Owns(A, T1)"))
    facts.append(parse_atom("Missile(T1)"))
    # American(Robert)
    facts.append(parse_atom("American(Robert)"))
    # Enemy(A, America)
    facts.append(parse_atom("Enemy(A, America)"))

    # Rules:
    # Missile(x) => Weapon(x)
    r1 = Rule([parse_atom("Missile(x)")], parse_atom("Weapon(x)"))
    rules.append(r1)

    # All missiles were sold to country A by Robert:
    # For all x: Missile(x) ∧ Owns(A, x) ⇒ Sells(Robert, x, A)
    r2 = Rule([parse_atom("Missile(x)"), parse_atom("Owns(A, x)")], parse_atom("Sells(Robert, x, A)"))
    rules.append(r2)

    # Enemy(x, America) => Hostile(x)
    r3 = Rule([parse_atom("Enemy(x, America)")], parse_atom("Hostile(x)"))
    rules.append(r3)

    # American(p) ∧ Weapon(q) ∧ Sells(p, q, r) ∧ Hostile(r) ⇒ Criminal(p)
    r4 = Rule([
        parse_atom("American(p)"),
        parse_atom("Weapon(q)"),
        parse_atom("Sells(p, q, r)"),
        parse_atom("Hostile(r)")
    ], parse_atom("Criminal(p)"))
    rules.append(r4)

    return facts, rules

# -----------------------
# Demo run
# -----------------------

if __name__ == "__main__":
    kb_facts, kb_rules = build_robert_kb()
    query = parse_atom("Criminal(Robert)")
    print("\nProving query:", query)
    subst = fol_fc_ask(kb_facts, kb_rules, query)
    print("\nFinal result:")
    if subst is None:
        print("Could NOT prove", query)
    else:
        print("Proved", query, "with substitution:", subst)
