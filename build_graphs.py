"""
build_graphs.py  —  v3  (Rich Features)
========================================
Problem diagnosed:
  - integer_overflow graphs: avg 1.1 nodes  ← parser missing most of contract
  - reentrancy vs logic_error L2 = 0.94    ← features indistinguishable
  - All vulnerable classes collapse to type_id~26, everything else ~0

Fix:
  - Better regex parser that catches ALL function-level blocks
  - 16-dim node feature vector with vulnerability-specific signals
  - Graph-level feature vector (10-dim) fed into classifier separately
  - Call-before-state-write detection (reentrancy signature)
  - Arithmetic op detection (overflow signature)
  - Modifier/require detection (access control signature)

RUN:
  python build_graphs.py
  (delete graphs/ folder first if rebuilding from scratch)
"""

import os, re, json
import pandas as pd
from pathlib import Path

CONTRACTS_INDEX = "contracts_index.csv"
CONTRACTS_DIR   = "contracts"
GRAPHS_DIR      = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ── Node type vocabulary ────────────────────────────────────────
NODE_TYPES = {
    "ContractDefinition":  0,
    "FunctionDefinition":  1,
    "ModifierDefinition":  2,
    "EventDefinition":     3,
    "StateVariable":       4,
    "LocalVariable":       5,
    "Parameter":           6,
    "ReturnStatement":     7,
    "IfStatement":         8,
    "ForLoop":             9,
    "WhileLoop":           10,
    "DoWhileLoop":         11,
    "ExternalCall":        12,
    "InternalCall":        13,
    "TransferCall":        14,
    "SendCall":            15,
    "DelegatecallNode":    16,
    "RequireStatement":    17,
    "RevertStatement":     18,
    "AssertStatement":     19,
    "AssignStatement":     20,
    "ArithmeticOp":        21,
    "MappingAccess":       22,
    "EmitStatement":       23,
    "Modifier":            24,
    "PayableFunction":     25,
    "Constructor":         26,
    "Fallback":            27,
    "Receive":             28,
    "Unknown":             29,
}
N_TYPES = len(NODE_TYPES)  # 30

# ── Vulnerability-specific keyword sets ────────────────────────
REENTRANCY_KEYWORDS    = {"call", "transfer", "send", "fallback", "receive",
                           "external", "payable", "withdraw", "reentr"}
OVERFLOW_KEYWORDS      = {"overflow", "underflow", "safemath", "unchecked",
                           "add", "sub", "mul", "div", "mod", "uint", "int"}
ACCESS_KEYWORDS        = {"onlyowner", "require", "modifier", "revert",
                           "access", "role", "admin", "owner", "auth",
                           "permission", "whitelist", "blacklist"}
ORACLE_KEYWORDS        = {"oracle", "price", "getprice", "latestanswer",
                           "chainlink", "twap", "feed", "pricefeed"}
LOGIC_KEYWORDS         = {"logic", "business", "condition", "check",
                           "assert", "invariant", "incorrect"}


def normalize(val, max_val):
    return min(float(val) / max_val, 1.0) if max_val > 0 else 0.0


# ════════════════════════════════════════════════════════════════
#   PARSER  —  extracts nodes and edges from raw Solidity text
# ════════════════════════════════════════════════════════════════

def parse_solidity(source: str, vuln_type: str):
    """
    Parse Solidity source into nodes + edges.
    Returns (nodes, edges) where:
      nodes = list of {id, type, name, features: [16 floats]}
      edges = list of {src, dst, type}
    """
    nodes = []
    edges = []
    node_id = 0

    def add_node(ntype, name, features):
        nonlocal node_id
        n = {"id": node_id, "type": ntype, "name": name, "features": features}
        nodes.append(n)
        node_id += 1
        return node_id - 1

    def add_edge(src, dst, etype):
        edges.append({"src": src, "dst": dst, "type": etype})

    # ── Remove comments ────────────────────────────────────────
    source = re.sub(r"//[^\n]*",        "",  source)
    source = re.sub(r"/\*.*?\*/",       "",  source, flags=re.DOTALL)
    source_lower = source.lower()

    # ── Contract-level node ────────────────────────────────────
    contract_names = re.findall(
        r'\bcontract\s+(\w+)', source, re.IGNORECASE)
    contract_name = contract_names[0] if contract_names else "Unknown"
    contract_nid  = add_node("ContractDefinition", contract_name,
                              make_features("ContractDefinition", source,
                                            vuln_type, depth=0))

    # ── State variables ────────────────────────────────────────
    state_var_pattern = re.compile(
        r'^\s*(uint\d*|int\d*|address|bool|bytes\d*|string|mapping[^;]+)\s+'
        r'(?:public\s+|private\s+|internal\s+)?(\w+)\s*[=;]',
        re.MULTILINE
    )
    state_vars = {}  # name → node_id
    for m in state_var_pattern.finditer(source):
        vname = m.group(2)
        nid   = add_node("StateVariable", vname,
                         make_features("StateVariable", m.group(0),
                                       vuln_type, depth=1))
        add_edge(contract_nid, nid, "AST_CHILD")
        state_vars[vname] = nid

    # ── Functions (the most important unit) ───────────────────
    func_pattern = re.compile(
        r'(function\s+(\w+)\s*\([^)]*\)[^{]*?)\{',
        re.DOTALL
    )
    # Also catch constructor, fallback, receive
    special_pattern = re.compile(
        r'\b(constructor|fallback|receive)\s*\([^)]*\)[^{]*?\{',
        re.DOTALL
    )

    func_bodies = []

    # Extract function bodies by brace matching
    def extract_body(src, start):
        depth = 0
        i = start
        while i < len(src):
            if src[i] == '{':
                depth += 1
            elif src[i] == '}':
                depth -= 1
                if depth == 0:
                    return src[start:i+1]
            i += 1
        return src[start:]

    for m in func_pattern.finditer(source):
        sig  = m.group(1)
        name = m.group(2)
        body = extract_body(source, m.start())
        func_bodies.append(("FunctionDefinition", name, sig, body))

    for m in special_pattern.finditer(source):
        kind = m.group(1).capitalize()
        body = extract_body(source, m.start())
        func_bodies.append((kind, m.group(1), m.group(0), body))

    # If no functions found (happens with minimal contracts),
    # treat the whole source as one function block
    if not func_bodies:
        func_bodies.append(("FunctionDefinition", "main", source, source))

    for (ftype, fname, sig, fbody) in func_bodies:
        sig_lower  = sig.lower()
        body_lower = fbody.lower()

        # Determine node type more precisely
        if "payable" in sig_lower and ("withdraw" in fname.lower()
                                        or "transfer" in fname.lower()):
            ntype = "PayableFunction"
        elif fname.lower() in ("constructor", "fallback", "receive"):
            ntype = fname.capitalize()
        else:
            ntype = ftype

        func_nid = add_node(ntype, fname,
                            make_features(ntype, fbody, vuln_type, depth=1))
        add_edge(contract_nid, func_nid, "AST_CHILD")

        # ── Statements inside function ─────────────────────────
        prev_stmt_nid = None

        # require / revert / assert
        for kw in ("require", "revert", "assert"):
            for _ in re.finditer(rf'\b{kw}\s*\(', fbody, re.IGNORECASE):
                ntype_s = {"require": "RequireStatement",
                           "revert":  "RevertStatement",
                           "assert":  "AssertStatement"}[kw]
                sid = add_node(ntype_s, kw,
                               make_features(ntype_s, fbody, vuln_type, depth=2))
                add_edge(func_nid, sid, "AST_CHILD")
                if prev_stmt_nid is not None:
                    add_edge(prev_stmt_nid, sid, "CFG_NEXT")
                prev_stmt_nid = sid
                break  # one per function to avoid explosion

        # external calls: .call{}, .call(), transfer(), send()
        call_patterns = [
            (r'\.call\s*[\(\{]',          "ExternalCall",  "external_call"),
            (r'\.transfer\s*\(',           "TransferCall",  "transfer_call"),
            (r'\.send\s*\(',               "SendCall",      "send_call"),
            (r'\.delegatecall\s*\(',       "DelegatecallNode", "delegatecall"),
        ]
        call_nids = []
        for pattern, ctype, _ in call_patterns:
            for _ in re.finditer(pattern, fbody, re.IGNORECASE):
                cid = add_node(ctype, ctype,
                               make_features(ctype, fbody, vuln_type, depth=2))
                add_edge(func_nid, cid, "CALL_DEP")
                call_nids.append(cid)
                if prev_stmt_nid is not None:
                    add_edge(prev_stmt_nid, cid, "CFG_NEXT")
                prev_stmt_nid = cid
                break  # one per type per function

        # state variable assignments after calls → reentrancy edge
        assign_after_call = re.search(
            r'(\.call|\.transfer|\.send)[^;]*;[^;]*(\w+)\s*[+\-]?=',
            fbody, re.DOTALL
        )
        if assign_after_call and call_nids:
            # This is THE reentrancy signature: call before state update
            for vname, vnid in state_vars.items():
                if vname.lower() in body_lower:
                    add_edge(call_nids[-1], vnid, "DATA_DEP")

        # arithmetic operations (overflow signature)
        arith_ops = re.findall(
            r'(\w+)\s*[\+\-\*\/\%]=|(\w+)\s*=\s*\w+\s*[\+\-\*\/\%]', fbody)
        if arith_ops:
            aid = add_node("ArithmeticOp", "arithmetic",
                           make_features("ArithmeticOp", fbody, vuln_type, depth=2))
            add_edge(func_nid, aid, "AST_CHILD")
            if prev_stmt_nid is not None:
                add_edge(prev_stmt_nid, aid, "CFG_NEXT")
            prev_stmt_nid = aid

        # if/else branches
        for _ in re.finditer(r'\bif\s*\(', fbody, re.IGNORECASE):
            iid = add_node("IfStatement", "if",
                           make_features("IfStatement", fbody, vuln_type, depth=2))
            add_edge(func_nid, iid, "AST_CHILD")
            if prev_stmt_nid is not None:
                add_edge(prev_stmt_nid, iid, "CFG_BRANCH")
            prev_stmt_nid = iid
            break  # one per function

        # loops (overflow / logic error patterns)
        for kw in ("for", "while"):
            if re.search(rf'\b{kw}\s*\(', fbody, re.IGNORECASE):
                ltype = "ForLoop" if kw == "for" else "WhileLoop"
                lid   = add_node(ltype, kw,
                                 make_features(ltype, fbody, vuln_type, depth=2))
                add_edge(func_nid, lid, "AST_CHILD")
                if prev_stmt_nid is not None:
                    add_edge(prev_stmt_nid, lid, "CFG_NEXT")
                prev_stmt_nid = lid
                break

        # emit events
        if re.search(r'\bemit\s+\w+', fbody, re.IGNORECASE):
            eid = add_node("EmitStatement", "emit",
                           make_features("EmitStatement", fbody, vuln_type, depth=2))
            add_edge(func_nid, eid, "AST_CHILD")

    # ── Modifiers ─────────────────────────────────────────────
    for m in re.finditer(r'\bmodifier\s+(\w+)', source, re.IGNORECASE):
        mid = add_node("ModifierDefinition", m.group(1),
                       make_features("ModifierDefinition", source,
                                     vuln_type, depth=1))
        add_edge(contract_nid, mid, "AST_CHILD")

    return nodes, edges


# ════════════════════════════════════════════════════════════════
#   16-DIM NODE FEATURE VECTOR
#   Designed so each vulnerability class has a DISTINCT signature
# ════════════════════════════════════════════════════════════════

def make_features(node_type, context, vuln_type, depth):
    """
    Returns a 16-float feature vector for a node.

    Dim  0  : node_type_id normalised (0-1)
    Dim  1  : has external call in context
    Dim  2  : has transfer/send in context
    Dim  3  : has delegatecall in context
    Dim  4  : modifies state variable (= assignment)
    Dim  5  : is public/external function
    Dim  6  : depth normalised (0-1, max depth assumed 5)
    Dim  7  : call-before-state-write pattern (REENTRANCY signal)
    Dim  8  : has arithmetic op (+,-,*,/) (OVERFLOW signal)
    Dim  9  : has unchecked block (OVERFLOW signal)
    Dim 10  : has require/revert/modifier (ACCESS CONTROL signal)
    Dim 11  : has oracle/price keyword (ORACLE signal)
    Dim 12  : is payable
    Dim 13  : has loop
    Dim 14  : reentrancy keyword density (0-1)
    Dim 15  : overflow keyword density (0-1)
    """
    ctx = context.lower()
    ntype_id = NODE_TYPES.get(node_type, NODE_TYPES["Unknown"])

    # Dim 0 — node type normalised
    f0 = normalize(ntype_id, N_TYPES)

    # Dim 1 — external call
    f1 = 1.0 if re.search(r'\.call\s*[\(\{]', ctx) else 0.0

    # Dim 2 — transfer / send
    f2 = 1.0 if re.search(r'\.(transfer|send)\s*\(', ctx) else 0.0

    # Dim 3 — delegatecall
    f3 = 1.0 if "delegatecall" in ctx else 0.0

    # Dim 4 — state modification (assignment)
    f4 = 1.0 if re.search(r'\b\w+\s*[+\-\*\/]?=\s*', ctx) else 0.0

    # Dim 5 — public / external visibility
    f5 = 1.0 if re.search(r'\b(public|external)\b', ctx) else 0.0

    # Dim 6 — depth normalised
    f6 = normalize(depth, 5)

    # Dim 7 — call BEFORE state write (THE reentrancy signature)
    #         Pattern: call/transfer/send appears, then an assignment
    call_pos    = re.search(r'\.(call|transfer|send)\s*[\(\{]', ctx)
    assign_pos  = re.search(r'\w+\s*[+\-]?=(?!=)', ctx)
    if call_pos and assign_pos and call_pos.start() < assign_pos.start():
        f7 = 1.0
    else:
        f7 = 0.0

    # Dim 8 — arithmetic operation (overflow signal)
    f8 = 1.0 if re.search(
        r'\w+\s*[\+\-\*\/\%]=|\w+\s*=\s*\w+\s*[\+\-\*\/\%]\s*\w+', ctx
    ) else 0.0

    # Dim 9 — unchecked block (Solidity 0.8+ overflow)
    f9 = 1.0 if "unchecked" in ctx else 0.0

    # Dim 10 — require / revert / modifier (access control signal)
    f10 = 1.0 if re.search(r'\b(require|revert|modifier|onlyowner)\b', ctx) else 0.0

    # Dim 11 — oracle / price manipulation signal
    f11 = 1.0 if re.search(
        r'\b(oracle|getprice|latestanswer|pricefeed|twap)\b', ctx
    ) else 0.0

    # Dim 12 — payable
    f12 = 1.0 if "payable" in ctx else 0.0

    # Dim 13 — loop present
    f13 = 1.0 if re.search(r'\b(for|while)\s*\(', ctx) else 0.0

    # Dim 14 — reentrancy keyword density (count/100)
    re_count = sum(ctx.count(kw) for kw in REENTRANCY_KEYWORDS)
    f14 = normalize(re_count, 100)

    # Dim 15 — overflow keyword density (count/100)
    ov_count = sum(ctx.count(kw) for kw in OVERFLOW_KEYWORDS)
    f15 = normalize(ov_count, 100)

    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]


# ════════════════════════════════════════════════════════════════
#   10-DIM GRAPH-LEVEL FEATURE VECTOR
#   Captures whole-contract properties — fed separately into
#   the classifier head alongside the GAT node embeddings
# ════════════════════════════════════════════════════════════════

def make_graph_features(source, vuln_type):
    """
    Returns a 10-float graph-level feature vector.
    These are properties of the WHOLE contract, not individual nodes.

    Dim  0  : external call count normalised
    Dim  1  : call-before-state-write occurrences (reentrancy)
    Dim  2  : arithmetic op count normalised (overflow)
    Dim  3  : unchecked block present
    Dim  4  : require/modifier count normalised (access control)
    Dim  5  : oracle/price keyword count (oracle manipulation)
    Dim  6  : function count normalised
    Dim  7  : payable function count normalised
    Dim  8  : loop count normalised
    Dim  9  : safemath / openzeppelin usage
    """
    ctx = source.lower()

    # Dim 0 — external calls
    ext_calls = len(re.findall(r'\.call\s*[\(\{]', ctx))
    f0 = normalize(ext_calls, 20)

    # Dim 1 — call-before-state-write pattern count
    cbsw = len(re.findall(
        r'(\.call|\.transfer|\.send)[^;]{0,200}?\w+\s*[+\-]?=',
        ctx, re.DOTALL
    ))
    f1 = normalize(cbsw, 10)

    # Dim 2 — arithmetic ops
    arith = len(re.findall(
        r'\w+\s*[\+\-\*\/\%]=|\w+\s*=\s*\w+\s*[\+\-\*\/\%]\s*\w+', ctx))
    f2 = normalize(arith, 50)

    # Dim 3 — unchecked block
    f3 = 1.0 if "unchecked" in ctx else 0.0

    # Dim 4 — require / modifier usage
    req_count = ctx.count("require") + ctx.count("revert") + ctx.count("modifier")
    f4 = normalize(req_count, 20)

    # Dim 5 — oracle keywords
    oracle_count = sum(ctx.count(k) for k in
                       ["oracle", "getprice", "latestanswer", "pricefeed", "twap"])
    f5 = normalize(oracle_count, 10)

    # Dim 6 — function count
    func_count = len(re.findall(r'\bfunction\s+\w+', ctx))
    f6 = normalize(func_count, 30)

    # Dim 7 — payable functions
    payable_count = len(re.findall(r'\bpayable\b', ctx))
    f7 = normalize(payable_count, 10)

    # Dim 8 — loops
    loop_count = len(re.findall(r'\b(for|while)\s*\(', ctx))
    f8 = normalize(loop_count, 10)

    # Dim 9 — SafeMath / OpenZeppelin
    f9 = 1.0 if re.search(r'\b(safemath|openzeppelin|safetransfer)\b', ctx) else 0.0

    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]


# ════════════════════════════════════════════════════════════════
#   LABEL MAP
# ════════════════════════════════════════════════════════════════

LABEL_MAP = {
    "safe":             0,
    "reentrancy":       1,
    "access_control":   2,
    "integer_overflow": 3,
    "logic_error":      4,
    "flash_loan":       5,
    "oracle_manipulation": 5,
    "bridge_hack":      5,
    "unknown":          5,
    "other_vuln":       5,
}


# ════════════════════════════════════════════════════════════════
#   MAIN BUILD LOOP
# ════════════════════════════════════════════════════════════════

def build_all_graphs():
    df = pd.read_csv(CONTRACTS_INDEX)
    print(f"Loaded {len(df)} contracts from {CONTRACTS_INDEX}")

    stats = {"built": 0, "skipped": 0}

    for _, row in df.iterrows():
        filename  = str(row.get("filename", ""))
        vuln_type = str(row.get("vuln_type", "unknown")).lower().strip()
        label_raw = int(row.get("label", 0))

        # Map vuln_type to integer label
        label = LABEL_MAP.get(vuln_type, 5)

        # Build sol file path
        sol_path =filename
        if not os.path.exists(sol_path):
            print(f"  [skip] not found: {sol_path}")
            stats["skipped"] += 1
            continue

        # Read source
        with open(sol_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()

        if len(source.strip()) < 20:
            print(f"  [skip] empty file: {filename}")
            stats["skipped"] += 1
            continue

        # Parse into nodes + edges
        try:
            nodes, edges = parse_solidity(source, vuln_type)
        except Exception as e:
            print(f"  [err] {filename}: {e}")
            stats["skipped"] += 1
            continue

        # Graph-level feature vector
        graph_features = make_graph_features(source, vuln_type)

        # Build output JSON
        graph = {
            "filename":      filename,
            "label":         label,
            "vuln_type":     vuln_type,
            "method":        "rich_regex_v3",
            "num_nodes":     len(nodes),
            "num_edges":     len(edges),
            "nodes":         nodes,
            "edges":         edges,
            "graph_features": graph_features,   # ← NEW: 10-dim graph vector
        }

        # Save
        out_name = Path(filename).stem + ".json"
        out_path = os.path.join(GRAPHS_DIR, out_name)
        with open(out_path, "w") as f:
            json.dump(graph, f)

        stats["built"] += 1

    print(f"\nDone. Built={stats['built']}  Skipped={stats['skipped']}")


def print_summary():
    """Quick sanity check after building."""
    import glob
    from collections import defaultdict, Counter

    files = glob.glob(f"{GRAPHS_DIR}/*.json")
    print(f"\nGraph files: {len(files)}")

    class_stats = defaultdict(list)
    for fp in files[:200]:
        with open(fp) as f:
            g = json.load(f)
        class_stats[g["vuln_type"]].append((g["num_nodes"], g["num_edges"]))

    print(f"\n{'CLASS':<25} {'N':>5} {'AVG_NODES':>10} {'AVG_EDGES':>10}")
    print("-" * 55)
    for cls, vals in sorted(class_stats.items()):
        avg_n = sum(v[0] for v in vals) / len(vals)
        avg_e = sum(v[1] for v in vals) / len(vals)
        print(f"{cls:<25} {len(vals):>5} {avg_n:>10.1f} {avg_e:>10.1f}")

    # Feature mean check — should now be distinct
    print(f"\nFeature vector means (first 8 dims) per class:")
    print(f"{'CLASS':<20} [type  call  send  delg  stat  pub   dep   cbsw  arith unck]")
    print("-" * 80)
    class_feats = defaultdict(list)
    for fp in files[:200]:
        with open(fp) as f:
            g = json.load(f)
        for node in g["nodes"]:
            class_feats[g["vuln_type"]].append(node["features"])

    import numpy as np
    for cls, feat_list in sorted(class_feats.items()):
        arr  = np.array(feat_list)
        mean = arr.mean(axis=0)
        print(f"{cls:<20} {[round(float(x),3) for x in mean[:10]]}")


if __name__ == "__main__":
    build_all_graphs()
    print_summary()