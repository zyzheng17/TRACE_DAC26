#!/usr/bin/env python3
"""Prepare RTL contrastive datasets from raw RTL designs.

The pipeline mirrors the CircuitFusion-based flow used for TRACE RTL experiments:
raw RTL -> transformed positive RTL -> register endpoints -> RTL cones ->
cone graphs with Graphormer and DG5 fields -> DGL paired train/valid pickles.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
RTL_ROOT = ROOT.parent
TRACE_SRC = RTL_ROOT / "src"
CIRCUITFUSION = ROOT / "circuitfusion"
DESIGN_GRAPH_SCR = CIRCUITFUSION / "data_collect" / "rtl" / "dataset" / "scr_design2graph"
CONE_GRAPH_SCR = CIRCUITFUSION / "data_collect" / "rtl" / "rtl2graph" / "scr"
DATASET_GRAPH_SCR = CIRCUITFUSION / "dataset" / "dataset_graph"
DEFAULT_DESIGNS = ROOT / "config" / "default_designs.json"
DEFAULT_SPLIT_DIR = CIRCUITFUSION / "dataset" / "dataset_js"


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def load_designs(path: Path, only: list[str] | None = None) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        designs = {name: name for name in data}
    elif isinstance(data, dict):
        designs = {str(name): str(top) for name, top in data.items()}
    else:
        raise ValueError("Design manifest must be a JSON object or list")
    if only:
        missing = [name for name in only if name not in designs]
        if missing:
            raise ValueError(f"Designs not present in manifest: {missing}")
        designs = {name: designs[name] for name in only}
    return designs


def run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(str(item) for item in command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def clean_verilog_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"\(\*(.*?)\*\)", "", text, flags=re.DOTALL)
    lines = [line for line in text.splitlines() if line.strip()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_registers(verilog_file: Path) -> list[str]:
    content = verilog_file.read_text(encoding="utf-8", errors="ignore")
    content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    reg_pattern = r"reg\s+(?:\[([^\]]+)\]\s+)?([a-zA-Z_][a-zA-Z0-9_]*\s*(?:,\s*[a-zA-Z_][a-zA-Z0-9_]*\s*)*)\s*;"
    registers: set[str] = set()
    for match in re.finditer(reg_pattern, content, re.MULTILINE | re.IGNORECASE):
        for name in match.group(2).split(","):
            name = name.strip()
            if name:
                registers.add(name)
    return sorted(registers)


def stage_registers(designs: dict[str, str], ori_dir: Path, reg_dir: Path) -> None:
    reg_dir.mkdir(parents=True, exist_ok=True)
    for design in designs:
        rtl_file = ori_dir / f"{design}.v"
        if not rtl_file.exists():
            print(f"[registers] skip missing {rtl_file}")
            continue
        registers = extract_registers(rtl_file)
        out_file = reg_dir / f"{design}.json"
        out_file.write_text(json.dumps(registers, ensure_ascii=False), encoding="utf-8")
        print(f"[registers] {design}: {len(registers)} endpoints -> {out_file}")


def stage_transform(designs: dict[str, str], ori_dir: Path, pos_dir: Path, yosys_bin: str, tmp_dir: Path) -> None:
    pos_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for design, top in designs.items():
        src = ori_dir / f"{design}.v"
        dst = pos_dir / f"{design}.v"
        if not src.exists():
            print(f"[transform] skip missing {src}")
            continue
        script = tmp_dir / f"transform_{design}.ys"
        script.write_text(
            "\n".join(
                [
                    f"read_verilog -sv {src}",
                    f"hierarchy -check -top {top}",
                    "proc",
                    "opt_expr",
                    "opt_merge",
                    "opt_reduce",
                    "opt_clean",
                    f"write_verilog {dst}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run_command([yosys_bin, "-s", str(script)])
        clean_verilog_file(dst)


def stage_whole_graph(designs: dict[str, str], rtl_dir: Path, cmd: str, rtl_graph_root: Path) -> None:
    out_dir = rtl_graph_root / cmd
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_script = DESIGN_GRAPH_SCR / "analyze.py"
    for design in designs:
        src = rtl_dir / f"{design}.v"
        if not src.exists():
            print(f"[whole-graph:{cmd}] skip missing {src}")
            continue
        run_command(
            [sys.executable, str(analyze_script), str(src), "-N", design, "-C", cmd, "-O", str(out_dir) + "/"],
            cwd=DESIGN_GRAPH_SCR,
        )


def get_coi_signal(line: str) -> set[str]:
    ret_set: set[str] = set()
    for signal in set(re.split(r"[ ]+", line)):
        signal = re.sub(r",$", "", signal)
        ps_re = re.findall(r"(\S+)\[(\d+):(\d+)\]$", signal)
        ps_re2 = re.findall(r"(\S+)\[(\D+)\[", signal)
        ptr_re = re.findall(r"(\S+)\[(\d+)\]$", signal)
        if re.findall(r"(\d+)'", signal) or re.findall(r"^\[(\d+)\]$", signal):
            continue
        if re.findall(r"^\[(\d+):(\d+)\]$", signal):
            continue
        if ps_re:
            signal = ps_re[0][0]
        if ps_re2:
            for item in ps_re2[0][:2]:
                if re.findall(r"[A-Za-z0-9_\.]", item):
                    ret_set.add(item)
            continue
        if ptr_re:
            signal = ptr_re[0][0]
        if re.findall(r"[A-Za-z0-9_\.]", signal):
            ret_set.add(signal)
    return ret_set


def add_signal_to_cone(signal_name: str, lines: list[str], state: dict, not_endpoint: bool = True) -> None:
    signal_name = re.sub(r"\\\\", "", signal_name)
    signal_name = re.sub(r"\\", "", signal_name)
    signal_pattern = re.escape(signal_name)
    for idx, line in enumerate(lines):
        def_line = re.findall(rf"^  (reg|wire|input|output|inout)(.*)(\s*){signal_pattern}(\s*);", line)
        def_reg_line = re.findall(rf"^  reg(.*)(\s+){signal_pattern}(\s*);", line)
        assign_seq_line2 = re.findall(rf"{signal_pattern}(\s*)(\[(.*)\])*(\s+)<=\s+(.*);", line)
        assign_seq_line1 = re.findall(rf"{signal_pattern}(\s*)(\[(.*)\])*(\s*)<=(\s*)(.*)(\s*)(\[(.*)\]);", line)
        assign_comb_line = re.findall(rf"{signal_pattern}(\[(.*)\])*(\s+)=(\s+)(.*);", line)

        if def_reg_line and not_endpoint:
            state["coi_dict"][idx] = f"  input {def_reg_line[0][1]} {signal_name};\n"
            state["in_set"].add(signal_name)
            return
        if def_reg_line and not not_endpoint:
            state["ep_dict"][0] = f"  output reg {def_reg_line[0][0]} {signal_name};\n"
            state["out_set"].add(signal_name)

        if not not_endpoint:
            if assign_seq_line1:
                rhs = assign_seq_line1[0][-4]
                state["ep_dict"][-1] = f"  always @(posedge clk) begin\n    {signal_name} <= {rhs};\n  end"
                state["un_add_set"].update(get_coi_signal(rhs))
            elif assign_seq_line2:
                rhs = assign_seq_line2[0][-1]
                state["ep_dict"][-1] = f"  always @(posedge clk) begin\n    {signal_name} <= {rhs};\n  end"
                state["un_add_set"].update(get_coi_signal(rhs))

        if def_line and not_endpoint:
            state["coi_dict"][idx] = line
            if "input" in line:
                state["in_set"].add(signal_name)
            elif "output" in line:
                state["out_set"].add(signal_name)
        elif assign_comb_line:
            state["coi_dict"][idx] = line
            state["un_add_set"].update(get_coi_signal(assign_comb_line[0][-1]))


def build_cone_text(endpoint: str, lines: list[str]) -> str | None:
    state = {
        "add_set": {endpoint, ":", "?", "{", "}", "~", "|"},
        "un_add_set": set(),
        "coi_dict": {},
        "ep_dict": {},
        "in_set": set(),
        "out_set": set(),
    }
    add_signal_to_cone(endpoint, lines, state, not_endpoint=False)
    while True:
        todo_set = state["un_add_set"] - state["add_set"]
        before_len = len(state["un_add_set"])
        for signal in todo_set:
            add_signal_to_cone(signal, lines, state, not_endpoint=True)
            state["add_set"].add(signal)
        if before_len == len(state["un_add_set"]):
            break

    if 0 not in state["ep_dict"] or -1 not in state["ep_dict"]:
        return None
    if not state["in_set"] and not state["out_set"] and not state["coi_dict"] and not state["ep_dict"]:
        return None

    ports = ["clk", "rst"] + sorted(state["in_set"]) + sorted(state["out_set"])
    cone_lines = [f"module coi ({', '.join(ports)});\n", "  input clk;\n", "  input rst;\n"]
    cone_lines.append(state["ep_dict"][0])
    for idx in sorted(state["coi_dict"]):
        cone_lines.append(state["coi_dict"][idx])
    cone_lines.append(state["ep_dict"][-1])
    cone_lines.append("\nendmodule\n")
    return "".join(cone_lines)


def stage_cones(designs: dict[str, str], rtl_dir: Path, reg_dir: Path, cone_root: Path, cmd: str) -> None:
    for design in designs:
        rtl_file = rtl_dir / f"{design}.v"
        reg_file = reg_dir / f"{design}.json"
        if not rtl_file.exists() or not reg_file.exists():
            print(f"[cones:{cmd}] skip {design}: missing RTL or reg list")
            continue
        lines = rtl_file.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        endpoints = json.loads(reg_file.read_text(encoding="utf-8"))
        out_dir = cone_root / cmd / design
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for endpoint in endpoints:
            cone_text = build_cone_text(endpoint, lines)
            if cone_text is None:
                continue
            safe_endpoint = endpoint.replace("/", "_")
            (out_dir / f"{safe_endpoint}.v").write_text(cone_text, encoding="utf-8")
            count += 1
        print(f"[cones:{cmd}] {design}: {count} cones -> {out_dir}")


def load_dg5_module():
    sys.path.insert(0, str(CONE_GRAPH_SCR))
    spec = importlib.util.spec_from_file_location("trace_vlg_cone2graph_dg5", CONE_GRAPH_SCR / "vlg_cone2graph_dg5.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load vlg_cone2graph_dg5.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_dg5_fields(graph_file: Path, node_dict_file: Path, dg5_module) -> None:
    topology_data = dg5_module.extract_dg5_topology_from_graph(str(graph_file), str(node_dict_file))
    dg5_data = dg5_module.create_dg5_format_data(topology_data)
    with open(graph_file, "rb") as handle:
        graph_data = pickle.load(handle)
    graph_data["dg5_gate"] = dg5_data["gate"]
    graph_data["dg5_edge_index"] = dg5_data["edge_index"]
    graph_data["dg5_forward_level"] = dg5_data["forward_level"]
    graph_data["dg5_forward_index"] = dg5_data["forward_index"]
    graph_data["dg5_x"] = dg5_data["x"]
    graph_data["dg5_prob"] = dg5_data["prob"]
    graph_data["dg5_num_nodes"] = dg5_data["num_nodes"]
    graph_data["dg5_node_names"] = dg5_data["node_names"]
    with open(graph_file, "wb") as handle:
        pickle.dump(graph_data, handle)


def stage_cone_graphs(
    designs: dict[str, str],
    reg_dir: Path,
    cone_root: Path,
    cone_graph_root: Path,
    rtl_graph_root: Path,
    cmd: str,
) -> None:
    analyze_script = CONE_GRAPH_SCR / "analyze.py"
    dg5_module = load_dg5_module()
    env = os.environ.copy()
    env["TRACE_RTL_FUNC_DIR"] = str(rtl_graph_root)
    for design in designs:
        reg_file = reg_dir / f"{design}.json"
        if not reg_file.exists():
            continue
        endpoints = json.loads(reg_file.read_text(encoding="utf-8"))
        out_dir = cone_graph_root / cmd / design
        out_dir.mkdir(parents=True, exist_ok=True)
        for endpoint in endpoints:
            safe_endpoint = endpoint.replace("/", "_")
            cone_file = cone_root / cmd / design / f"{safe_endpoint}.v"
            if not cone_file.exists():
                continue
            run_command(
                [sys.executable, str(analyze_script), str(cone_file), "-D", design, "-N", safe_endpoint, "-C", cmd, "-O", str(out_dir) + "/"],
                cwd=CONE_GRAPH_SCR,
                env=env,
            )
            graph_file = out_dir / f"{safe_endpoint}_{cmd}.pkl"
            node_dict_file = out_dir / f"{safe_endpoint}_{cmd}_node_dict.pkl"
            if graph_file.exists() and node_dict_file.exists():
                add_dg5_fields(graph_file, node_dict_file, dg5_module)


def graph_to_dgl(graph_file: Path, node_dict_file: Path):
    import dgl
    import networkx as nx
    import numpy as np
    import torch
    from dgl import from_networkx

    sys.path.insert(0, str(DATASET_GRAPH_SCR))
    from graph2dgl import edge_feat_extra_word, node_feat_extra_word

    with open(graph_file, "rb") as handle:
        graph = pickle.load(handle)
    with open(node_dict_file, "rb") as handle:
        node_dict = pickle.load(handle)

    if isinstance(graph, (nx.Graph, nx.DiGraph)):
        g_nx = nx.DiGraph(graph)
    else:
        g_nx = nx.DiGraph()
        for src, dst_list in graph.items():
            if isinstance(src, str) and src.startswith("dg5_"):
                continue
            if isinstance(dst_list, (list, tuple, set)):
                for dst in dst_list:
                    if isinstance(dst, (str, int, float)):
                        g_nx.add_edge(str(src), str(dst))

    feat_matrix = []
    for node_name in g_nx.nodes():
        if node_name not in node_dict:
            feat_vec = np.array([0 for _ in range(27)])
        else:
            feat_vec = node_feat_extra_word(node_name, node_dict[node_name], g_nx, node_dict)
        feat_matrix.append(feat_vec)
    feat_matrix = torch.FloatTensor(np.array(feat_matrix)) if feat_matrix else torch.zeros((0, 27))

    edge_features = []
    for edge_pair in g_nx.edges():
        edge_features.append(edge_feat_extra_word(edge_pair, g_nx, node_dict))
    edge_features = torch.FloatTensor(np.array(edge_features)) if edge_features else torch.zeros((0, 12))

    dgl_graph = from_networkx(g_nx)
    dgl_graph.ndata["feat"] = feat_matrix
    dgl_graph.edata["feat"] = edge_features
    spd, path = dgl.shortest_dist(dgl_graph, root=None, return_paths=True)
    dgl_graph.ndata["spd"] = spd
    dgl_graph.ndata["path"] = path
    if isinstance(graph, dict):
        for key, value in graph.items():
            if key.startswith("dg5_"):
                setattr(dgl_graph, key, value)
    return dgl_graph


def read_split(split_dir: Path, tag: str) -> list[str]:
    split_file = split_dir / f"{tag}_lst.json"
    with open(split_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stage_dataset(reg_dir: Path, cone_graph_root: Path, dataset_out: Path, split_dir: Path, tags: Iterable[str]) -> None:
    sys.path.insert(0, str(TRACE_SRC))
    from dataset.rtl_parser import MyDataset

    dataset_out.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        design_list = read_split(split_dir, tag)
        for cmd in ("ori", "pos"):
            dataset = MyDataset()
            data_dict: dict[int, list[str]] = {}
            for design in design_list:
                reg_file = reg_dir / f"{design}.json"
                if not reg_file.exists():
                    continue
                endpoints = json.loads(reg_file.read_text(encoding="utf-8"))
                for endpoint in endpoints:
                    safe_endpoint = endpoint.replace("/", "_")
                    ori_graph = cone_graph_root / "ori" / design / f"{safe_endpoint}_ori.pkl"
                    pos_graph = cone_graph_root / "pos" / design / f"{safe_endpoint}_pos.pkl"
                    if not ori_graph.exists() or not pos_graph.exists():
                        continue
                    graph_file = cone_graph_root / cmd / design / f"{safe_endpoint}_{cmd}.pkl"
                    node_dict_file = cone_graph_root / cmd / design / f"{safe_endpoint}_{cmd}_node_dict.pkl"
                    if not graph_file.exists() or not node_dict_file.exists():
                        continue
                    dataset.add_graph_data(graph_to_dgl(graph_file, node_dict_file), [])
                    data_dict[len(data_dict)] = [design, safe_endpoint]
            with open(dataset_out / f"dataset_{tag}_{cmd}.pkl", "wb") as handle:
                pickle.dump(dataset, handle)
            (dataset_out / f"data_dict_{tag}_{cmd}.json").write_text(json.dumps(data_dict, indent=2), encoding="utf-8")
            print(f"[dataset] {tag}/{cmd}: {len(dataset)} graphs -> {dataset_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TRACE RTL contrastive datasets")
    parser.add_argument("--designs", default=str(DEFAULT_DESIGNS), help="JSON design manifest: {design: top_module}")
    parser.add_argument("--only-design", action="append", help="Restrict to one design; can be repeated")
    parser.add_argument("--stages", nargs="+", default=["registers", "transform", "whole_graph", "cones", "cone_graph", "dataset"],
                        choices=["registers", "transform", "whole_graph", "cones", "cone_graph", "dataset"],
                        help="Pipeline stages to run")
    parser.add_argument("--work-dir", default=str(ROOT / "work"), help="Working directory for generated intermediate files")
    parser.add_argument("--ori-dir", help="Directory containing original RTL {design}.v")
    parser.add_argument("--pos-dir", help="Directory for transformed positive RTL")
    parser.add_argument("--reg-dir", help="Directory for endpoint/register JSON files")
    parser.add_argument("--cone-root", help="Directory for extracted RTL cones")
    parser.add_argument("--rtl-graph-root", help="Directory for whole-design graph pickles")
    parser.add_argument("--cone-graph-root", help="Directory for cone graph pickles")
    parser.add_argument("--dataset-out", default=str(RTL_ROOT / "data" / "dataset_graph" / "data_bench"),
                        help="Output directory for dataset_{train,valid}_{ori,pos}.pkl")
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR), help="Directory with train_lst.json and valid_lst.json")
    parser.add_argument("--tags", nargs="+", default=["train", "valid"], help="Dataset splits to emit")
    parser.add_argument("--yosys-bin", default=os.environ.get("YOSYS_BIN", "yosys"), help="Yosys executable")
    parser.add_argument("--skip-pos-transform", action="store_true", help="Use existing --pos-dir files instead of running Yosys")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = resolve(args.work_dir)
    ori_dir = resolve(args.ori_dir) if args.ori_dir else work_dir / "ori"
    pos_dir = resolve(args.pos_dir) if args.pos_dir else work_dir / "pos"
    reg_dir = resolve(args.reg_dir) if args.reg_dir else work_dir / "reg_lst"
    cone_root = resolve(args.cone_root) if args.cone_root else work_dir / "cone"
    rtl_graph_root = resolve(args.rtl_graph_root) if args.rtl_graph_root else work_dir / "rtl_graph"
    cone_graph_root = resolve(args.cone_graph_root) if args.cone_graph_root else work_dir / "cone_graph"
    dataset_out = resolve(args.dataset_out)
    split_dir = resolve(args.split_dir)
    designs = load_designs(resolve(args.designs), args.only_design)

    if "registers" in args.stages:
        stage_registers(designs, ori_dir, reg_dir)
    if "transform" in args.stages and not args.skip_pos_transform:
        stage_transform(designs, ori_dir, pos_dir, args.yosys_bin, work_dir / "tmp")
    if "whole_graph" in args.stages:
        stage_whole_graph(designs, ori_dir, "ori", rtl_graph_root)
        stage_whole_graph(designs, pos_dir, "pos", rtl_graph_root)
    if "cones" in args.stages:
        stage_cones(designs, ori_dir, reg_dir, cone_root, "ori")
        stage_cones(designs, pos_dir, reg_dir, cone_root, "pos")
    if "cone_graph" in args.stages:
        stage_cone_graphs(designs, reg_dir, cone_root, cone_graph_root, rtl_graph_root, "ori")
        stage_cone_graphs(designs, reg_dir, cone_root, cone_graph_root, rtl_graph_root, "pos")
    if "dataset" in args.stages:
        stage_dataset(reg_dir, cone_graph_root, dataset_out, split_dir, args.tags)


if __name__ == "__main__":
    main()