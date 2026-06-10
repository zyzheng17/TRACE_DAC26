from __future__ import absolute_import
from __future__ import print_function
import sys
import os, time, pickle
from optparse import OptionParser
import networkx as nx

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyverilog
from pyverilog.vparser.parser import parse
from AST_analyzer import *


def main(design_name=None, cmd=None, out_path=None):
    start_time = time.perf_counter()
    INFO = "Verilog code parser"
    VERSION = pyverilog.__version__
    USAGE = "Usage: python example_parser.py file ..."

    def showVersion():
        print(INFO)
        print(VERSION)
        print(USAGE)
        sys.exit()

    optparser = OptionParser()
    optparser.add_option("-v", "--version", action="store_true", dest="showversion",
                         default=False, help="Show the version")
    optparser.add_option("-I", "--include", dest="include", action="append",
                         default=[], help="Include path")
    optparser.add_option("-D", dest="design", action="append",
                         default=[], help="Entire design name")
    optparser.add_option("-N", dest="Name", action="append",
                         default=[], help="Design Name")
    optparser.add_option("-C", dest="cmd", action="append",
                         default=[], help="Design command")
    optparser.add_option("-O", dest="out_path", action="append",
                         default=[], help="Output path")
    (options, args) = optparser.parse_args()

    if options.design:
        vlg_design = options.design[0]
    if options.Name:
        design_name = options.Name[0]
    if options.cmd:
        cmd = options.cmd[0]
    if options.out_path:
        out_path = options.out_path[0]

    filelist = args
    if options.showversion:
        showVersion()

    for f in filelist:
        if not os.path.exists(f):
            raise IOError("file not found: " + f)

    if len(filelist) == 0:
        showVersion()


    ast, directives = parse(filelist,
                            preprocess_include=options.include)
    
    print('Verilog2AST Finish!')
    func_root = os.environ.get('TRACE_RTL_FUNC_DIR')
    if func_root:
        func_dict_path = os.path.join(func_root, cmd, f"{vlg_design}_{cmd}_func_dict.pkl")
    else:
        func_dict_path = f"../../dataset/rtl_graph/{cmd}/{vlg_design}_{cmd}_func_dict.pkl"
    with open(func_dict_path, 'rb') as f:
        func_dict = pickle.load(f)
    ast_analysis = AST_analyzer(ast, func_dict)
    ast_analysis.AST2Graph(ast)

    g = ast_analysis.graph

    g_nx = nx.DiGraph(g.graph)
    print(g_nx)
    # g.show_graph()

    g.graph2pkl(design_name, cmd, out_path)


if __name__ == '__main__':
    main()