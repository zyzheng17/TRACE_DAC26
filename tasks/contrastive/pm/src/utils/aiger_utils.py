import os
import aiger



def seqaig_to_xdata(aig_filename, tmp_aag_filename='', gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}):
    assert 'AND' in gate_to_index.keys()
    assert 'PI' in gate_to_index.keys()
    assert 'NOT' in gate_to_index.keys()
    assert 'DFF' in gate_to_index.keys()
    if tmp_aag_filename == '':
        tmp_aag_filename = aig_filename + '.aag'
    aig2aag_cmd = 'aigtoaig {} {}'.format(aig_filename, tmp_aag_filename)
    info = os.popen(aig2aag_cmd).readlines()
    f = open(tmp_aag_filename, 'r')
    lines = f.readlines()
    f.close()
    os.remove(tmp_aag_filename)

    header = lines[0].strip().split(" ")
    if len(header) == 7:
        # “M”, “I”, “L”, “O”, “A” separated by spaces.
        n_variables = int(header[1])
        n_inputs = int(header[2])
        n_latch = int(header[3])
        unknown = int(header[4])
        n_and = int(header[5])
        n_outputs = int(header[6])
    elif len(header) == 6:
        n_variables = int(header[1])
        n_inputs = int(header[2])
        n_outputs = int(header[4])
        n_and = int(header[5])
        n_latch = int(header[3])
    else:
        return [], []

    # if n_outputs != 1 or n_variables != (n_inputs + n_and) or n_variables == n_inputs:
    #     return [], []
    # assert n_outputs == 1, 'The AIG has multiple outputs.'
    # assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
    # assert n_variables != n_inputs, '# variable equals to # inputs'
    # Construct AIG graph
    x_data = []
    edge_index = []
    # node_labels = []

    for i in range(n_inputs + n_and + n_latch):
        x_data.append([len(x_data), gate_to_index['PI']])

    # AND Connections
    has_not = [-1] * (len(x_data) + 1)
    for i, line in enumerate(lines[1 + n_inputs + n_outputs + n_latch :]):
        arr = line.replace('\n', '').split(' ')
        if len(arr) != 3:
            continue
        and_index = int(int(arr[0]) / 2) - 1
        x_data[and_index][1] = gate_to_index['AND']
        fanin_1_index = int(int(arr[1]) / 2) - 1
        fanin_2_index = int(int(arr[2]) / 2) - 1
        fanin_1_not = int(arr[1]) % 2
        fanin_2_not = int(arr[2]) % 2
        if fanin_1_not == 1:
            if has_not[fanin_1_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([fanin_1_index, not_index])
                has_not[fanin_1_index] = not_index
            fanin_1_index = has_not[fanin_1_index]
        if fanin_2_not == 1:
            if has_not[fanin_2_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([fanin_2_index, not_index])
                has_not[fanin_2_index] = not_index
            fanin_2_index = has_not[fanin_2_index]
        edge_index.append([fanin_1_index, and_index])
        edge_index.append([fanin_2_index, and_index])

    # DFF Connections
    for i, line in enumerate(lines[1 + n_inputs : 1 + n_inputs + n_latch]):
        arr = line.replace('\n', '').split(' ')
        assert len(arr) in [2, 3]
        latch_index = int(int(arr[0]) / 2) - 1
        x_data[latch_index][1] = gate_to_index['DFF']
        fanin_index = int(int(arr[1]) / 2) - 1
        fanin_not = int(arr[1]) % 2
        if fanin_not == 1:
            if has_not[fanin_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([fanin_index, not_index])
                has_not[fanin_index] = not_index
            fanin_index = has_not[fanin_index]
        edge_index.append([fanin_index, latch_index])

    # PO NOT check
    for i, line in enumerate(lines[1 + n_inputs + n_latch : 1 + n_inputs + n_latch + n_outputs]):
        arr = line.replace('\n', '').split(' ')
        if len(arr) != 1:
            continue
        po_index = int(int(arr[0]) / 2) - 1
        if po_index < 0:
            continue
        po_not = int(arr[0]) % 2
        if po_not == 1:
            if has_not[po_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([po_index, not_index])
                has_not[po_index] = not_index

    # Remove Constraints
    const_idx = -1
    for edge_idx, edge in enumerate(edge_index):
        if edge[0] == -1:
            if const_idx == -1:
                const_idx = len(x_data)
                x_data.append([len(x_data), gate_to_index['PI']])
                edge_index[edge_idx][0] = const_idx
            else:
                edge_index[edge_idx][0] = const_idx

    return x_data, edge_index


def myaig_to_xdata(aig_filename, gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2}):
    with open(aig_filename, 'rb') as file:
        first_line = file.readline()
        first_line = first_line.decode('ascii')
        header = first_line.strip().split(" ")
        n_variables = int(header[1])
        n_inputs = int(header[2])
        n_outputs = int(header[4])
        n_and = int(header[5])
        no_latch = int(header[3])
        assert no_latch == 0, 'The AIG has latches.'
        x_data = []
        edge_index = []
        # PI
        for i in range(n_inputs):
            x_data.append([len(x_data), gate_to_index['PI']])
        # AND
        for i in range(n_and):
            x_data.append([len(x_data), gate_to_index['AND']])
        has_not = [-1] * (n_inputs + n_and)
        for i in range(n_outputs):
            line = file.readline()
        for i in range(n_and):
            t = 0
            child1 = 0
            child2 = 0
            while True:
                block = file.read(1)
                unsigned_int = int.from_bytes(block, byteorder='little')
                child1 |= (unsigned_int & 0x7F) << (7 * t)
                if unsigned_int & 0x80 == 0:
                    break
                t += 1
            t = 0
            while True:
                block = file.read(1)
                unsigned_int = int.from_bytes(block, byteorder='little')
                child2 |= (unsigned_int & 0x7F) << (7 * t)
                if unsigned_int & 0x80 == 0:
                    break
                t += 1
            fanin_1_index = int(int(2 * (i + 1 + n_inputs) - child1) / 2) - 1
            fanin_2_index = int(int(2 * (i + 1 + n_inputs) - child1 - child2) / 2) - 1
            fanin_1_not = int(2 * (i + 1 + n_inputs) - child1) % 2
            fanin_2_not = int(2 * (i + 1 + n_inputs) - child1 - child2) % 2
            if fanin_1_not == 1:
                if has_not[fanin_1_index] == -1:
                    x_data.append([len(x_data), gate_to_index['NOT']])
                    not_index = len(x_data) - 1
                    edge_index.append([fanin_1_index, not_index])
                    has_not[fanin_1_index] = not_index
                fanin_1_index = has_not[fanin_1_index]
            if fanin_2_not == 1:
                if has_not[fanin_2_index] == -1:
                    x_data.append([len(x_data), gate_to_index['NOT']])
                    not_index = len(x_data) - 1
                    edge_index.append([fanin_2_index, not_index])
                    has_not[fanin_2_index] = not_index
                fanin_2_index = has_not[fanin_2_index]
            # x_data.append([len(x_data), gate_to_index['AND']])
            edge_index.append([fanin_1_index, i + n_inputs])
            edge_index.append([fanin_2_index, i + n_inputs])
    with open(aig_filename, 'rb') as file:
        first_line = file.readline()
        for i in range(n_outputs):
            line = file.readline()
            line = line.decode('ascii')
            arr = line.replace('\n', '').split(' ')
            if len(arr) != 1:
                continue
            po_index = int(int(arr[0]) / 2) - 1
            if po_index < 0:
                continue
            po_not = int(arr[0]) % 2
            if po_not == 1:
                if has_not[po_index] == -1:
                    x_data.append([len(x_data), gate_to_index['NOT']])
                    not_index = len(x_data) - 1
                    edge_index.append([po_index, not_index])
                    has_not[po_index] = not_index
    return x_data, edge_index, n_inputs + n_and


def read_aig(aig_filename):
    with open(aig_filename, 'rb') as file:
        first_line = file.readline()
        first_line = first_line.decode('ascii')
        header = first_line.strip().split(" ")
        n_variables = int(header[1])
        n_inputs = int(header[2])
        n_outputs = int(header[4])
        n_and = int(header[5])
        no_latch = int(header[3])
        assert no_latch == 0, 'The AIG has latches.'
        inputs = ''
        outputs = ''
        ands = ''
        for i in range(n_inputs):
            inputs += f'{2*(i+1)}\n'
        for i in range(n_outputs):
            line = file.readline()
            line = line.decode('ascii')
            outputs += f'{line}'
        for i in range(n_and):
            t = 0
            child1 = 0
            child2 = 0
            while True:
                block = file.read(1)
                unsigned_int = int.from_bytes(block, byteorder='little')
                child1 |= (unsigned_int & 0x7F) << (7 * t)
                if unsigned_int & 0x80 == 0:
                    break
                t += 1
            t = 0
            while True:
                block = file.read(1)
                unsigned_int = int.from_bytes(block, byteorder='little')
                child2 |= (unsigned_int & 0x7F) << (7 * t)
                if unsigned_int & 0x80 == 0:
                    break
                t += 1
            fanin_1_index = int(int(2 * (i + 1 + n_inputs) - child1))
            fanin_2_index = int(int(2 * (i + 1 + n_inputs) - child1 - child2))
            ands += f'{2*(i + 1 + n_inputs)} {fanin_1_index} {fanin_2_index}\n'
        res = file.read()
        res = res.decode('ascii')
        f = first_line + inputs + outputs + ands + res
    return f

def aig_to_xdata(aig_filename, gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2}):
    aig = aiger.load(aig_filename)
    f = str(aiger.BoolExpr(aig))
    lines = f.split('\n')
    header = lines[0].strip().split(" ")
    # “M”, “I”, “L”, “O”, “A” separated by spaces.
    n_variables = eval(header[1])
    n_inputs = eval(header[2])
    n_outputs = eval(header[4])
    n_and = eval(header[5])
    no_latch = eval(header[3])
    assert no_latch == 0, 'The AIG has latches.'
    # if n_outputs != 1 or n_variables != (n_inputs + n_and) or n_variables == n_inputs:
    #     return [], []
    # assert n_outputs == 1, 'The AIG has multiple outputs.'
    # assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
    # assert n_variables != n_inputs, '# variable equals to # inputs'
    # Construct AIG graph
    x_data = []
    edge_index = []
    # node_labels = []
    
    # PI 
    for i in range(n_inputs):
        x_data.append([len(x_data), gate_to_index['PI']])
    # AND 
    for i in range(n_and):
        x_data.append([len(x_data), gate_to_index['AND']])
    
    # AND Connections
    has_not = [-1] * (len(x_data) + 1)
    for (i, line) in enumerate(lines[1+n_inputs+n_outputs: ]):
        arr = line.replace('\n', '').split(' ')
        if len(arr) != 3:
            continue
        and_index = int(int(arr[0]) / 2) - 1
        fanin_1_index = int(int(arr[1]) / 2) - 1
        fanin_2_index = int(int(arr[2]) / 2) - 1
        fanin_1_not = int(arr[1]) % 2
        fanin_2_not = int(arr[2]) % 2
        if fanin_1_not == 1:
            if has_not[fanin_1_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([fanin_1_index, not_index])
                has_not[fanin_1_index] = not_index
            fanin_1_index = has_not[fanin_1_index]
        if fanin_2_not == 1:
            if has_not[fanin_2_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([fanin_2_index, not_index])
                has_not[fanin_2_index] = not_index
            fanin_2_index = has_not[fanin_2_index]
        edge_index.append([fanin_1_index, and_index])
        edge_index.append([fanin_2_index, and_index])

    # PO NOT check 
    for (i, line) in enumerate(lines[1+n_inputs: 1+n_inputs+n_outputs]):
        arr = line.replace('\n', '').split(' ')
        if len(arr) != 1:
            continue
        po_index = int(int(arr[0]) / 2) - 1
        if po_index < 0:
            continue
        po_not = int(arr[0]) % 2
        if po_not == 1:
            if has_not[po_index] == -1:
                x_data.append([len(x_data), gate_to_index['NOT']])
                not_index = len(x_data) - 1
                edge_index.append([po_index, not_index])
                has_not[po_index] = not_index

    return x_data, edge_index

# def aig_to_xdata(aig_filename, gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2}):
#     #aig = aiger.load(aig_filename)
#     #f = str(aiger.BoolExpr(aig))
#     f = read_aig(aig_filename)
#     with open('t.aag','w') as file:
#         file.write(f)
#     lines = f.split('\n')
#     header = lines[0].strip().split(" ")
#     # “M”, “I”, “L”, “O”, “A” separated by spaces.
#     n_variables = eval(header[1])
#     n_inputs = eval(header[2])
#     n_outputs = eval(header[4])
#     n_and = eval(header[5])
#     no_latch = eval(header[3])
#     assert no_latch == 0, 'The AIG has latches.'
#     # if n_outputs != 1 or n_variables != (n_inputs + n_and) or n_variables == n_inputs:
#     #     return [], []
#     # assert n_outputs == 1, 'The AIG has multiple outputs.'
#     # assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
#     # assert n_variables != n_inputs, '# variable equals to # inputs'
#     # Construct AIG graph
#     x_data = []
#     edge_index = []
#     # node_labels = []
    
#     # PI 
#     # x_data.append([len(x_data), gate_to_index['PI']])
#     for i in range(n_inputs):
#         x_data.append([len(x_data), gate_to_index['PI']])
#     # AND 
#     for i in range(n_and):
#         x_data.append([len(x_data), gate_to_index['AND']])
    
#     # AND Connections
#     has_not = [-1] * (len(x_data) + 1)
#     for (i, line) in enumerate(lines[1+n_inputs+n_outputs: ]):
#         arr = line.replace('\n', '').split(' ')
#         if len(arr) != 3:
#             continue
#         and_index = int(int(arr[0]) / 2) - 1
#         fanin_1_index = int(int(arr[1]) / 2) - 1
#         fanin_2_index = int(int(arr[2]) / 2) - 1
#         fanin_1_not = int(arr[1]) % 2
#         fanin_2_not = int(arr[2]) % 2
#         if fanin_1_not == 1:
#             if has_not[fanin_1_index] == -1:
#                 x_data.append([len(x_data), gate_to_index['NOT']])
#                 not_index = len(x_data) - 1
#                 edge_index.append([fanin_1_index, not_index])
#                 has_not[fanin_1_index] = not_index
#             fanin_1_index = has_not[fanin_1_index]
#         if fanin_2_not == 1:
#             if has_not[fanin_2_index] == -1:
#                 x_data.append([len(x_data), gate_to_index['NOT']])
#                 not_index = len(x_data) - 1
#                 edge_index.append([fanin_2_index, not_index])
#                 has_not[fanin_2_index] = not_index
#             fanin_2_index = has_not[fanin_2_index]
#         edge_index.append([fanin_1_index, and_index])
#         edge_index.append([fanin_2_index, and_index])

#     # PO NOT check 
#     for (i, line) in enumerate(lines[1+n_inputs: 1+n_inputs+n_outputs]):
#         arr = line.replace('\n', '').split(' ')
#         if len(arr) != 1:
#             continue
#         po_index = int(int(arr[0]) / 2) - 1
#         if po_index < 0:
#             continue
#         po_not = int(arr[0]) % 2
#         if po_not == 1:
#             if has_not[po_index] == -1:
#                 x_data.append([len(x_data), gate_to_index['NOT']])
#                 not_index = len(x_data) - 1
#                 edge_index.append([po_index, not_index])
#                 has_not[po_index] = not_index

#     return x_data, edge_index,n_inputs+n_and
