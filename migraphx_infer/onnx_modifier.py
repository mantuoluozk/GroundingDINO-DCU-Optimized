      
"""
onnx modifier: provide a conviennt way to modify onnx model
1. add node
2. remove node
3. modify node
4. query node
"""

from collections import defaultdict, deque
import os
import os.path as osp
import shutil
import tempfile
from typing import List, Dict, Set, Tuple, Optional, Union
import uuid
import warnings

import numpy as np
import onnx
from onnx import AttributeProto, numpy_helper
from onnx import shape_inference
from onnx.helper import make_attribute, make_node, make_opsetid, make_tensor, \
                        tensor_dtype_to_np_dtype
from onnxconverter_common import float16
import tqdm
# from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE


SUPPORT_DTYPES = [
    'BOOL', 'STRING', 'BFLOAT16', 'DOUBLE', 'FLOAT', 'FLOAT16', 
    'INT16', 'INT32', 'INT4', 'INT64', 'INT8', 'UINT16', 'UINT32', 'UINT4', 'UINT64', 'UINT8',
]
SUPPORT_DTYPES.extend([dt.lower() for dt in SUPPORT_DTYPES])


class Node:
    def __init__(self, onnx_modifier=None, obj=None, index=None):
        self.onnx_modifier = onnx_modifier
        self.obj = obj
        self.index = index
    
    @property
    def name(self):
        return self.obj.name
    
    @property
    def op_type(self):
        return self.obj.op_type

    @property
    def inputs(self):
        return self.obj.input

    @property
    def outputs(self):
        return self.obj.output

    @property
    def input_names(self):
        return self.inputs

    @property
    def output_names(self):
        return self.outputs

    def check_modifier(self):
        if self.onnx_modifier is None:
            raise RuntimeError("onnx_modifier is not initialized")
    @property
    def prev_nodes(self):
        self.check_modifier()
        return self.onnx_modifier.get_prev_nodes(self)
    
    @property
    def next_nodes(self):
        self.check_modifier()
        return self.onnx_modifier.get_next_nodes(self)

    def replace_input(self, old_name, new_name):
        assert old_name in self.obj.input, \
            f'"{old_name}" not in input name list of node named "{self.name}"'
        for i, in_name in enumerate(self.obj.input):
            if in_name == old_name:
                self.set_input(i, new_name)

    def set_input(self, index, name):
        # assert index < len(self.obj.input), "index out of range"
        # orig_name = self.obj.input[index]
        # self.obj.input[index] = name
        assert index < len(self.onnx_modifier.graph.node[self.index].input), "index out of range"
        orig_name = self.onnx_modifier.graph.node[self.index].input[index]
        self.onnx_modifier.graph.node[self.index].input[index] = name

        self.check_modifier()

        # Can not execute connection.pop_to_node() method directly. 
        # When node inputs contain multiple orig_name, need to remain the node in to_nodes.
        if list(self.onnx_modifier.graph.node[self.index].input).count(orig_name) == 0:
            self.onnx_modifier.connection_map[orig_name].pop_to_node(self)

        if name not in self.onnx_modifier.connection_map:
            self.onnx_modifier.connection_map[name] = Connection(name, self.onnx_modifier)
        self.onnx_modifier.connection_map[name].add_to_node(self)

    def set_inputs(self, names):
        assert len(names) == len(self.obj.input), "number of inputs does not match"
        assert all(isinstance(name, str) for name in names), "input names must be strings"
        self.obj.input[:] = names

    def set_output(self, index, name):
        assert index < len(self.obj.output), "index out of range"
        orig_name = self.obj.output[index]
        self.obj.output[index] = name

        self.check_modifier()
        self.onnx_modifier.connection_map[orig_name].clear_from_node()
        if name not in self.onnx_modifier.connection_map:
            self.onnx_modifier.connection_map[name] = Connection(name, self.onnx_modifier)
        self.onnx_modifier.connection_map[name].set_from_node(self)

    def set_outputs(self, names):
        assert len(names) == len(self.obj.output), "number of outputs does not match"
        assert all(isinstance(name, str) for name in names), "output names must be strings"
        self.obj.output[:] = names

    @property
    def attrs(self):
        attrs = {}
        for attr in self.obj.attribute:
            if attr.type == AttributeProto.FLOAT:  # 1
                value = attr.f
            elif attr.type == AttributeProto.INT:  # 2
                value = attr.i
            elif attr.type == AttributeProto.STRING:  # 3
                value = attr.s.decode('utf-8')
            elif attr.type == AttributeProto.TENSOR:  # 4
                value = numpy_helper.to_array(attr.t)
            elif attr.type == AttributeProto.FLOATS:  # 6
                value = list(attr.floats)
            elif attr.type == AttributeProto.INTS:  # 7
                value = list(attr.ints)
            else:
                value = f"Unsupported type: {attr.type}"
            attrs[attr.name] = value
        return attrs

    def set_attribute(self, name, value, name2attr=None):
        if not name2attr:
            name2attr = {}
            for attr in self.obj.attribute:
                name2attr[attr.name] = attr

        if name in name2attr:
            if isinstance(value, float):
                name2attr[name].f = value
                name2attr[name].type = AttributeProto.FLOAT
            elif isinstance(value, int):
                name2attr[name].i = value
                name2attr[name].type = AttributeProto.INT
            elif isinstance(value, str):
                name2attr[name].s = value.encode('utf-8')
                name2attr[name].type = AttributeProto.STRING
            elif isinstance(value, np.ndarray):
                name2attr[name].ClearField("t")
                name2attr[name].t.CopyFrom(numpy_helper.from_array(value))
            elif isinstance(value, list):
                is_all_float = all(isinstance(x, float) for x in value)
                is_all_int = all(isinstance(x, int) for x in value)
                assert is_all_float or is_all_int
                if is_all_float:
                    name2attr[name].ClearField("floats")
                    name2attr[name].floats.extend(value)
                    name2attr[name].type = AttributeProto.FLOATS
                else:
                    name2attr[name].ClearField("ints")
                    name2attr[name].ints.extend(value)
                    name2attr[name].type = AttributeProto.INTS
        else:
            if isinstance(value, np.ndarray):
                value = numpy_helper.from_array(value)
            self.obj.attribute.append(make_attribute(name, value))

    def set_attributes(self, attr_dict):
        name2attr = {}
        for attr in self.obj.attribute:
            name2attr[attr.name] = attr
        
        for name, value in attr_dict.items():
            self.set_attribute(name, value, name2attr)
        

class Connection:
    def __init__(self, conn_name, onnx_modifier=None):
        self.name = conn_name
        self.onnx_modifier = onnx_modifier
        self.from_node = None
        self.to_nodes = []
        self.to_node_names = set()

    def check_modifier(self):
        if self.onnx_modifier is None:
            raise RuntimeError("onnx_modifier is not initialized")

    def set_from_node(self, node: str | Node):

        if isinstance(node, str):
            self.check_modifier()
            _node = self.onnx_modifier.get_node(Node)
            assert node is not None, f'No node named "{node}" in onnx graph!'
        elif isinstance(node, Node):
            _node = node
        else:
            raise TypeError(f"Connection.set_from_node except input argument type" \
                            f" is str or Node, but received: {type(node)}")
        
        self.from_node = _node

    def clear_from_node(self):
        self.from_node = None

    def add_to_node(self, node: str | Node):
        if isinstance(node, str):
            _name = node
            self.check_modifier()
            _node = self.onnx_modifier.get_node(Node)
            assert node is not None, f'No node named "{node}" in onnx graph!'
        elif isinstance(node, Node):
            _name = node.name
            _node = node
        else:
            raise TypeError(f"Connection.add_to_node except input argument type" \
                            f" is str or Node, but received: {type(node)}")

        if _name not in self.to_node_names:
            self.to_node_names.add(_name)
            self.to_nodes.append(_node)
    
    def pop_to_node(self, node: str | Node):
        if isinstance(node, str):
            _name = node
            self.check_modifier()
            _node = self.onnx_modifier.get_node(Node)
            assert node is not None, f'No node named "{node}" in onnx graph!'
        elif isinstance(node, Node):
            _name = node.name
            _node = node
        else:
            raise TypeError(f"Connection.pop_to_node except input argument type" \
                            f" is str or Node, but received: {type(node)}")
    
        if _name not in self.to_node_names:
            raise ValueError(f'Node "{_name}" not in target nodes of connction "{self.name}"!')
        
        self.to_node_names.remove(_name)
        for i in range(len(self.to_nodes)):
            if self.to_nodes[i].name == _name:
                return self.to_nodes.pop(i)
        else:
            raise RuntimeError("to_nodes dismatch to_node_names!")


class ONNXModifier:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        
        self.node_map = {}
        self.initializer_map = {}
        self.sparse_initializer_map = {}
        self.connection_map = {}
        self.value_info_map = {}

        self.parse_onnx(self.onnx_path)

    def parse_onnx(self, onnx_path):
        model = onnx.load(onnx_path)

        self.model = model
        self.domain = model.domain
        self.graph = model.graph
        self.ir_version = model.ir_version
        self.mdoel_version = model.model_version
        self.opset_import = model.opset_import

        self.update_map()
    
    def add_node_name_if_nameless(self, node: Node):
        if not hasattr(self, "node_suffixes"):
            self.name_suffixes = set()
        if node.name == "" or node.name == None:
            suffix = None
            while True:
                suffix = uuid.uuid4().hex[:8]
                if suffix not in self.name_suffixes:
                    break
            node.obj.name = node.op_type + "_" + suffix

    def add_opset_import(self, domain: str, version: int):
        self.model.opset_import.append(make_opsetid(domain, version))

    @property
    def input_names(self):
        return [i.name for i in self.graph.input]

    @property
    def output_names(self):
        return [o.name for o in self.graph.output]

    def add_input(self, name, dtype='float32', shape=None):
        assert dtype in set(SUPPORT_DTYPES)
        self.create_value_info(name, dtype=dtype, shape=shape)

        new_input = self.value_info_map.pop(name)
        _new_input = self.graph.value_info.pop()
        assert id(new_input) == id(_new_input)
        assert name == new_input.name

        self.graph.input.append(new_input)
        return new_input

    def add_output(self, name, new_name=None, shape=None):
        if name not in self.value_info_map:
            raise ValueError(f"{name} not in onnx_modifier.value_info_map")
        
        index = None
        for i, v in enumerate(self.graph.value_info):
            if v.name == name:
                index = i
                break
        else:
            raise ValueError(f"{name} not in model.graph.value_info")
        
        value_info = self.value_info_map.pop(name)
        assert value_info.name == name
        assert id(value_info) == id(self.graph.value_info[index])
        self.graph.value_info.pop(index)

        if shape is not None:
            tensor_type = onnx.helper.make_tensor_type_proto(
                elem_type=value_info.type.tensor_type.elem_type,
                shape=shape
            )
            value_info.type.CopyFrom(tensor_type)

        if new_name is None:
            self.graph.output.append(value_info)
        else:
            from_node = self.get_from_node(name)
            to_nodes = self.get_to_nodes(name)
            for i, output_name in enumerate(from_node.output_names):
                if output_name == name:
                    from_node.set_output(i, new_name)
            for node in to_nodes:
                node.replace_input(name, new_name)
            value_info.name = new_name
            self.graph.output.append(value_info)

    def remove_output(self, name):
        """根据名称删除模型输出"""
        assert name in self.output_names
        # print("need remove output name:", name)

        index = None
        for i, out in enumerate(self.graph.output):
            # print(f"current(index={i}) output name:", out.name)
            if out.name == name:
                index = i
                break
        else:
            raise RuntimeError(f"ONNX graphx not has a output named '{name}'.")
        
        self.graph.output.pop(index)

    def get_node(self, name_or_index: Union[str, int]):
        """根据节点名称或索引获取节点实例"""

        if isinstance(name_or_index, str):
            if name_or_index in self.node_map:
                return self.node_map.get(name_or_index, None)
        elif isinstance(name_or_index, int):
            if name_or_index < len(self.graph.node):
                return self.node_map.get(
                    self.graph.node[name_or_index].name, None)
            else:
                raise ValueError(f"Node index {name_or_index} out of range")

    def get_nodes(self, *op_types: str):
        """根据节点类型获取节点实例"""

        assert len(op_types) >= 1
        op_types_set = set(op_types)
        node_names = [node.name for node in self.graph.node if node.op_type in op_types_set]
        nodes = [self.node_map[name] for name in node_names]
        return nodes

    def get_initializer(self, name: str):
        """根据initializer名称获取initializer"""
        return self.initializer_map.get(name)

    def get_connection(self, name: str):
        """根据边名称获取边"""
        return self.connection_map.get(name)

    def get_from_node(self, conn: Union[str, Connection]):
        """获取某条边的输入节点名"""

        if isinstance(conn, str):
            assert conn in self.connection_map, f"Connection {conn} not in connection_map!"
            return self.connection_map[conn].from_node
        elif isinstance(conn, Connection):
            return conn.from_node
        else:
            raise TypeError(f"Invalid connection type {type(conn)}")

    def get_to_nodes(self, conn: Union[str, Connection]):
        """获取某条边的输出节点"""

        if isinstance(conn, str):
            assert conn in self.connection_map, f"Connection {conn} not in connection_map!"
            return self.connection_map[conn].to_nodes
        elif isinstance(conn, Connection):
            return conn.to_nodes
        else:
            raise TypeError(f"Invalid connection type {type(conn)}")

    def get_prev_nodes(self, node: Union[str, Node]):
        """获取某节点的上游输入节点"""
        
        if isinstance(node, str):
            node = self.node_map[node]
        elif isinstance(node, Node):
            pass
        else:
            raise TypeError(f"Invalid node type {type(node)}")
        
        nodes = []
        for conn_name in node.inputs:
            from_node = self.get_from_node(conn_name)
            if from_node:
                nodes.append(from_node)
        return nodes

    def get_next_nodes(self, node: Union[str, Node]):
        """获取某节点的下游节点"""

        if isinstance(node, str):
            node = self.node_map[node]
        elif isinstance(node, Node):
            pass
        else:
            raise TypeError(f"Invalid node type {type(node)}")
        
        nodes = []
        for conn_name in node.outputs:
            to_nodes = self.get_to_nodes(conn_name)
            nodes.extend(to_nodes)
        return nodes

    def create_node(self, op_type, op_name, inputs, outputs, doc_string=None, 
                    domain=None, index=None, **attrs):
        """创建一个新节点"""
        onnx_node = make_node(op_type, inputs, outputs, name=op_name, 
                              doc_string=doc_string, domain=domain, **attrs)

        if index is None:
            self.graph.node.append(onnx_node)
            index = len(self.graph.node) - 1
        else:
            assert index <= len(self.graph.node), "index out of range"
            self.graph.node.insert(index, onnx_node)
            for i in range(index + 1, len(self.graph.node)):
                node_name = self.graph.node[i].name
                old_idx = self.node_map[node_name].index
                assert old_idx == i - 1, \
                    f"Node {node_name} index conflict: {old_idx} != {i - 1}"
                self.node_map[node_name].index = i
        
        new_node = Node(self, self.graph.node[index], index)
        self.node_map[op_name] = new_node

        for in_name in new_node.input_names:
            if in_name not in self.value_info_map:
                self.create_value_info(in_name, dtype="float")
            if in_name not in self.connection_map:
                self.connection_map[in_name] = Connection(in_name, self)
            self.connection_map[in_name].add_to_node(new_node)
        for out_name in new_node.output_names:
            if out_name not in self.value_info_map:
                self.create_value_info(out_name, dtype="float")
            if out_name not in self.connection_map:
                self.connection_map[out_name] = Connection(out_name, self)
            self.connection_map[out_name].set_from_node(new_node)

        return new_node

    def create_initializer(self, name, value: np.ndarray):
        """创建一个 initializer"""
        assert name not in self.initializer_map, f"initializer {name} already exists!"
        init_node = numpy_helper.from_array(value, name=name)

        use_external_data = value.nbytes / 1024 / 1024 / 1024 > 2
        if use_external_data:
            print("use external data:", name)
            init_node.data_location = onnx.TensorProto.EXTERNAL
            location = name.replace('/', '+') + '.data'
            onnx.external_data_helper.set_external_data(init_node, location)
            with tempfile.TemporaryDirectory() as tmp_dir:
                onnx.external_data_helper.save_external_data(init_node, tmp_dir)
                init_node.ClearField("raw_data")
                self.graph.initializer.append(init_node)
                onnx.external_data_helper.load_external_data_for_tensor(
                    self.graph.initializer[-1], tmp_dir)
                del self.graph.initializer[-1].external_data[:]
                self.graph.initializer[-1].ClearField("data_location")
        else:
            self.graph.initializer.append(init_node)

        self.initializer_map[name] = self.graph.initializer[-1]
        return self.graph.initializer[-1]

    def create_value_info(self, name, dtype=None, shape=None):
        if dtype is None:
            elem_type = None
        else:
            assert isinstance(dtype, str)
            assert dtype in set(SUPPORT_DTYPES)
            elem_type = getattr(onnx.TensorProto, dtype.upper())

        value_info = onnx.helper.make_tensor_value_info(name=name, 
                                                        elem_type=elem_type, 
                                                        shape=shape)
        self.graph.value_info.append(value_info)
        self.value_info_map[name] = self.graph.value_info[-1]
        return self.graph.value_info[-1]

    def get_initializer_value(self, name):
        """获取initializer的数值"""

        init = self.get_initializer(name)
        return numpy_helper.to_array(init)

    def set_initializer_value(self, name, value: np.ndarray):
        """为initializer设置新的数值"""
        
        init = self.get_initializer(name)

        # 检查形状和类型
        old_shape = list(init.dims)
        new_shape = list(value.shape)
        # old_dtype = TENSOR_TYPE_TO_NP_TYPE.get(init.data_type, None)
        old_dtype = tensor_dtype_to_np_dtype(init.data_type)
        new_dtype = value.dtype
        if old_shape != new_shape:
            warn_message = f"Initailizer {name} shape changed: {old_shape} -> {new_shape}"
            warnings.warn(warn_message, RuntimeWarning)
        if old_dtype is not None and old_dtype != new_dtype:
            warn_message = f"Initailizer {name} dtype changed: {old_dtype} -> {new_dtype}"
            warnings.warn(warn_message, RuntimeWarning)
        
        new_tensor_proto = numpy_helper.from_array(value, name=name)
        init.CopyFrom(new_tensor_proto)

    def connect_node(self, node, inputs_map, outputs_map):
        """将某个节点与其上下游节点连接起来
        
        Args:
            node: Node
            inputs_map: [(node0, out_idx0), (node1, out_idx1), ...]
            outputs_map: [(node0, in_idx0), (node1, in_idx1), ...]
        """
        
        # 在连接 A -> B 时，若 A 的输出名与 B 的输入名冲突时，优先使用 A 的输出名，
        # 即：B.input[i] = A.output[j]
        for i, (n, j) in enumerate(inputs_map):
            if isinstance(n, str):
                n = self.node_map[n]
            assert j < len(n.outputs), \
                f"output index {i} out of node {n.name} outputs range"
            node.set_input(i, n.outputs[j])

        for name, (n, i) in zip(node.outputs, outputs_map):
            if isinstance(n, str):
                n = self.node_map[n]
            assert i < len(n.outputs), \
                f"output index {i} out of node {n.name} outputs range"
            n.set_output(i, name)
        
        # TODO: update self.connection_map
        
    def pop_node(self, node: Union[str, Node, int], auto_connect=True):
        """根据节点名称或索引移除节点"""

        if isinstance(node, str):
            node = self.node_map.get(node, None)
            if node is None:
                return None
            index = node.index
            assert node.name == self.graph.node[index].name
        elif isinstance(node, int):
            if node >= len(self.graph.node):
                raise ValueError(f"Node index {node} out of range")
            index = node
        elif isinstance(node, Node):
            index = node.index
        else:
            raise ValueError(f"Invalid node name or index: {node}")
        
        for i in range(index + 1, len(self.graph.node)):
            node = self.graph.node[i]
            self.node_map[node.name].index -= 1

        # print(f"node_name={self.graph.node[index].name} node_index={index}")
        _node_obj = self.graph.node[index]
        _node = self.get_node(_node_obj.name)
        next_nodes = self.get_next_nodes(_node)
        self.graph.node.pop(index)
        self.node_map.pop(_node_obj.name)

        # automatic connecting edges
        if auto_connect and len(_node.inputs) == 1 and len(_node.outputs) == 1:
            # self.connection_map[_node.inputs[0]].pop_to_node(_node)
            for next_node in next_nodes:
                next_node.replace_input( _node.outputs[0], _node.inputs[0])
                # self.connection_map[_node.inputs[0]].add_to_node(next_node)
            # self.connection_map.pop(_node.outputs[0])
        
        # update connection_map
        for in_name in _node.input_names:
            if _node.name in self.connection_map[in_name].to_node_names:
                self.connection_map[in_name].pop_to_node(_node)

        for i, out_name in enumerate(_node.output_names):
            self.connection_map[out_name].clear_from_node()

        return _node

    def remove_nodes(self, nodes: List[str | Node], auto_connect=False):
        """同时删除多个节点"""

        indices = set()
        _nodes = []
        invalid_nodes = set()
        for node in nodes:
            if isinstance(node, str):
                if node in self.node_map:
                    node = self.node_map[node]
                    if node.index not in indices:
                        _nodes.append(node)
                        indices.add(node.index)
                else:
                    invalid_nodes.add(node)
            elif isinstance(node, Node):
                if node.index not in indices:
                    _nodes.append(node)
                    indices.add(node.index)
            else:
                invalid_nodes.add(node)
        
        _nodes.sort(key=lambda x:x.index, reverse=True)

        use_progress_bar = len(_nodes) > 500
        if use_progress_bar:
            pbar = tqdm.tqdm(total=len(_nodes), desc="Removing nodes")
        for node in _nodes:
            self.pop_node(node, auto_connect=auto_connect)
            if use_progress_bar:
                pbar.update(1)
        if use_progress_bar:
            pbar.close()
                
        # print(f"{len(nodes) - len(invalid_nodes)} nodes have been removed.")
        # if len(invalid_nodes) > 0:
        #     print(f"find {len(invalid_nodes)} invalid nodes:\n", invalid_nodes)

    def pop_initializer(self, init_name: str, update_node_inputs: bool = True):
        """根据initializer名字移除initializer"""

        _init1 = self.initializer_map.pop(init_name)
        init_index = None
        for i in range(len(self.graph.initializer)):
            if self.graph.initializer.name == init_name:
                init_index = i
                break
        else:
            raise ValueError(f"Not existing a Initializer named {init_name}")
        
        _init2 = self.graph.initializer.pop(init_index)
        assert id(_init1) == id(_init2)

        # if update_node_inputs and init_name in self.connection_map:
        #     to_nodes = self.get_to_nodes(init_name)
        #     self.connection_map.pop(init_name)
        #     for node in to_nodes:
        #         num_inputs = len(node.inputs)
        #         for i in range(num_inputs-1, -1, -1):
        #             if node.inputs[i] == init_name:
        #                 node.inputs.pop(i)

        return _init1

    def update_map(self):
        """更新connection_map与node_map"""
        self.node_map.clear()
        self.connection_map.clear()
        self.initializer_map.clear()
        self.sparse_initializer_map.clear()
        self.value_info_map.clear()

        for i, node in enumerate(self.graph.node):
            new_node = Node(self, node, i)
            self.add_node_name_if_nameless(new_node)
            self.node_map[node.name] = new_node

            for conn_name in node.input:
                if conn_name not in self.connection_map:
                    self.connection_map[conn_name] = Connection(conn_name, self)
                self.connection_map[conn_name].add_to_node(new_node)
            for conn_name in node.output:
                if conn_name not in self.connection_map:
                    self.connection_map[conn_name] = Connection(conn_name, self)
                self.connection_map[conn_name].set_from_node(new_node)

        for i, node in enumerate(self.graph.initializer):
            self.initializer_map[node.name] = node
        
        for i, node in enumerate(self.graph.sparse_initializer):
            self.sparse_initializer_map[node.name] = [node, i]
        
        for i, conn in enumerate(self.graph.value_info):
            self.value_info_map[conn.name] = conn

    def find_unuseful_nodes(self):
        """寻找没有用到的节点"""

        end_names = set()
        for output_name in self.output_names:
            end_names.add(self.get_from_node(output_name).name)

        unuseful_names = set()
        for node in self.node_map.values():
            if node.name in end_names:
                continue
            next_nodes = self.get_next_nodes(node)
            if len(next_nodes) == 0:
                unuseful_names.add(node.name)
        
        model_output_names = set(self.output_names)
        q = deque([self.node_map[name] for name in unuseful_names])
        while len(q) != 0:
            node = q.popleft()
            prev_nodes = self.get_prev_nodes(node)
            for node1 in prev_nodes:
                next_nodes = self.get_next_nodes(node1)
                next_names = set([node2.name for node2 in next_nodes])
                # if (next_names - end_names).issubset(unuseful_names):
                if next_names.issubset(unuseful_names):
                    if node1.name not in unuseful_names and set(node1.output_names).isdisjoint(model_output_names):
                        q.append(node1)
                        unuseful_names.add(node1.name)
        
        unuseful_nodes = [self.node_map[name] for name in unuseful_names]
        return unuseful_nodes


    def remove_trash(self):
        """
        1. 移除无用的节点
        2. 移除无用的initializer
        3. 移除没有输入节点的connection
        4. 移除没有用到的模型输入与输出
        5. 移除没有用到的value_info
        """

        self.update_map()
        unuseful_nodes = self.find_unuseful_nodes()
        print(f"Find unuseful {len(unuseful_nodes)} nodes!")
        for i, node in enumerate(unuseful_nodes):
            print(f"remove unuseful node {i}:", node.name)
        self.remove_nodes(unuseful_nodes)
        self.update_map()

        all_node_inputs = set()
        for node in self.node_map.values():
            all_node_inputs.update(node.input_names)

        # remove unuseful initializers
        cnt = 0
        for init_name in list(self.initializer_map.keys()):
            if init_name in all_node_inputs:
                continue
            index = None
            for i, init in enumerate(self.graph.initializer):
                if init.name == init_name:
                    index = i
                    break
            else:
                raise ValueError(
                    f"{init_name} not in model.graph.initializer")
            print(f"remove unuseful initializer {cnt}:", init_name)
            self.graph.initializer.pop(index)
            cnt += 1

        # remove unuseful sparse_initializers
        cnt = 0
        for init_name in list(self.sparse_initializer_map.keys()):
            if init_name in all_node_inputs:
                continue
            index = None
            for i, init in enumerate(self.graph.sparse_initializer):
                if init.name == init_name:
                    index = i
                    break
            else:
                raise ValueError(
                    f"{init_name} not in model.graph.sparse_initializer")
            print(f"remove unuseful sparse initializer {cnt}:", init_name)
            self.graph.sparse_initializer.pop(index)
            cnt += 1
        
        self.update_map()
        
        # remove unuseful inputs and outputs
        for in_name in self.input_names:
            # print(in_name, [n.name for n in self.get_to_nodes(in_name)])
            if len(self.get_to_nodes(in_name)) != 0:
                continue
            for i, _in in enumerate(self.graph.input):
                if in_name == _in.name:
                    self.graph.input.pop(i)
                    break
        for out_name in self.output_names:
            # print(out_name, self.get_from_node(out_name).name)
            if self.get_from_node(out_name) is not None:
                continue
            for i, _out in enumerate(self.graph.output):
                if out_name == _out.name:
                    self.graph.output.pop(i)
                    break

        self.update_map()

        # remove unuseful value_info
        cnt = 0
        num_value_info = len(self.graph.value_info)
        for i in range(num_value_info-1, -1, -1):
            v = self.graph.value_info[i]
            if v.name not in self.connection_map:
                self.graph.value_info.pop(i)
                print(f"remove unuseful value_info {cnt}:", v.name)
                cnt += 1

        self.update_map()
    
    def infer_shape(self, strict_mode=False):

        for vi in self.graph.value_info:
            if vi.type.HasField("tensor_type"):
                vi.type.tensor_type.ClearField("shape")

        model = shape_inference.infer_shapes(self.model, strict_mode=strict_mode)
        self.model = model
        self.domain = model.domain
        self.graph = model.graph
        self.ir_version = model.ir_version
        self.mdoel_version = model.model_version
        self.opset_import = model.opset_import

        self.update_map()

    def infer_node_shpe(self, node):
        input_shapes = []
        input_dtypes = []
        for input_name in node.inputs:
            value_info = self.value_info_map[input_name]
            input_shapes.append(value_info.type.tensor_type.dims)
            input_dtypes.append(value_info.type.tensor_type.type)

        shape_inference.infer_node_outputs(node.obj, input_shapes, input_dtypes)
    
    def convert_float_to_float16(self):
        self.model = float16.convert_float_to_float16(self.model, keep_io_types=True)

    def save(self, save_path, save_as_external_data=False, 
             all_tensors_to_one_file=True):
        
        self.remove_trash()
        
        external_data_name = osp.basename(save_path) + '.data'
        external_data_path = osp.join(osp.dirname(save_path), external_data_name)
        if save_as_external_data and osp.isfile(external_data_path):
            os.remove(external_data_path)

        onnx.save(self.model, 
                  save_path, 
                  save_as_external_data=save_as_external_data, 
                  all_tensors_to_one_file=all_tensors_to_one_file, 
                  location=external_data_name, 
                  size_threshold=1024, 
                  convert_attribute=False)
