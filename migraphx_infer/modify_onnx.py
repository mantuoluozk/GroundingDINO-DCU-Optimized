import sys
import numpy as np
from onnx_modifier import ONNXModifier


BATCH_SIZE = None         # 1
TEXT_SEQ_LEN = None       # 4
BERT_NUM_HEADS = None     # 12

ENCODER_NUM_HEADS = None  # 4
ENCODER_HEAD_DIM = None   # 64
ENCODER_ATTN_DIM = None   # 256

DECODER_NUM_HEADS = None  # 8
DECODER_HEAD_DIM = None   # 32
DECODER_ATTN_DIM = None   # 256

ENCODER_VISION_SEQ_LEN = None  # 19947
DECODER_VISION_SEQ_LEN = None  #900


def initialize_global_variables(om: ONNXModifier):
    def _get_dim_from_value_info(value_info_name: str, axis: int) -> int:
        try:
            value_info = None
            if value_info_name in om.value_info_map:
                value_info = om.value_info_map[value_info_name]
            else:
                for v in om.graph.input:
                    if v.name == value_info_name:
                        value_info = v
                        break
            if value_info is None:
                raise RuntimeError(f"Value info {value_info_name} not found in onnx graph!")
            return value_info.type.tensor_type.shape.dim[axis].dim_value
        except Exception as e:
            raise RuntimeError(f"Failed to get dimension from value info: {value_info_name}")

    global BATCH_SIZE
    BATCH_SIZE = _get_dim_from_value_info("img", 0)
    global TEXT_SEQ_LEN
    TEXT_SEQ_LEN = _get_dim_from_value_info("input_ids", 1)
    global BERT_NUM_HEADS
    BERT_NUM_HEADS = _get_dim_from_value_info("/bert/encoder/layer.0/attention/self/Softmax_output_0", 1)

    global ENCODER_NUM_HEADS, ENCODER_HEAD_DIM
    batchsize_x_numheads = _get_dim_from_value_info("/transformer/encoder/text_layers.0/self_attn/Reshape_output_0", -2)
    assert batchsize_x_numheads % BATCH_SIZE == 0, \
        f"Encoder batch_heads({batchsize_x_numheads}) is not divisible by batch_size({BATCH_SIZE})"
    ENCODER_NUM_HEADS = batchsize_x_numheads // BATCH_SIZE
    ENCODER_HEAD_DIM = _get_dim_from_value_info("/transformer/encoder/text_layers.0/self_attn/Reshape_output_0", -1)
    global ENCODER_ATTN_DIM
    ENCODER_ATTN_DIM = _get_dim_from_value_info("/transformer/encoder/text_layers.0/self_attn/Gemm_output_0", -1)
    assert ENCODER_NUM_HEADS * ENCODER_HEAD_DIM == ENCODER_ATTN_DIM

    global DECODER_NUM_HEADS, DECODER_HEAD_DIM
    batchsize_x_numheads = _get_dim_from_value_info("/transformer/decoder/layers.0/self_attn/Reshape_output_0", -2)
    assert batchsize_x_numheads % BATCH_SIZE == 0, \
        f"Decoder batch_heads({batchsize_x_numheads}) is not divisible by batch_size({BATCH_SIZE})"
    DECODER_NUM_HEADS = batchsize_x_numheads // BATCH_SIZE
    DECODER_HEAD_DIM = _get_dim_from_value_info("/transformer/decoder/layers.0/self_attn/Reshape_output_0", -1)
    global DECODER_ATTN_DIM
    DECODER_ATTN_DIM = _get_dim_from_value_info("/transformer/decoder/layers.0/self_attn/Gemm_output_0", -1)
    assert DECODER_NUM_HEADS * DECODER_HEAD_DIM == DECODER_ATTN_DIM

    global ENCODER_VISION_SEQ_LEN
    ENCODER_VISION_SEQ_LEN = _get_dim_from_value_info("/transformer/Concat_8_output_0", 1)
    global DECODER_VISION_SEQ_LEN
    DECODER_VISION_SEQ_LEN = _get_dim_from_value_info("/transformer/TopK_output_1", -1)


def change_inf_to_value(om: ONNXModifier):
    records = set()
    for where_node in om.get_nodes("Where"):
        for input_name in where_node.inputs[1:]:
            init = om.get_initializer(input_name)
            if init is None:
                continue
            assert input_name == init.name
            init_name = input_name
            if init_name in records:
                continue

            # info = np.finfo(np.float32)
            info = np.finfo(np.float16)
            data = om.get_initializer_value(init.name)
            if data.size > 1:
                continue
            if data == np.inf:
                om.set_initializer_value(init_name, np.array(info.max, dtype=np.float32))
            elif data == -np.inf:
                om.set_initializer_value(init_name, np.array(info.min, dtype=np.float32))
            else:
                continue
            # print("Changed value:", init_name)
            records.add(init_name)


def optimize_where_ndoes(om: ONNXModifier):
    """Where节点等价替换
    (1) condition为initializer, X为0, Y为输入数据:
        Where(cond, X, Y) ==> Mul(Y, ~cond)
    (2) condition为initializer, X为负无穷, Y为输入数据
        Where(cond, X, Y) ==>  Sub(Y, Where(cond, np.inf, 0))
    (3) condition为真实输入, X为负无穷, Y为输入数据
        Where(cond, X, Y) ==> Sub(Y, Mul(Cast(cond, to=float32), np.inf))

    cases:
        1. Where(cond, -inf, input)
            a. /transformer/encoder/fusion_layers.*/attn/Where
            b. /transformer/encoder/fusion_layers.*/attn/Where_1
            c. /class_embed.0_*/Where: Where(cond, -inf, input)
        2. Where(cond, 0, input):
            a. /transformer/encoder/layers.*/self_attn/Where
            b. /transformer/decoder/layers.*/cross_attn/Where
    """

    for where_node in om.get_nodes("Where"):
        where_name = where_node.name
        # print("Process where node:", where_name)
        x_value = om.get_initializer_value(where_node.inputs[1])
        assert x_value.size == 1
        assert x_value == np.array(0.0, dtype=np.float32) or \
            x_value == np.array(-np.inf, dtype=np.float32)

        cond_init = om.get_initializer(where_node.inputs[0])
        if cond_init is not None:
            cond_value = om.get_initializer_value(where_node.inputs[0])
            if x_value == np.array(0.0, dtype=np.float32):
                # Where(cond, X, Y) ==> Mul(Y, ~cond)
                mul_name = where_name.replace("Where", "NewMul")
                mul_b_init = om.create_initializer(mul_name + "_B",
                                                   (~cond_value).astype(np.float32))
                mul_node = om.create_node("Mul",
                                          mul_name,
                                          [where_node.inputs[2], mul_b_init.name],
                                          [mul_name+"_output_0"],
                                          index=where_node.index)
                next_nodes = where_node.next_nodes
                for next_node in next_nodes:
                    next_node.replace_input(where_node.outputs[0], mul_node.outputs[0])
            elif x_value == np.array(-np.inf, dtype=np.float32):
                # Where(cond, X, Y) ==>  Sub(Y, Where(cond, np.inf, 0))
                sub_name = where_name.replace("Where", "NewSub")
                sub_b_init = om.create_initializer(
                    sub_name + "_B",
                    np.where(cond_value.astype(np.float32),
                             np.finfo(np.float16).max, 0.0).astype(np.float32)
                )
                sub_node = om.create_node("Sub",
                                          sub_name,
                                          [where_node.inputs[2], sub_b_init.name],
                                          [sub_name+"_output_0"],
                                          index=where_node.index)
                next_nodes = where_node.next_nodes
                for next_node in next_nodes:
                    next_node.replace_input(where_node.outputs[0], sub_node.outputs[0])
        else:
            # Where(cond, X, Y) ==> Sub(Y, Mul(Cast(cond, to=float32), np.inf))
            assert x_value == np.array(-np.inf, dtype=np.float32)
            cast_name = where_name.replace("Where", "NewCast")
            mul_name = where_name.replace("Where", "NewMul")
            sub_name = where_name.replace("Where", "NewSub")
            cast_node = om.create_node("Cast",
                                       cast_name,
                                       [where_node.inputs[0]],
                                       [cast_name+"_output_0"],
                                       to=1,
                                       index=where_node.index)
            mul_b_init = om.create_initializer(mul_name + "_B",
                                               np.array([np.finfo(np.float16).max], np.float32))
            mul_node = om.create_node("Mul",
                                      mul_name,
                                      [cast_node.outputs[0], mul_b_init.name],
                                      [mul_name+"_output_0"],
                                      index=cast_node.index+1)
            sub_node = om.create_node("Sub",
                                      sub_name,
                                      [where_node.inputs[2], mul_node.outputs[0]],
                                      [sub_name+"_output_0"],
                                      index=mul_node.index+1)
            next_nodes = where_node.next_nodes
            for next_node in next_nodes:
                next_node.replace_input(where_node.outputs[0], sub_node.outputs[0])

    om.update_map()


def optimize_transpose_nodes(om: ONNXModifier):
    from onnx import TensorProto

    def _register_shape_initializer(init):
        """Reshape shape inputs must be int64 in value_info; create_node defaults to float."""
        rank = len(om.get_initializer_value(init.name))
        if init.name not in om.value_info_map:
            om.create_value_info(init.name, dtype="int64", shape=[rank])
        else:
            om.value_info_map[init.name].type.tensor_type.elem_type = TensorProto.INT64

    transpose_list = [
        "/transformer/encoder/Transpose",
        "/transformer/encoder/Transpose_1",
        "/transformer/encoder/Transpose_2",
        "/transformer/encoder/Transpose_3",
        "/transformer/encoder/Transpose_4",
        "/transformer/encoder/Transpose_5",
        "/transformer/encoder/Transpose_6",
        "/transformer/encoder/Transpose_7",
        "/transformer/encoder/Transpose_8",
        "/transformer/encoder/Transpose_9",
        "/transformer/encoder/Transpose_10",
        "/transformer/encoder/Transpose_11",
        "/transformer/decoder/layers.0/Transpose",
        "/transformer/decoder/layers.0/Transpose_1",
        "/transformer/decoder/layers.0/Transpose_2",
        "/transformer/decoder/layers.1/Transpose",
        "/transformer/decoder/layers.1/Transpose_1",
        "/transformer/decoder/layers.1/Transpose_2",
        "/transformer/decoder/layers.2/Transpose",
        "/transformer/decoder/layers.2/Transpose_1",
        "/transformer/decoder/layers.2/Transpose_2",
        "/transformer/decoder/layers.3/Transpose",
        "/transformer/decoder/layers.3/Transpose_1",
        "/transformer/decoder/layers.3/Transpose_2",
        "/transformer/decoder/layers.4/Transpose",
        "/transformer/decoder/layers.4/Transpose_1",
        "/transformer/decoder/layers.4/Transpose_2",
        "/transformer/decoder/layers.5/Transpose",
        "/transformer/decoder/layers.5/Transpose_1",
        "/transformer/decoder/layers.5/Transpose_2",
        "/transformer/Transpose_8",
        "/transformer/decoder/Transpose",
        "/transformer/decoder/Transpose_1",
        "/transformer/decoder/Transpose_2",
        "/transformer/decoder/Transpose_3",
        "/transformer/decoder/Transpose_4",
        "/transformer/decoder/Transpose_5",
        "/transformer/decoder/Transpose_6",
        "/transformer/decoder/Transpose_7",
        "/transformer/decoder/Transpose_8",
        "/transformer/decoder/Transpose_9",
        "/transformer/decoder/Transpose_10",
        "/transformer/decoder/Transpose_11"
    ]
    for name in transpose_list:
        node = om.get_node(name)
        assert node.attrs['perm'] == [1, 0 , 2] or node.attrs['perm'] == [1, 0 , 2, 3], \
            f"perm={node.attrs['perm']}"
        next_nodes = om.get_next_nodes(node)
        for node_ in next_nodes:
            node_.replace_input(node.outputs[0], node.inputs[0])

    global BATCH_SIZE, ENCODER_NUM_HEADS, TEXT_SEQ_LEN
    shape_init1 = om.create_initializer(
        "/transformer/encoder/text_layers.x/self_attn/mask_des_shape",
        np.array([BATCH_SIZE, ENCODER_NUM_HEADS, TEXT_SEQ_LEN, TEXT_SEQ_LEN], np.int64)
    )
    _register_shape_initializer(shape_init1)
    mask_add_node_0 = om.get_node("/transformer/encoder/text_layers.0/self_attn/Add_3")
    mask_reshape_name = "/transformer/encoder/text_layers.x/self_attn/mask/Reshape"
    mask_reshape_node = om.create_node("Reshape",
                                       mask_reshape_name,
                                       [mask_add_node_0.inputs[1], shape_init1.name],
                                       [mask_reshape_name+"_output_0"],
                                       index=mask_add_node_0.index+1)
    shape_init2 = om.create_initializer(
        "/transformer/encoder/text_layers.x/self_attn/q_k_v_des_shape",
        np.array([BATCH_SIZE, TEXT_SEQ_LEN, ENCODER_NUM_HEADS, ENCODER_HEAD_DIM], np.int64)
    )
    _register_shape_initializer(shape_init2)
    shape_init3 = om.create_initializer(
        "/transformer/encoder/text_layers.x/self_attn/des_shape",
        np.array([BATCH_SIZE, TEXT_SEQ_LEN, ENCODER_ATTN_DIM], np.int64)
    )
    _register_shape_initializer(shape_init3)
    for i in range(6):
        q_reshape_node = om.get_node(f"/transformer/encoder/text_layers.{i}/self_attn/Reshape")
        q_reshape_node.set_input(1, shape_init2.name)
        q_trans_node = om.get_node(f"/transformer/encoder/text_layers.{i}/self_attn/Transpose")
        q_trans_node.set_attribute("perm", [0, 2, 1, 3])
        k_reshape_node = om.get_node(f"/transformer/encoder/text_layers.{i}/self_attn/Reshape_1")
        k_reshape_node.set_input(1, shape_init2.name)
        k_trans_node = om.get_node(f"/transformer/encoder/text_layers.{i}/self_attn/Transpose_2")
        k_trans_node.set_attribute("perm", [0, 2, 3, 1])
        v_reshape_node = om.get_node(f"/transformer/encoder/text_layers.{i}/self_attn/Reshape_2")
        v_reshape_node.set_input(1, shape_init2.name)
        v_trans_node = om.get_node(f"/transformer/encoder/text_layers.{i}/self_attn/Transpose_1")
        v_trans_node.set_attribute("perm", [0, 2, 1, 3])
        mask_add_node = om.get_node(f"/transformer/encoder/text_layers.{i}/self_attn/Add_3")
        mask_add_node.set_input(1, mask_reshape_node.outputs[0])
        mha_trans_node = om.get_node(f"/transformer/encoder/text_layers.{i}/self_attn/Transpose_3")
        mha_trans_node.set_attribute("perm", [0, 2, 1, 3])
        mlp_reshape_node = om.get_node(f"/transformer/encoder/text_layers.{i}/self_attn/Reshape_4")
        mlp_reshape_node.set_input(1, shape_init3.name)

    # modify /transformer/enc_out_class_embed/Transpose
    om.get_node("/transformer/enc_out_class_embed/Transpose").set_attribute("perm", [0, 2, 1])

    # modify /transformer/decoder/Reshape_*
    dst_shape_name = om.get_node("/transformer/decoder/Reshape").inputs[1]
    om.set_initializer_value(dst_shape_name, np.array([BATCH_SIZE, DECODER_VISION_SEQ_LEN, -1], np.int64))

    shape_init4 = om.create_initializer(
        "/transformer/decoder/vision/q_k_v_des_shape",
        np.array([BATCH_SIZE, DECODER_VISION_SEQ_LEN, DECODER_NUM_HEADS, DECODER_HEAD_DIM], np.int64)
    )
    _register_shape_initializer(shape_init4)
    shape_init5 = om.create_initializer(
        "/transformer/decoder/layers.x/self_attn_ca_text/des_shape",
        np.array([BATCH_SIZE, DECODER_VISION_SEQ_LEN, DECODER_ATTN_DIM], np.int64)
    )
    _register_shape_initializer(shape_init5)
    for i in range(6):
        q_reshape_node = om.get_node(f"/transformer/decoder/layers.{i}/self_attn/Reshape")
        q_reshape_node.set_input(1, shape_init4.name)
        q_trans_node = om.get_node(f"/transformer/decoder/layers.{i}/self_attn/Transpose")
        q_trans_node.set_attribute("perm", [0, 2, 1, 3])
        k_reshape_node = om.get_node(f"/transformer/decoder/layers.{i}/self_attn/Reshape_1")
        k_reshape_node.set_input(1, shape_init4.name)
        k_trans_node = om.get_node(f"/transformer/decoder/layers.{i}/self_attn/Transpose_2")
        k_trans_node.set_attribute("perm", [0, 2, 3, 1])
        if i == 0:
            mm_node = om.get_node("/transformer/decoder/layers.0/self_attn/MatMul_4")
            mm_b_value = om.get_initializer_value(mm_node.inputs[1])
            new_mm_b_value = np.ascontiguousarray(
                mm_b_value.reshape(BATCH_SIZE, DECODER_NUM_HEADS, DECODER_VISION_SEQ_LEN, DECODER_HEAD_DIM)
            )
            om.set_initializer_value(mm_node.inputs[1], new_mm_b_value)
        else:
            v_reshape_node = om.get_node(f"/transformer/decoder/layers.{i}/self_attn/Reshape_2")
            v_reshape_node.set_input(1, shape_init4.name)
            v_trans_node = om.get_node(f"/transformer/decoder/layers.{i}/self_attn/Transpose_1")
            v_trans_node.set_attribute("perm", [0, 2, 1, 3])
        mha_trans_node = om.get_node(f"/transformer/decoder/layers.{i}/self_attn/Transpose_3")
        mha_trans_node.set_attribute("perm", [0, 2, 1, 3])
        mlp_reshape_node = om.get_node(f"/transformer/decoder/layers.{i}/self_attn/Reshape_4")
        mlp_reshape_node.set_input(1, shape_init5.name)

    shape_init6 = om.create_initializer(
        "/transformer/decoder/layers.0/self_attn/Reshape",
        np.array([BATCH_SIZE, TEXT_SEQ_LEN, DECODER_NUM_HEADS, DECODER_HEAD_DIM], np.int64)
    )
    _register_shape_initializer(shape_init6)
    for i in range(6):
        q_reshape_node = om.get_node(f"/transformer/decoder/layers.{i}/ca_text/Reshape")
        q_reshape_node.set_input(1, shape_init4.name)
        q_trans_node = om.get_node(f"/transformer/decoder/layers.{i}/ca_text/Transpose")
        q_trans_node.set_attribute("perm", [0, 2, 1, 3])
        k_reshape_node = om.get_node(f"/transformer/decoder/layers.{i}/ca_text/Reshape_1")
        k_reshape_node.set_input(1, shape_init6.name)
        k_trans_node = om.get_node(f"/transformer/decoder/layers.{i}/ca_text/Transpose_3")
        k_trans_node.set_attribute("perm", [0, 2, 3, 1])
        v_reshape_node = om.get_node(f"/transformer/decoder/layers.{i}/ca_text/Reshape_2")
        v_reshape_node.set_input(1, shape_init6.name)
        v_trans_node = om.get_node(f"/transformer/decoder/layers.{i}/ca_text/Transpose_2")
        v_trans_node.set_attribute("perm", [0, 2, 1, 3])
        mask_add_node = om.get_node(f"/transformer/decoder/layers.{i}/ca_text/Add_3")
        mask_add_node.set_input(1, "/transformer/decoder/layers.0/ca_text/Reshape_3_output_0")
        mha_trans_node = om.get_node(f"/transformer/decoder/layers.{i}/ca_text/Transpose_4")
        mha_trans_node.set_attribute("perm", [0, 2, 1, 3])
        mlp_reshape_node = om.get_node(f"/transformer/decoder/layers.{i}/ca_text/Reshape_6")
        mlp_reshape_node.set_input(1, shape_init5.name)

    # modify /transformer/decoder/layers.0/Add
    # modify /transformer/decoder/layers.0/Add_1
    init_name = "/transformer/Tile_1_output_0"
    add_value = om.get_initializer_value(init_name)
    om.set_initializer_value(init_name, np.ascontiguousarray(add_value.transpose(1, 0, 2)))

    mul_node = om.get_node("/transformer/decoder/Mul")
    mul_b_value = om.get_initializer_value(mul_node.inputs[1])
    assert mul_b_value.ndim == 4 and mul_b_value.shape[0] == 1 and  mul_b_value.shape[1] == BATCH_SIZE
    om.set_initializer_value(mul_node.inputs[1], mul_b_value.transpose(1, 0, 2, 3))

    om.update_map()
    om.infer_shape()


def optmize_sin_cos_block(om: ONNXModifier):
    node_pairs = [
        ("/transformer/decoder/Gather_1",  "/transformer/decoder/ref_point_head/layers.0/MatMul"),
        ("/transformer/decoder/Gather_6",  "/transformer/decoder/ref_point_head/layers.0_1/MatMul"),
        ("/transformer/decoder/Gather_11", "/transformer/decoder/ref_point_head/layers.0_2/MatMul"),
        ("/transformer/decoder/Gather_16", "/transformer/decoder/ref_point_head/layers.0_3/MatMul"),
        ("/transformer/decoder/Gather_21", "/transformer/decoder/ref_point_head/layers.0_4/MatMul"),
        ("/transformer/decoder/Gather_26", "/transformer/decoder/ref_point_head/layers.0_5/MatMul"),
    ]

    unsqueeze_axes_init1 = om.create_initializer(
        "/transformer/decoder/sin_cos_block/unsqueeze_axes1",
        np.array([3, 4], np.int64)
    )
    slice_axes_init = om.create_initializer(
        "/transformer/decoder/sin_cos_block/slice_axes",
        np.array([4], np.int64)
    )
    slice_steps_init = om.create_initializer(
        "/transformer/decoder/sin_cos_block/slice_steps",
        np.array([1], np.int64)
    )
    slice_starts_init1 = om.create_initializer(
        "/transformer/decoder/sin_cos_block/slice_starts1",
        np.array([0], np.int64)
    )
    slice_ends_init1 = om.create_initializer(
        "/transformer/decoder/sin_cos_block/slice_ends1",
        np.array([1], np.int64)
    )
    slice_starts_init2 = om.create_initializer(
        "/transformer/decoder/sin_cos_block/slice_steps2",
        np.array([1], np.int64)
    )
    slice_ends_init2 = om.create_initializer(
        "/transformer/decoder/sin_cos_block/slice_ends2",
        np.array([2], np.int64)
    )
    reshape_init = om.create_initializer(
        "/transformer/decoder/sin_cos_block/reshape_dst_shape",
        np.array([BATCH_SIZE, DECODER_VISION_SEQ_LEN, -1], np.int64)
    )

    for i, (gather_name, matmul_name) in enumerate(node_pairs):
        gather_node = om.get_node(gather_name)
        next_node = om.get_next_nodes(gather_node)[0]
        assert next_node.op_type == "Mul", f"{next_node.op_type} {next_node.name}"
        mul_init_value = om.get_initializer_value(next_node.inputs[1])
        assert mul_init_value.size == 1
        next_node = om.get_next_nodes(next_node)[0]
        assert next_node.op_type == "Unsqueeze"
        next_node.set_inputs([gather_node.inputs[0], unsqueeze_axes_init1.name])
        next_node = om.get_next_nodes(next_node)[0]
        assert next_node.op_type == "Div"
        div_init_value = om.get_initializer_value(next_node.inputs[1])
        new_value = (div_init_value / mul_init_value).reshape(1, 1, 1, -1, 2)
        new_init = om.create_initializer(next_node.name + "_B", new_value)
        next_node.set_input(1, new_init.name)

        next_nodes = om.get_next_nodes(next_node)
        assert len(next_nodes) == 2 and all(x.op_type == 'Slice' for x in next_nodes)
        sin_node, cos_node = None, None
        for j, slice_node in enumerate(next_nodes):
            slice_node.set_inputs([slice_node.inputs[0],
                                   slice_starts_init1.name if j == 0 else slice_starts_init2.name,
                                   slice_ends_init1.name if j == 0 else slice_ends_init2.name,
                                   slice_axes_init.name,
                                   slice_steps_init.name])
            next_node = om.get_next_nodes(slice_node)[0]
            if next_node.op_type == "Sin":
                sin_node = next_node
            elif next_node.op_type == "Cos":
                cos_node = next_node
            else:
                raise RuntimeError("match fail!")
            next_node = om.get_next_nodes(next_node)[0]
            assert next_node.op_type == "Unsqueeze"
            next_node = om.get_next_nodes(next_node)[0]

        assert next_node.op_type == "Concat"
        next_node.set_inputs([sin_node.outputs[0], cos_node.outputs[0]])
        next_node.set_attribute("axis", 4)
        next_node = om.get_next_nodes(next_node)[0]
        assert next_node.op_type == "Reshape"
        next_node.set_input(1, reshape_init.name)

        matmul_node = om.get_node(matmul_name)
        matmul_node.set_input(0, next_node.outputs[0])
        value_info = om.value_info_map["/transformer/decoder/Div_output_0"]
        step = value_info.type.tensor_type.shape.dim[-1].dim_value
        if i == 0:
            mm_b_value = om.get_initializer_value(matmul_node.inputs[1])
            mm_b_value = np.concatenate([mm_b_value[step:step*2, ...],
                                         mm_b_value[0:step, ...],
                                         mm_b_value[step*2:, ...]],
                                        axis=0)
            om.set_initializer_value(matmul_node.inputs[1], mm_b_value)

    om.update_map()
    om.infer_shape()


def fuse_one_attention(om: ONNXModifier,
                       softmax_name: str,
                       padding_mask: str = None,
                       attn_mask: str = None,
                       pos_bias: str = None,
                       num_heads: int = 12):
    softmax_node = om.get_node(softmax_name)
    tmp_node = om.get_prev_nodes(softmax_node)[0]
    assert tmp_node.op_type in ["MatMul", "Add"]

    mask = None
    if tmp_node.op_type == "Add":
        mask_node = tmp_node
        tmp_node = om.get_from_node(mask_node.inputs[0])
        if tmp_node.op_type == "Div":
            tmp_node = om.get_from_node(tmp_node.inputs[0])
        assert tmp_node.op_type == "MatMul"
        mask = mask_node.inputs[1]
        if padding_mask is not None and attn_mask is not None:
            raise ValueError("padding_mask and attn_mask cannot be provided at the same time")
        if padding_mask is None and attn_mask is None:
            raise ValueError("padding_mask or attn_mask must be provided")

    tmp_node1 = om.get_from_node(tmp_node.inputs[0])
    if tmp_node1.op_type == "Mul":
        tmp_node1 = om.get_prev_nodes(tmp_node1)[0]
    tmp_node2 = om.get_from_node(tmp_node.inputs[1])
    if tmp_node2.op_type == "Mul":
        tmp_node2 = om.get_prev_nodes(tmp_node2)[0]
    assert tmp_node1.op_type == tmp_node2.op_type == "Transpose"
    tmp_node1 = om.get_prev_nodes(tmp_node1)[0]
    tmp_node2 = om.get_prev_nodes(tmp_node2)[0]
    assert tmp_node1.op_type == tmp_node2.op_type == "Reshape"

    global BATCH_SIZE

    if attn_mask is not None:
        q, k = tmp_node1.outputs[0], tmp_node2.outputs[0]
        q_dst_shape_value = om.get_initializer_value(tmp_node1.inputs[1])
        if q_dst_shape_value.size == 3:
            q_dst_shape_value_new = np.array([BATCH_SIZE, *q_dst_shape_value.tolist()], np.int64)
            om.set_initializer_value(tmp_node1.inputs[1], q_dst_shape_value_new)
        k_dst_shape_value = om.get_initializer_value(tmp_node2.inputs[1])
        if k_dst_shape_value.size == 3:
            k_dst_shape_value_new = np.array([BATCH_SIZE, *k_dst_shape_value.tolist()], np.int64)
            om.set_initializer_value(tmp_node2.inputs[1], k_dst_shape_value_new)
    else:
        q, k = tmp_node1.inputs[0], tmp_node2.inputs[0]

    tmp_node = om.get_next_nodes(softmax_node)[0]
    assert tmp_node.op_type == "MatMul"
    tmp_node3 = om.get_from_node(tmp_node.inputs[1])
    if tmp_node3 is not None:
        assert tmp_node3.op_type == "Transpose"
        tmp_node3 = om.get_prev_nodes(tmp_node3)[0]
        assert tmp_node3.op_type == "Reshape"
        if attn_mask is not None:
            v = tmp_node3.outputs[0]
            v_dst_shape_value = om.get_initializer_value(tmp_node3.inputs[1])
            if v_dst_shape_value.size == 3:
                v_dst_shape_value_new = np.array([BATCH_SIZE,
                                                 v_dst_shape_value[0].item(),
                                                 v_dst_shape_value[1].item(),
                                                 v_dst_shape_value[2].item()], np.int64)
                om.set_initializer_value(tmp_node3.inputs[1], v_dst_shape_value_new)
        else:
            v = tmp_node3.inputs[0]
    else:
        v_init = om.get_initializer(tmp_node.inputs[1])
        v_init_value = om.get_initializer_value(tmp_node.inputs[1])
        v_init_value = v_init_value.transpose(0, 2, 1, 3)
        B, S, H, D = v_init_value.shape
        v_init_value = np.ascontiguousarray(v_init_value.reshape(B, S, H*D))
        om.set_initializer_value(tmp_node.inputs[1], v_init_value)
        v = v_init.name

    tmp_node = om.get_next_nodes(tmp_node)[0]
    assert tmp_node.op_type == "Transpose"
    tmp_node = om.get_next_nodes(tmp_node)[0]
    assert tmp_node.op_type == "Reshape"

    if attn_mask is not None:
        mha_next_node = tmp_node
        mha_out_shape_value = om.get_initializer_value(tmp_node.inputs[1])
        if mha_out_shape_value.size == 2:
            mha_out_shape_value_new = np.array([BATCH_SIZE, -1, mha_out_shape_value[-1].item()], np.int64)
            om.set_initializer_value(tmp_node.inputs[1], mha_out_shape_value_new)
    else:
        mha_next_node = om.get_next_nodes(tmp_node)[0]
        assert mha_next_node.op_type in ["Gemm", "MatMul"]

    gemm_node = None
    if om.get_next_nodes(tmp_node)[0].op_type == "Gemm":
        gemm_node = om.get_next_nodes(tmp_node)[0]
        gemm_next_node = om.get_next_nodes(gemm_node)[0]
        assert gemm_next_node.op_type == "Reshape"
        reshape_next_node = om.get_next_nodes(gemm_next_node)[0]
        assert reshape_next_node.op_type == "Add"

    name_prefix = '/'.join(softmax_name.split('/')[:-1])
    node_inputs = [q, k, v]
    if mask is None:
        mha_type = "MultiHeadAttention"
    else:
        if padding_mask is not None:
            mha_type = "MultiHeadAttention"
            node_inputs.append(padding_mask)
        elif any([attn_mask is not None, pos_bias is not None]):
            mha_type = "MultiHeadAttentionWithAttnMask"
            if pos_bias is not None:
                node_inputs.append(pos_bias)
            if attn_mask is not None:
                node_inputs.append(attn_mask)
        else:
            raise ValueError("padding_mask or attn_mask must be provided")
    mha_name = f"{name_prefix}/{mha_type}"
    mha_node = om.create_node(mha_type,
                              mha_name,
                              node_inputs,
                              [mha_name+'_output_0'],
                              num_heads=num_heads,
                              domain="com.microsoft",
                              index=mha_next_node.index-1)
    mha_next_node.replace_input(mha_next_node.inputs[0], mha_node.outputs[0])

    if gemm_node is not None:
        weights = om.get_initializer_value(gemm_node.inputs[1])
        transB = gemm_node.attrs["transB"]
        assert transB == 1
        weights = np.ascontiguousarray(weights.transpose(1, 0))
        om.set_initializer_value(gemm_node.inputs[1], weights)
        new_matmul_name = gemm_node.name.replace("Gemm", "MatMul(Gemm)")
        new_matmul_node = om.create_node("MatMul",
                                        new_matmul_name,
                                        [mha_next_node.outputs[0] if attn_mask is not None else mha_node.outputs[0],
                                         gemm_node.inputs[1]],
                                        [new_matmul_name + "_output_0"],
                                        index=gemm_node.index)
        new_bias_name = gemm_node.name.replace("Gemm", "Add(Gemm)")
        new_add_node = om.create_node("Add",
                                    new_bias_name,
                                    [new_matmul_node.outputs[0], gemm_node.inputs[2]],
                                    [new_bias_name + "_output_0"],
                                    index=new_matmul_node.index+1)
        reshape_next_node.replace_input(gemm_next_node.outputs[0], new_add_node.outputs[0])


def optimize_normal_attention(om: ONNXModifier):
    def _create_new_padding_mask():
        mask_next_node = om.get_to_nodes("attention_mask")[0]
        cast_node = om.create_node("Cast",
                                   "Cast_for_padding_mask",
                                   ["attention_mask"],
                                   ["Cast_for_padding_mask_output_0"],
                                   to=6,  # int32
                                   index=mask_next_node.index)
        reducesum_node = om.create_node("ReduceSum",
                                        "ReduceSum_for_mask",
                                        [cast_node.outputs[0]],
                                        ["ReduceSum_for_mask_output_0"],
                                        axes=1,
                                        keepdims=0,
                                        index=cast_node.index+1)
        return reducesum_node.outputs[0]

    def _create_new_attn_mask(_num_heads: int):
        cast_node = om.get_node("/bert/Cast")
        tile_node = om.create_node("Tile",
                                   f"Tile_for_attn_mask_{_num_heads}heads",
                                   ["/bert/Unsqueeze_output_0",
                                    om.create_initializer(f"{_num_heads}heads_repeats",
                                                          np.array([1, _num_heads, 1, 1], np.int64)).name],
                                   [f"Tile_for_attn_mask_{_num_heads}heads_output_0"],
                                   index=cast_node.index)
        return tile_node.outputs[0]


    def _create_new_attn_mask1(key_seq_len: int):
        reshape_node = om.get_node("/transformer/decoder/layers.0/ca_text/Reshape_3")
        om.set_initializer_value(reshape_node.inputs[1], np.array([-1, 1, TEXT_SEQ_LEN], np.int64))
        expand_node = om.get_node("/transformer/decoder/layers.0/ca_text/Expand")
        om.set_initializer_value(expand_node.inputs[1], np.array([1, key_seq_len, 1], np.int64))
        return expand_node.outputs[0]

    attn_mask1 = _create_new_attn_mask(BERT_NUM_HEADS)
    attn_mask2 = _create_new_attn_mask(ENCODER_NUM_HEADS)
    attn_mask3 = _create_new_attn_mask1(DECODER_VISION_SEQ_LEN)
    zeros_pos_bias = om.create_initializer(
        "/transformer/decoder/layers.x/ca_text/zeros_pos_bias",
        np.zeros((DECODER_NUM_HEADS, DECODER_VISION_SEQ_LEN, TEXT_SEQ_LEN), np.float32)
    ).name

    # bert
    for i in range(12):
        print(f"/bert/encoder/layer.{i}/attention/self/Softmax")
        fuse_one_attention(om,
                           f"/bert/encoder/layer.{i}/attention/self/Softmax",
                           attn_mask=attn_mask1,
                           num_heads=BERT_NUM_HEADS)

    for i in range(6):
        # /transformer/encoder
        fuse_one_attention(om,
                           f"/transformer/encoder/text_layers.{i}/self_attn/Softmax",
                           attn_mask=attn_mask2,
                           num_heads=ENCODER_NUM_HEADS)
        # /transformer/decoder
        fuse_one_attention(om,
                           f"/transformer/decoder/layers.{i}/self_attn/Softmax",
                           num_heads=DECODER_NUM_HEADS)
        fuse_one_attention(om,
                           f"/transformer/decoder/layers.{i}/ca_text/Softmax",
                           attn_mask=attn_mask3,
                           pos_bias=zeros_pos_bias,
                           num_heads=DECODER_NUM_HEADS)

    om.update_map()


def optimize_backbone_attention(om: ONNXModifier):
    def get_original_mask(mask_name, name_prefix):
        mask_value = om.get_initializer_value(mask_name)
        # mask_value = mask_value.astype(np.int64)
        assert mask_value.ndim == 5 and mask_value.shape[2] == 1
        mask_value = mask_value.reshape(mask_value.shape[0] * mask_value.shape[1], mask_value.shape[3], mask_value.shape[4])
        #  = np.where(mask_value==0, 1, 0).astype(np.bool_)

        new_mask_init = om.create_initializer(f"{name_prefix}/mask", mask_value)
        return new_mask_init.name

    def _fuse_one_attention_with_bias(softmax_name: str):
        name_prefix = '/'.join(softmax_name.split('/')[:-2])

        softmax_node = om.get_node(softmax_name)
        tmp_node = om.get_prev_nodes(softmax_node)[0]
        mask = None
        if tmp_node.op_type == "Reshape":
            tmp_node = om.get_prev_nodes(tmp_node)[0]
            assert tmp_node.op_type == "Add"
            mask = get_original_mask(tmp_node.inputs[1], name_prefix)
            tmp_node = om.get_prev_nodes(tmp_node)[0]
            assert tmp_node.op_type == "Reshape"
            tmp_node = om.get_prev_nodes(tmp_node)[0]
        assert tmp_node.op_type == "Add"
        pos_bias_init_value = om.get_initializer_value(tmp_node.inputs[1])
        assert pos_bias_init_value.shape[0] == 1
        om.set_initializer_value(tmp_node.inputs[1], pos_bias_init_value.squeeze(axis=0))
        pos_bias_init = tmp_node.inputs[1]

        tmp_node = om.get_prev_nodes(tmp_node)[0]
        assert tmp_node.op_type == "MatMul"
        qk_matmul = tmp_node
        tmp_node = om.get_from_node(qk_matmul.inputs[0])
        assert tmp_node.op_type == "Mul"
        tmp_node = om.get_prev_nodes(tmp_node)[0]
        assert tmp_node.op_type == "Gather"
        q_gather_node = tmp_node

        tmp_node = om.get_prev_nodes(tmp_node)[0]
        assert tmp_node.op_type == "Transpose"
        tmp_node = om.get_prev_nodes(tmp_node)[0]
        assert tmp_node.op_type == "Reshape"
        reshape_node = tmp_node

        tmp_node = om.get_from_node(qk_matmul.inputs[1])
        assert tmp_node.op_type == "Transpose"
        tmp_node = om.get_prev_nodes(tmp_node)[0]
        assert tmp_node.op_type == "Gather"
        k_gather_node = tmp_node

        tmp_node = om.get_next_nodes(softmax_node)[0]
        assert tmp_node.op_type == "MatMul"
        v_gather_node = om.get_from_node(tmp_node.inputs[1])
        assert v_gather_node.op_type == "Gather"
        tmp_node = om.get_next_nodes(tmp_node)[0]
        assert tmp_node.op_type == "Transpose"
        tmp_node = om.get_next_nodes(tmp_node)[0]
        assert tmp_node.op_type == "Reshape"
        # mha_out = tmp_node.outputs[0]
        mha_out = tmp_node.inputs[0]

        old_dst_shape = om.get_initializer_value(reshape_node.inputs[1])
        b, s, _, h, d = old_dst_shape
        # new_dst_shape = [b, s, _, h*d]
        # new_dst_shape_init = om.create_initializer(f"{name_prefix}/qkv_hidden_states_shape",
        #                                            np.array(new_dst_shape, np.int64))
        # reshape_node.set_input(1, new_dst_shape_init.name)
        for node in [q_gather_node, k_gather_node, v_gather_node]:
            node.set_input(0, reshape_node.outputs[0])
            node.set_attribute("axis", 2)

        mha_name = f"{name_prefix}/MultiHeadAttentionWithAttnMask"
        inputs = [q_gather_node.outputs[0],
                  k_gather_node.outputs[0],
                  v_gather_node.outputs[0],
                  pos_bias_init]
        if mask is not None:
            inputs.append(mask)
        mha_node = om.create_node("MultiHeadAttentionWithAttnMask",
                                    mha_name,
                                    inputs,
                                    [mha_name+'_output_0'],
                                    num_heads=h,
                                    domain="com.microsoft",
                                    index=softmax_node.index)
        mha_next_node = om.get_to_nodes(mha_out)[0]
        mha_next_node.replace_input(mha_out, mha_node.outputs[0])

    num_layers = 4
    for l in range(num_layers):
        num_blocks = 18 if l == 2 else 2
        for b in range(num_blocks):
            _fuse_one_attention_with_bias(f"/backbone/backbone.0/layers.{l}/blocks.{b}/attn/softmax/Softmax")


def optimize_ms_deform_attn(om: ONNXModifier):
    def fuse_ms_deform_attn(value, spatial_shapes, level_start_index, sampling_locations,
                            attention_weights, output):
        value_next_node = om.get_to_nodes(value)[0]
        index = value_next_node.index
        name_prefix = '/'.join(value.split('/')[:-1])
        node_name = f"{name_prefix}/MSDeformAttn"
        fusion_node = om.create_node("MSDeformAttn",
                                     node_name,
                                     [value, spatial_shapes, level_start_index,
                                      sampling_locations, attention_weights],
                                     [f"{node_name}_output_0"],
                                     index=index)
        next_nodes = om.get_to_nodes(output)
        for node in next_nodes:
            node.replace_input(output, fusion_node.outputs[0])

    # spatial_shapes_int = om.create_initializer(
    #     "/transformer/spatial_shapes",
    #     np.array([(100, 150), (50, 75), (25, 38), (13, 19)], dtype=np.int64)
    # )
    # level_start_index_init = om.create_initializer(
    #     "/transformer/level_start_index",
    #     np.array([0, 15000, 18750, 19700], dtype=np.int64)
    # )
    mul_output_lists = [
        "/input_proj.0/input_proj.0.1/Mul_output_0",
        "/input_proj.1/input_proj.1.1/Mul_output_0",
        "/input_proj.2/input_proj.2.1/Mul_output_0",
        "/input_proj.3/input_proj.3.1/Mul_output_0"
    ]
    spatial_shapes_list = []
    level_start_index_list = []
    index = 0
    for mul_output_name in mul_output_lists:
        value_info = om.value_info_map[mul_output_name]
        dim1 = value_info.type.tensor_type.shape.dim[2].dim_value
        dim2 = value_info.type.tensor_type.shape.dim[3].dim_value
        spatial_shapes_list.append((dim1, dim2))
        level_start_index_list.append(index)
        index += (dim1 * dim2)
    spatial_shapes_int = om.create_initializer("/transformer/spatial_shapes",
                                               np.array(spatial_shapes_list, dtype=np.int64))
    level_start_index_init = om.create_initializer("/transformer/level_start_index",
                                                   np.array(level_start_index_list, dtype=np.int64))

    for i in range(6):
        fuse_ms_deform_attn(
            f"/transformer/encoder/layers.{i}/self_attn/Reshape_output_0",
            spatial_shapes_int.name,
            level_start_index_init.name,
            f"/transformer/encoder/layers.{i}/self_attn/Add_output_0",
            f"/transformer/encoder/layers.{i}/self_attn/Reshape_3_output_0",
            f"/transformer/encoder/layers.{i}/self_attn/Transpose_9_output_0"
        )
        fuse_ms_deform_attn(
            f"/transformer/decoder/layers.{i}/cross_attn/Reshape_output_0",
            spatial_shapes_int.name,
            level_start_index_init.name,
            f"/transformer/decoder/layers.{i}/cross_attn/Add_output_0",
            f"/transformer/decoder/layers.{i}/cross_attn/Reshape_3_output_0",
            f"/transformer/decoder/layers.{i}/cross_attn/Transpose_9_output_0"
        )

    om.update_map()


def optimize_bidirect_attention(om: ONNXModifier):
    for i in range(6):
        reduce_max_name = f"/transformer/encoder/fusion_layers.{i}/attn/ReduceMax_1"
        reduce_max_node = om.get_node(reduce_max_name)
        next_node = om.get_next_nodes(reduce_max_node)[0]
        assert next_node.op_type == "Sub"

        name_prefix = '/'.join(reduce_max_name.split('/')[:-1])
        matmul_name = f"{name_prefix}/identity_MatMul"
        matmul_init = om.create_initializer(matmul_name + "_B",
                                            np.diag(np.array([1] * 1)).astype(np.float32))
        matmul_node = om.create_node(
            "MatMul",
            matmul_name,
            [reduce_max_node.outputs[0], matmul_init.name],
            [f"{matmul_name}_output_0"],
            index = reduce_max_node.index+1
        )
        next_node.set_input(1, matmul_node.outputs[0])


def optimize_clip_ndoes(om: ONNXModifier):
    """优化串联的两个clip: clip(min)->clip(max)"""
    pass


def optimize_matmul_nodes(om: ONNXModifier):
    r"""
     input_data
      /   |   \                                                     / slice1
    mm1  mm2  mm3          ==>    input_data - fused_mm - fused_add - slice2
     |    |    |                                                    \ slice3
    add1 add2 add3
    """
    def fuse_parallel_matmul_nodes(input_name: str, name_prefix: str):
        weight_list = []
        bias_list = []
        matmul_indexes = []
        add_indexes = []
        add_nodes = []
        M = None
        for node in om.get_to_nodes(input_name) :
            if node.op_type != "MatMul":
                continue
            if om.get_initializer(node.inputs[1]) is None:
                continue
            if len(om.get_next_nodes(node)) != 1:
                continue
            next_node = om.get_next_nodes(node)[0]
            if next_node.op_type != "Add":
                continue
            if om.get_initializer(next_node.inputs[0]) is None:
                continue
            weight = om.get_initializer_value(node.inputs[1])
            bias = om.get_initializer_value(next_node.inputs[0])
            if M is None:
               M = weight.shape[0]
            else:
                assert M == weight.shape[0]
            assert weight.shape[1] == bias.shape[0]
            weight_list.append(weight)
            bias_list.append(bias)
            matmul_indexes.append(node.index)
            add_indexes.append(next_node.index)
            add_nodes.append(next_node)

        fused_weights = np.concatenate(weight_list, axis=1)
        fused_weights_init = om.create_initializer(f"{name_prefix}/fused_weight", fused_weights)
        fused_bias = np.concatenate(bias_list, axis=0)
        fused_bias_init = om.create_initializer(f"{name_prefix}/fused_bias", fused_bias)
        new_matmul_name = f"{name_prefix}/MatMul(Parallel)"
        new_matmul_node = om.create_node("MatMul",
                                         new_matmul_name,
                                         [input_name, fused_weights_init.name],
                                         [new_matmul_name + "_output_0"],
                                         index=matmul_indexes[0])
        new_add_name = f"{name_prefix}/Add(Parallel)"
        new_add_node = om.create_node("Add",
                                      new_add_name,
                                      [new_matmul_node.outputs[0], fused_bias_init.name],
                                      [new_add_name + "_output_0"],
                                      index=add_indexes[0])
        start = 0
        end = 0
        for i, add_node in enumerate(add_nodes):
            end += bias_list[i].shape[0]
            slice_name = f"{name_prefix}/Slice(Parallel)_{i}"
            slice_node = om.create_node(
                "Slice",
                slice_name,
                [new_add_node.outputs[0],
                 om.create_initializer(f"{name_prefix}/Slice(Parallel)_{i}_starts", np.array([start], np.int64)).name,
                 om.create_initializer(f"{name_prefix}/Slice(Parallel)_{i}_ends", np.array([end], np.int64)).name,
                 om.create_initializer(f"{name_prefix}/Slice(Parallel)_{i}_axes", np.array([2], np.int64)).name,
                 om.create_initializer(f"{name_prefix}/Slice(Parallel)_{i}_steps", np.array([1], np.int64)).name],
                [slice_name + "_output_0"],
                index=add_indexes[i]+1
            )
            # add_nodes[i].set_output(0, "")
            next_nodes = om.get_next_nodes(add_node)
            for node in next_nodes:
                node.replace_input(add_node.outputs[0], slice_node.outputs[0])
            start = end

    # process bert
    for i in range(12):
        if i == 0:
            input_name_ = "/bert/embeddings/LayerNorm/LayerNormalization_output_0"
        else:
            input_name_ = f"/bert/encoder/layer.{i-1}/output/LayerNorm/LayerNormalization_output_0"
        name_prefix_ = f"/bert/encoder/layer.{i}/attention/self"
        fuse_parallel_matmul_nodes(input_name_, name_prefix_)

    # process neck
    for i in range(6):
        input_name_ = f"/transformer/encoder/fusion_layers.{i}/layer_norm_l/LayerNormalization_output_0"
        name_prefix_ = f"/transformer/encoder/fusion_layers.{i}/attn/l"
        fuse_parallel_matmul_nodes(input_name_, name_prefix_)
        input_name_ = f"/transformer/encoder/fusion_layers.{i}/layer_norm_v/LayerNormalization_output_0"
        name_prefix_ = f"/transformer/encoder/fusion_layers.{i}/attn/v"
        fuse_parallel_matmul_nodes(input_name_, name_prefix_)
        input_name_ = f"/transformer/encoder/text_layers.{i}/Add_output_0"
        name_prefix_ = f"/transformer/encoder/text_layers.{i}/self_attn"
        fuse_parallel_matmul_nodes(input_name_, name_prefix_)
        input_name_ = f"/transformer/encoder/layers.{i}/Add_output_0"
        name_prefix_ = f"/transformer/encoder/layers.{i}/self_attn"
        fuse_parallel_matmul_nodes(input_name_, name_prefix_)

    # process head
    for i in range(6):
        input_name_ = f"/transformer/decoder/layers.{i}/Add_output_0"
        name_prefix_ = f"/transformer/decoder/layers.{i}/self_attn"
        fuse_parallel_matmul_nodes(input_name_, name_prefix_)
        input_name_ = f"/transformer/decoder/layers.{i}/Add_4_output_0"
        name_prefix_ = f"/transformer/decoder/layers.{i}/cross_attn"
        fuse_parallel_matmul_nodes(input_name_, name_prefix_)
    input_name_ = "/transformer/encoder/text_layers.5/norm2/LayerNormalization_output_0"
    name_prefix_ = "/transformer/encoder/text_layers.5/fused_proj"
    fuse_parallel_matmul_nodes(input_name_, name_prefix_)
    input_name_ = "/transformer/encoder/layers.5/norm2/LayerNormalization_output_0"
    name_prefix_ = "/transformer/encoder/layers.5/fused_proj"
    fuse_parallel_matmul_nodes(input_name_, name_prefix_)

    om.update_map()


def optimize_gather_nodes(om: ONNXModifier):
    r"""
           input_data                        input_data
          /    |    \            ==>         /    |    \
    gather1 gather2 gather3               slice1 slice2 slice3
    """
    backbone_mha_nodes = [node for node in om.get_nodes("MultiHeadAttentionWithAttnMask")
                          if node.name.startswith("/backbone/")]
    for nha_node in backbone_mha_nodes:
        reshape_node = None
        start = 0
        end = 0
        dim = 0
        for i, input_name in enumerate(nha_node.inputs[:3]):
            gather_node = om.get_from_node(input_name)
            assert gather_node.op_type == "Gather"
            indices = om.get_initializer_value(gather_node.inputs[1]).item()
            assert indices in set([0, 1, 2])
            axis = gather_node.attrs["axis"]
            assert axis == 2
            if i == 0:
                reshape_node = om.get_from_node(gather_node.inputs[0])
                assert reshape_node.op_type == "Reshape"
                dst_shape = om.get_initializer_value(reshape_node.inputs[1])
                assert dst_shape.size == 4 or dst_shape.size == 5
                if dst_shape.size == 5:
                    assert dst_shape[axis] == 3
                    dim = dst_shape[3]
                    om.set_initializer_value(reshape_node.inputs[1],
                                             np.array([dst_shape[0], dst_shape[1], 3*dim, dst_shape[4]],
                                             np.int64))
                else:
                    dim = dst_shape[axis] // 3

            end += dim
            slice_name = f"{gather_node.name}(Slice)"
            slice_node = om.create_node(
                "Slice",
                slice_name,
                [gather_node.inputs[0],
                 om.create_initializer(f"{slice_name}_starts", np.array([start], np.int64)).name,
                 om.create_initializer(f"{slice_name}_ends", np.array([end], np.int64)).name,
                 om.create_initializer(f"{slice_name}_axes", np.array([axis], np.int64)).name,
                 om.create_initializer(f"{slice_name}_steps", np.array([1], np.int64)).name],
                [slice_name + "_output_0"],
                index=gather_node.index+1
            )
            nha_node.set_input(i, slice_node.outputs[0])
            start = end

    om.update_map()


def optimize_gemm_act_nodes(om: ONNXModifier):
    for i, num_blocks in enumerate([2, 2, 18, 2]):
        for j in range(num_blocks):
            matmul_node1 = om.get_node(f"/backbone/backbone.0/layers.{i}/blocks.{j}/mlp/fc1/MatMul")
            bias_node1 = om.get_node(f"/backbone/backbone.0/layers.{i}/blocks.{j}/mlp/fc1/Add")
            matmul_node2 = om.get_node(f"/backbone/backbone.0/layers.{i}/blocks.{j}/mlp/fc2/MatMul")
            new_name = f"/backbone/backbone.0/layers.{i}/blocks.{j}/mlp/GemmAct"
            gemm_act_node =om.create_node(
                "GemmAct",
                new_name,
                [matmul_node1.inputs[0], matmul_node1.inputs[1], bias_node1.inputs[0]],
                [new_name + "_output_0"],
                act_type="gelu",
                index=matmul_node1.index,
            )
            matmul_node2.set_input(0, gemm_act_node.outputs[0])

    om.update_map()


def main():
    input_onnx_path = sys.argv[1]
    output_onnx_path = sys.argv[2]

    om = ONNXModifier(input_onnx_path)
    initialize_global_variables(om)
    optimize_where_ndoes(om)         # 1. 替换where节点
    optimize_transpose_nodes(om)     # 2. 优化transpose节点
    optmize_sin_cos_block(om)        # 3. 优化位置编码
    om.add_opset_import("com.microsoft", 1)
    optimize_normal_attention(om)    # 4. 融合bert、transformer中的mha
    optimize_backbone_attention(om)  # 5. 融合backbone中的注意力
    optimize_ms_deform_attn(om)      # 6. 融合多尺度可变形注意力
    # optimize_bidirect_attention(om)  # 7. 优化双向注意力
    optimize_matmul_nodes(om)         # 8. 融合并行matmul节点
    optimize_gather_nodes(om)         # 9. gather节点替换为slice节点
    optimize_gemm_act_nodes(om)       # 10. 融合gemm_act节点
    om.save(output_onnx_path, save_as_external_data=False)


if __name__ == "__main__":
    main()
