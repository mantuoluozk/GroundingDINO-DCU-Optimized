#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
from typing import List

import onnx
from onnx import shape_inference
from onnx.utils import Extractor


def extract_model(
        input_path: str,
        output_path: str,
        input_names: List[str],
        output_names: List[str],
        check_model: bool = False,
        infer_shape: bool = False,
        save_external_data: bool = False
    ) -> None:
    """Extracts sub-model from an ONNX model.

    this implemention modified from
        https://github.com/onnx/onnx/blob/v1.12.0/onnx/utils.py

    The sub-model is defined by the names of the input and output tensors *exactly*.

    Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
    which is defined by the input and output tensors, should not _cut through_ the
    subgraph that is connected to the _main graph_ as attributes of these operators.

    Arguments:
        input_path (string): The path to original ONNX model.
        output_path (string): The path to save the extracted ONNX model.
        input_names (list of string): The names of the input tensors that to be extracted.
        output_names (list of string): The names of the output tensors that to be extracted.
        check_model (bool): Whether to run model checker on the extracted model.
        infer_shape (bool): Whether to infer shapes for all nodes.
    """
    if not os.path.exists(input_path):
        raise ValueError(f"Invalid input model path: {input_path}")
    if not output_path:
        raise ValueError("Output model path shall not be empty!")

    if check_model:
        onnx.checker.check_model(input_path)
    model = onnx.load(input_path)

    if infer_shape:
        model = shape_inference.infer_shapes(model)

    value_info_names = set([v.name for v in model.graph.value_info])
    if not input_names:
        input_names = [i.name for i in model.graph.input if i.name not in value_info_names]
    if not output_names:
        output_names = [o.name for o in model.graph.output if o.name not in value_info_names]

    print("input_names:", input_names)
    print("output_names:", output_names)

    e = Extractor(model)
    extracted = e.extract_model(input_names, output_names)

    if save_external_data:
        external_data_path = osp.basename(output_path) + '.data'
        if osp.isfile(external_data_path):
            os.remove(external_data_path)
        onnx.save(extracted,
                output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_path)
    else:
        onnx.save(extracted, output_path,)

    if check_model:
        onnx.checker.check_model(output_path)



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-onnx', type=str,
                        help='The path to original ONNX model.')
    parser.add_argument('-o', '--output-onnx', type=str,
                        help='The path to save the extracted ONNX model.')
    parser.add_argument('-in', '--input-names', type=str, nargs='+',
                        help='The names of the input tensors that to be extracted.')
    parser.add_argument('-on', '--output-names', type=str, nargs='+',
                        help='The names of the output tensors that to be extracted.')
    parser.add_argument('--check-model', action='store_true',
                        help='Whether to run model checker on the extracted model.')
    parser.add_argument('--infer-shape', action='store_true',
                        help='Whether to infer shapes for all nodes.')
    parser.add_argument('--save-external-data', action='store_true',
                        help="save data to externel")
    args = parser.parse_args()

    extract_model(
        args.input_onnx, args.output_onnx, args.input_names, args.output_names,
        check_model=args.check_model, infer_shape=args.infer_shape,
        save_external_data=args.save_external_data
    )
