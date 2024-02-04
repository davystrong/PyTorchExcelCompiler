import torch
from torch import nn
import torch.fx
import functorch.compile
import random
import torch.functional as F


def build_excel_array(tensor: torch.Tensor):
    if tensor.dim() == 1:
        # TODO: May need to unsqueeze 1 instead
        tensor = tensor.unsqueeze(0)
    assert tensor.dim() == 2
    return '{' + ';'.join(','.join(str(x.item()) for x in row) for row in tensor) + '}'


def build_excel_function(node: torch.fx.Node) -> str:
    # print(node.target.name())
    if node.kwargs != {}:
        raise NotImplementedError('kwargs not implemented')
    match node.target.name():
        case 'aten::view':
            return f'WRAPROWS(TOROW({node.args[0].name}),{node.args[1][-1]})'
        case 'aten::permute':
            if node.args[1] == [1, 0]:
                return f'TRANSPOSE({node.args[0].name})'
            else:
                return node.args[0].name
        case 'aten::addmm':
            return f'{node.args[0].name}+MMULT({node.args[1].name},{node.args[2].name})'
        case 'aten::relu':
            # MAX doesn't work because it doesn't support broadcasting
            return f'IF({node.args[0].name}>0,{node.args[0].name},0)'
        case 'aten::sigmoid':
            return f'1/(1+EXP(-{node.args[0].name}))'
        case 'aten::argmax':
            dim = node.args[1]
            rank = node.meta['val'].dim()
            if dim < 0:
                dim += rank
            arg = node.args[0].name
            if dim == rank - 1:
                return f'BYROW({arg},LAMBDA(x,MATCH(MAX(x),x,0)))-1'
            else:
                return f'BYCOL({arg},LAMBDA(x,MATCH(MAX(x),x,0)))-1'
        case _:
            raise NotImplementedError(f'Function {node.target.name()} not implemented')

    # Height of an array: =MAX(ROW(I9#))-MIN(ROW(I9#))+1


def compile(model: nn.Module, *args, **kwargs) -> str:
    export = torch.export.export(model, args, kwargs)
    export = export.run_decompositions()

    code = ''

    # TODO: If a node is only used once, we can inline it

    for node in reversed(export.graph.nodes):
        match node.op:
            case 'placeholder':
                pass
            case 'call_function':
                code = f'LET({node.name},{build_excel_function(node)},{code})'
            case 'output':
                output = node.args[0]
                if len(output) > 1:
                    raise NotImplementedError('Multiple outputs not implemented')
                code = output[0].name


    lets = []
    inputs_to_parameters = export.graph_signature.inputs_to_parameters
    parameters_to_inputs = {v: k for k, v in inputs_to_parameters.items()}

    for parameter in export.graph_signature.parameters:
        lets.append(parameters_to_inputs[parameter])
        lets.append(build_excel_array(model.state_dict()[parameter]))
        # lets.append('temp')

    code = f'LET({",".join(lets)},{code})'

    lambda_args = []
    for user_input in export.graph_signature.user_inputs:
        lambda_args.append(user_input)

    code = f'LAMBDA({",".join(lambda_args)},{code})'

    code = '=' + code
    return code
