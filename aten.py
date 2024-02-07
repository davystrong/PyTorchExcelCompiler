from typing import Iterable, Sequence
import torch.fx
from utils import register_lambdas, ExcelLambda
import math
import itertools

@register_lambdas(view_2d='WRAPROWS(TOROW($arr),$size)')
def view(node: torch.fx.Node) -> str:
    # TODO: Broadcasting
    return f'view_2d({node.args[0].name},{node.args[1][-1]})'


def permute(node: torch.fx.Node) -> str:
    # TODO: Broadcasting
    if node.args[1] == [1, 0]:
        return f'TRANSPOSE({node.args[0].name})'
    else:
        return node.args[0].name

@register_lambdas(addmm_2d='$input+MMULT($mat_1,$mat_2)')
def addmm(node: torch.fx.Node) -> str:
    # TODO: Broadcasting
    return f'addmm_2d({node.args[0].name},{node.args[1].name},{node.args[2].name})'

@register_lambdas(relu_2d='IF($input>0,$input,0)')
def relu(node: torch.fx.Node) -> str:
    # TODO: Broadcasting
    # MAX doesn't work because it doesn't support broadcasting
    return f'relu_2d({node.args[0].name})'

@register_lambdas(sigmoid_2d='1/(1+EXP(-$input))')
def sigmoid(node: torch.fx.Node) -> str:
    # TODO: Broadcasting
    return f'sigmoid_2d{node.args[0].name})'

@register_lambdas(
        argmax0_2d='BYCOL($input,LAMBDA(x,MATCH(MAX(x),x,0)))-1',
        argmax1_2d='BYROW($input,LAMBDA(x,MATCH(MAX(x),x,0)))-1',
)
def argmax(node: torch.fx.Node) -> str:
    # TODO: Broadcasting
    dim = node.args[1]
    # TODO: I think this should take the value from arg, not the output
    output_rank = node.meta['val'].dim()
    if dim < 0:
        dim += output_rank
    arg = node.args[0].name
    if dim == output_rank - 1:
        return f'argmax1_2d({arg})'
    else:
        return f'argmax0_2d({arg})'

@register_lambdas(
        sum_dim_IntList0_2d='BYCOL($input,LAMBDA(x,SUM(x)))',
        sum_dim_IntList1_2d='BYROW($input,LAMBDA(x,SUM(x)))',
        sum_dim_IntList_2d='SUM($input)',
)
def sum_dim_IntList(node: torch.fx.Node) -> dict[str, str] | str:
    dims = node.args[1]
    keep_dims = node.args[2] if len(node.args) > 2 else False
    arg_rank = node.args[0].meta['val'].dim()
    dims = [dim if dim < 0 else dim - arg_rank for dim in dims]
    codes = {}
    for output_index in _gen_indices(node.meta['val'].shape[:-2]):
        output_name = '_'.join([node.name, *map(str, output_index)])

        # For each dim not equal to -1 or -2, insert the axis at dim and sum all of them
        # Do this by going through all the values in input_indices and finding all that match all the FIXED indices, i.e. the ones that aren't in dims
        # For -1 or -2, wrap the entire block of code in one of the sum ops

        keep_dims_output_index = output_index[:]
        if not keep_dims:
            for dim in dims:
                if dim < -2:
                    keep_dims_output_index.insert(dim + arg_rank, 0)

        mats_to_sum = []
        for arg_index in _gen_indices(node.args[0].meta['val'].shape[:-2]):
            print(arg_index, keep_dims_output_index, dims)
            if arg_index == []:
                mats_to_sum.append(node.args[0].name)
            
            elif all(x == y for i, (x, y) in enumerate(zip(keep_dims_output_index, arg_index)) if i - arg_rank not in dims):
                mats_to_sum.append('_'.join([node.args[0].name, *map(str, arg_index)]))

        sums = '+'.join(mats_to_sum)

        if -1 in dims and -2 in dims:
            codes[output_name] = f'sum_dim_IntList_2d({sums})'
        elif -2 in dims:
            codes[output_name] = f'sum_dim_IntList0_2d({sums})'
        elif -1 in dims:
            code = f'sum_dim_IntList1_2d({sums})'
            if not keep_dims:
                code = f'TRANSPOSE({code})'
            codes[output_name] = code
        else:
            codes[output_name] = sums

    return codes            
    
def _get_closest_index(shape: Sequence[int], index: Sequence[int]) -> list[int, ...]:
    return [min(i, s - 1) for i, s in reversed(list(zip(index, shape)))]

def add_Tensor(node: torch.fx.Node) -> dict[str, str]:
    codes = {}
    for index in _gen_indices(node.meta['val'].shape[:-2]):
        output_name = '_'.join([node.name, *map(str, index)])
        var_name_0 = '_'.join([node.args[0].name, *map(str, _get_closest_index(node.args[0].meta['val'].shape[:-2], index))])
        var_name_1 = '_'.join([node.args[1].name, *map(str, _get_closest_index(node.args[1].meta['val'].shape[:-2], index))])
        codes[output_name] = f'{var_name_0}+{var_name_1}'
    return codes

def clone(node: torch.fx.Node) -> dict[str, str]:
    codes = {}
    for index in _gen_indices(node.meta['val'].shape[:-2]):
        output_name = '_'.join([node.name, *map(str, index)])
        var_name = '_'.join([node.args[0].name, *map(str, index)])
        codes[output_name] = var_name
    return codes

def _gen_indices(shape: Sequence[int]) -> Iterable[list[int]]:
    for i in range(math.prod(shape)):
        yield [(i // math.prod(shape[j+1:]) % shape[j]) for j in range(len(shape))]
    if math.prod(shape) == 0:
        # This allows handling 2d tensors in the same operation
        yield []

def unsqueeze(node: torch.fx.Node) -> dict[str, str] | str:
    output_rank = node.meta['val'].dim()
    dim = node.args[1]
    if dim >= 0:
        dim -= output_rank
    if output_rank == 2:
        if dim == -1:
            return f'TRANSPOSE({node.args[0].name})'
        else:
            return node.args[0].name
    else:
        codes = {}
        for index in _gen_indices(node.meta['val'].shape[:-2]):
            if dim < -2:
                old_index = [x for i, x in enumerate(index) if i != dim + output_rank]
                code = '_'.join([node.name, *map(str, old_index)])
            else:
                old_index = index[:-1]
                # TODO: This should be 1 indexed
                if dim == -2:
                    code = f'CHOOSEROWS({node.args[0].name},{index[-1]+1})'
                else:
                    code = f'TRANSPOSE(CHOOSEROWS({node.args[0].name},{index[-1]+1}))'
            codes['_'.join([node.name, *map(str, index)])] = code
        return codes

def slice_Tensor(node: torch.fx.Node) -> dict[str, str]:
    output_rank = node.meta['val'].dim()
    dim = node.args[1]
    if dim >= 0:
        dim -= output_rank
    codes = {}
    for index in _gen_indices(node.meta['val'].shape[:-2]):
        var_name = '_'.join([node.args[0].name, *map(str, index)])
        if dim < -2:
            old_index = [x if i != dim + output_rank else x + node.args[2] for i, x in enumerate(index)]
            code = '_'.join([node.name, *map(str, old_index)])
        else:
            if dim == -1:
                return f'TAKE(DROP({var_name},{node.args[2]}),{node.args[3]-node.args[2]})'
            else:
                return f'TAKE(DROP({var_name},,{node.args[2]}),,{node.args[3]-node.args[2]})'
        codes[var_name] = code
    return codes
