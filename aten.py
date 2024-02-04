import torch.fx
from utils import register_lambdas, ExcelLambda

@register_lambdas(view_2d='WRAPROWS(TOROW($arr),$size)')
def view(node: torch.fx.Node) -> str:
    return f'view_2d({node.args[0].name},{node.args[1][-1]})'


def permute(node: torch.fx.Node) -> str:
    if node.args[1] == [1, 0]:
        return f'TRANSPOSE({node.args[0].name})'
    else:
        return node.args[0].name

@register_lambdas(addmm_2d='$input+MMULT($mat_1,$mat_2)')
def addmm(node: torch.fx.Node) -> str:
    return f'addmm_2d({node.args[0].name},{node.args[1].name},{node.args[2].name})'

@register_lambdas(relu_2d='IF($input>0,$input,0)')
def relu(node: torch.fx.Node) -> str:
    # MAX doesn't work because it doesn't support broadcasting
    return f'relu_2d({node.args[0].name})'

@register_lambdas(sigmoid_2d='1/(1+EXP(-$input))')
def sigmoid(node: torch.fx.Node) -> str:
    return f'sigmoid_2d{node.args[0].name})'

@register_lambdas(
        argmax0_2d='BYCOL($input,LAMBDA(x,MATCH(MAX(x),x,0)))-1',
        argmax1_2d='BYROW($input,LAMBDA(x,MATCH(MAX(x),x,0)))-1',
)
def argmax(node: torch.fx.Node) -> str:
    dim = node.args[1]
    rank = node.meta['val'].dim()
    if dim < 0:
        dim += rank
    arg = node.args[0].name
    if dim == rank - 1:
        return f'argmax1_2d({arg})'
    else:
        return f'argmax0_2d({arg})'