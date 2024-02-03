# PyTorch to Excel compiler

A small proof-of-concept compiler for running a PyTorch model in Excel as a standalone Excel formula. Works by exporting the model to [IR](https://pytorch.org/docs/main/torch.compiler_ir.html#prims-ir) and implementing the IR as Excel functions. Only a very limited number of functions are currently implemented. In addition, it only currently works for at most 2D operations because Excel dynamic array operations only work in 2D. N-dimensional operations aren't theoretically impossible but would be a lot more complex.