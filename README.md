# PyTorch to Excel compiler

A small proof-of-concept compiler for running a PyTorch model in Excel as a standalone Excel formula. Works by exporting the model to [IR](https://pytorch.org/docs/main/torch.compiler_ir.html#prims-ir) and implementing the IR as Excel functions. Only a very limited number of functions are currently implemented.

Implements some basic N-dimensional arrays provided all dimensions other than the final two are fixed size. This is currently very early work and still isn't easily generalisable.

## Usage

Follow the example script: train your model (currently must be quite simple), then run the `compile` function to generate Excel code. Copy this code to excel and call it as you would a `LAMBDA`: add parentheses and select the appropriate input array.