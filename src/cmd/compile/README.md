<!---
// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
-->

## Introduction to the Go compiler

`cmd/compile` contains the main packages that form the Go compiler. The compiler
may be logically split in four phases, which we will briefly describe alongside
the list of packages that contain their code.

You may sometimes hear the terms "front-end" and "back-end" when referring to
the compiler. Roughly speaking, these translate to the first two and last two
phases we are going to list here. A third term, "middle-end", often refers to
much of the work that happens in the second phase.

Note that the `go/*` family of packages, such as `go/parser` and
`go/types`, are mostly unused by the compiler. Since the compiler was
initially written in C, the `go/*` packages were developed to enable
writing tools working with Go code, such as `gofmt` and `vet`.
However, over time the compiler's internal APIs have slowly evolved to
be more familiar to users of the `go/*` packages.

It should be clarified that the name "gc" stands for "Go compiler", and has
little to do with uppercase "GC", which stands for garbage collection.

### 1. Parsing

* `cmd/compile/internal/syntax` (lexer, parser, syntax tree)

In the first phase of compilation, source code is tokenized (lexical analysis),
parsed (syntax analysis), and a syntax tree is constructed for each source
file.

Each syntax tree is an exact representation of the respective source file, with
nodes corresponding to the various elements of the source such as expressions,
declarations, and statements. The syntax tree also includes position information
which is used for error reporting and the creation of debugging information.

### 2. Type checking

* `cmd/compile/internal/types2` (type checking)

The types2 package is a port of `go/types` to use the syntax package's
AST instead of `go/ast`.

### 3. IR construction ("noding")

* `cmd/compile/internal/types` (compiler types)
* `cmd/compile/internal/ir` (compiler AST)
* `cmd/compile/internal/typecheck` (AST transformations)
* `cmd/compile/internal/noder` (create compiler AST)

The compiler middle end uses its own AST definition and representation of Go
types carried over from when it was written in C. All of its code is written in
terms of these, so the next step after type checking is to convert the syntax
and types2 representations to ir and types. This process is referred to as
"noding."

There are currently two noding implementations:

1. irgen (aka "-G=3" or sometimes "noder2") is the implementation used starting
   with Go 1.18, and

2. Unified IR is another, in-development implementation (enabled with
   `GOEXPERIMENT=unified`), which also implements import/export and inlining.

Up through Go 1.18, there was a third noding implementation (just
"noder" or "-G=0"), which directly converted the pre-type-checked
syntax representation into IR and then invoked package typecheck's
type checker. This implementation was removed after Go 1.18, so now
package typecheck is only used for IR transformations.

### 4. Middle end

* `cmd/compile/internal/deadcode` (dead code elimination)
* `cmd/compile/internal/inline` (function call inlining)
* `cmd/compile/internal/devirtualize` (devirtualization of known interface method calls)
* `cmd/compile/internal/escape` (escape analysis)

Several optimization passes are performed on the IR representation:
dead code elimination, (early) devirtualization, function call
inlining, and escape analysis.

### 5. Walk

* `cmd/compile/internal/walk` (order of evaluation, desugaring)

The final pass over the IR representation is "walk," which serves two purposes:

1. It decomposes complex statements into individual, simpler statements,
   introducing temporary variables and respecting order of evaluation. This step
   is also referred to as "order."

2. It desugars higher-level Go constructs into more primitive ones. For example,
   `switch` statements are turned into binary search or jump tables, and
   operations on maps and channels are replaced with runtime calls.

### 6. Generic SSA

* `cmd/compile/internal/ssa` (SSA passes and rules)
* `cmd/compile/internal/ssagen` (converting IR to SSA)

In this phase, IR is converted into Static Single Assignment (SSA) form, a
lower-level intermediate representation with specific properties that make it
easier to implement optimizations and to eventually generate machine code from
it.

During this conversion, function intrinsics are applied. These are special
functions that the compiler has been taught to replace with heavily optimized
code on a case-by-case basis.

Certain nodes are also lowered into simpler components during the AST to SSA
conversion, so that the rest of the compiler can work with them. For instance,
the copy builtin is replaced by memory moves, and range loops are rewritten into
for loops. Some of these currently happen before the conversion to SSA due to
historical reasons, but the long-term plan is to move all of them here.

Then, a series of machine-independent passes and rules are applied. These do not
concern any single computer architecture, and thus run on all `GOARCH` variants.
These passes include dead code elimination, removal of
unneeded nil checks, and removal of unused branches. The generic rewrite rules
mainly concern expressions, such as replacing some expressions with constant
values, and optimizing multiplications and float operations.

### 7. Generating machine code

* `cmd/compile/internal/ssa` (SSA lowering and arch-specific passes)
* `cmd/internal/obj` (machine code generation)

The machine-dependent phase of the compiler begins with the "lower" pass, which
rewrites generic values into their machine-specific variants. For example, on
amd64 memory operands are possible, so many load-store operations may be combined.

Note that the lower pass runs all machine-specific rewrite rules, and thus it
currently applies lots of optimizations too.

Once the SSA has been "lowered" and is more specific to the target architecture,
the final code optimization passes are run. This includes yet another dead code
elimination pass, moving values closer to their uses, the removal of local
variables that are never read from, and register allocation.

Other important pieces of work done as part of this step include stack frame
layout, which assigns stack offsets to local variables, and pointer liveness
analysis, which computes which on-stack pointers are live at each GC safe point.

At the end of the SSA generation phase, Go functions have been transformed into
a series of obj.Prog instructions. These are passed to the assembler
(`cmd/internal/obj`), which turns them into machine code and writes out the
final object file. The object file will also contain reflect data, export data,
and debugging information.

### Further reading

To dig deeper into how the SSA package works, including its passes and rules,
head to [cmd/compile/internal/ssa/README.md](internal/ssa/README.md).
