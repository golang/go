This package contains the compiler's Static Single Assignment form
component. If you're not familiar with SSA, Wikipedia is a good starting
point:

	https://en.wikipedia.org/wiki/Static_single_assignment_form

SSA is useful to perform transformations and optimizations, which can be
found in this package in the form of compiler passes and rewrite rules.
The former can be found in the "passes" array in compile.go, while the
latter are generated from gen/*.rules.

Like most other SSA forms, funcs consist of blocks and values. Values
perform an operation, which is encoded in the form of an operator and a
number of arguments. The semantics of each Op can be found in
gen/*Ops.go.

gen/* is used to generate code in the ssa package. This includes
opGen.go from gen/*Ops.go, and the rewrite*.go files from gen/*.rules.
To regenerate these files, see gen/README.

Blocks can have multiple forms. For example, BlockPlain will always hand
the control flow to another block, and BlockIf will flow to one of two
blocks depending on a value. See block.go for more details.

Values also have types. For example, a constant boolean value will have
a Bool type, and a variable definition value will have a memory type.

The memory type is special - it represents the global memory state. For
example, an Op that takes a memory argument depends on that memory
state, and an Op which has the memory type impacts the state of memory.
This is important so that memory operations are kept in the right order.

For example, take this program:

	func f(a, b *int) {
		*a = 3
		*b = *a
	}

The two generated stores may show up as follows:

	v10 (4) = Store <mem> {int} v6 v8 v1
	v14 (5) = Store <mem> {int} v7 v8 v10

Since the second store has a memory argument v10, it cannot be reordered
before the first store, which sets that global memory state. And the
logic translates to the code; reordering the two assignments would
result in a different program.

A good way to see and get used to the compiler's SSA in action is via
GOSSAFUNC. For example, to see func Foo's initial SSA form and final
generated assembly, one can run:

	GOSSAFUNC=Foo go build

The generated ssa.html file will also contain the SSA func at each of
the compile passes, making it easy to see what each pass does to a
particular program. You can also click on values and blocks to highlight
them, to help follow the control flow and values.
