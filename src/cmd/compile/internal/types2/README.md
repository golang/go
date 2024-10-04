This file describes some of the typecheckers internal organization and conventions.
It is not meant to be complete; rather it is a living document that will be updated
as needed.

Read this file first before starting to make changes in the code.

#
### Overall organization

There are two almost identical typecheckers:

- cmd/compile/internal/types2 (or types2 for short)
- go/types

types2 is internal and used by the compiler.
go/types is the std library typechecker and its API must remain strictly
backward-compatible.
The types2 API closely matches the go/types API but may not have some
deprecated functions anymore (which we need to maintain in go/types).

They differ primarily in what syntax tree they operate on:

- types2 uses the syntax tree defined by cmd/compile/internal/syntax
- go/types uses the syntax tree defined by go/ast

We aim to keep the respective sources very closely in sync.
**Any change will need to be made to both typechecker source bases**.

Many go/types files can be generated automatically from the
corresponding types2 sources.
This is done via a generator (go/types/generate_test.go) which may be invoked via
`go generate` in the go/types directory.
Generated files are clearly marked with a comment at the top and should not
be modified by hand.
For this reason, it is usally best to make changes to the types2 sources first.
The changes only need to be ported by hand for the go/types files that cannot
be generated yet.

New files may be added to the list of generated files by adding a respective
entry to the table in generate_test.go (and possibly describing any necessary
source transformations).

In the following, examples and commands are based on types2 but usually apply
directly to go/types.


#
### Tests

There is a comprehensive suite of tests in the form of annotated source files.
The tests are in:

- src/internal/types/testdata/ (shared between go/types and types2)
- ./testdata/local (typechecker local tests, for rare situations only)

Tests are .go files annotated with `/* ERROR "msg" */` or `/* ERRORx "msg" */`
comments (or the respective line comment form).
For each such error comment, typechecking the respective file is expected to
report an error at the position of the syntactic token _immediately preceeding_
the comment.
For `ERROR`, the `"msg"` string must be a substring of the error message
reported by the typechecker;
for `ERRORx`, the `"msg"` string must be a regular expresspion matching the
reported error.

For each issue #NNNN that is fixed in the typecheckers, a test
should be added as src/internal/types/testdata/fixedbugs/issueNNNN.go.


#
### Debugging

The pre-existing template ./testdata/manual.go is convenient for debugging
on-off situations. Simply populate it with the code of interest and then
run `go test -run Manual` which will typecheck that file.

Useful debugging flags (together with `go test -run Manual`):

- -halt (panic and produce a stack trace where the first error is reported)
- -v    (produce a typechecking trace)
- -verify       (verify `ERROR` comments in manual.go)


#
### Frequently used types and variables

#### Checker

File: check.go

A `Checker` maintains all typechecking state relevant for typechecking a package.
Typically the receiver type for typechecker methods.


#### operand

File: operand.go

An `operand` describes the type and value (if any) of an expression.
The `operandMode` describes the kind of expression (constant, variable, etc.).
Operands are the primary result of typechecking an expression.
If typechecking of an expression fails, the resulting operand has mode `invalid`.


#### Typ

File: universe.go

The `Typ` array provides access to all predeclared basic types.
`Typ[Invalid]` is used to denote an invalid type.


#
### Internal coding conventions

#### Predicates

File: predicates.go (commonly used predicates only)

Predicates are typically named in form `isX`, such as `isInteger`.

#### Type-checking expressions

Typically, there is a Checker method for typechecking a particular expression.
For instance, there is a method `Checker.unary` that typechecks unary expressions.
The basic form of such a function f is as follows:
```
func (check *Checker) f(x *operand, e syntax.Expr, /* addition arguments, if any */)
```
The result of typechecking expression `e` is returned via the operand `x`
(which sometimes also serves as incoming argument).
If an error occured the function f will report the error and try to continue
as best as it can, but it may return an invalid operand (`x.mode == invalid`).
Callers may need to explicitly check for invalid operands.


#
### TODO

Add more relevant content.
