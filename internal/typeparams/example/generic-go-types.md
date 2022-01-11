<!-- To regenerate the readme, run: -->
<!-- go run golang.org/x/example/gotypes@latest generic-go-types.md -->

# Updating tools to support type parameters.

This guide is maintained by Rob Findley (`rfindley@google.com`).

**status**: this document is currently a work-in-progress. See
[golang/go#50447](https://go.dev/issues/50447) for more details.

%toc

# Introduction

With Go 1.18, Go now supports generic programming via type parameters. This
document is intended to serve as a guide for tool authors that want to update
their tools to support the new language constructs introduced for generic Go.

This guide assumes some knowledge of the language changes to support generics.
See the following references for more information:

- The [original proposal](https://go.dev/issue/43651) for type parameters.
- The [addendum for type sets](https://go.dev/issue/45346).
- The [latest language specfication](https://tip.golang.org/ref/spec) (still in-progress as of 2021-01-11).
- The proposals for new APIs in
  [go/token and go/ast](https://go.dev/issue/47781), and in
  [go/types](https://go.dev/issue/47916).

It also assumes existing knowledge of `go/ast` and `go/types`. If you're just
getting started,
[x/example/gotypes](https://github.com/golang/example/tree/master/gotypes) is
a great introduction (and was the inspiration for this guide).

# Summary of new language features and their APIs

While generic Go programming is a large change to the language, at a high level
it introduces only a few new concepts. Specifically, we can break down our
discussion into the following three broad categories. In each category, the
relevant new APIs are listed (some constructors and getters/setters may be
elided where they are trivial).

**Generic types**. Types and functions may be _generic_, meaning their
declaration has a non-empty _type parameter list_: as in `type  List[T any]
...` or `func f[T1, T2 any]() { ... }`. Type parameter lists define placeholder
types (_type parameters_), scoped to the declaration, which may be substituted
by any type satisfying their corresponding _constraint interface_ to
_instantiate_ a new type or function.

Generic types may have methods, which declare `receiver type parameters` via
their receiver type expression: `func (r T[P1, ..., PN]) method(...) (...)
{...}`.

_New APIs_:
 - The field `ast.TypeSpec.TypeParams` holds the type parameter list syntax for
   type declarations.
 - The field `ast.FuncType.TypeParams` holds the type parameter list syntax for
   function declarations.
 - The type `types.TypeParam` is a `types.Type` representing a type parameter.
   On this type, the `Constraint` and `SetConstraint` methods allow
   getting/setting the constraint, the `Index` method returns the index of the
   type parameter in the type parameter list that declares it, and the `Obj`
   method returns the object declared in the declaration scope for the type
   parameter (a `types.TypeName`).
 - The type `types.TypeParamList` holds a list of type parameters.
 - The method `types.Named.TypeParams` returns the type parameters for a type
   declaration.
 - The method `types.Named.SetTypeParams` sets type parameters on a defined
   type.
 - The function `types.NewSignatureType` creates a new (possibly generic)
   signature type.
 - The method `types.Signature.RecvTypeParams` returns the receiver type
   parameters for a method.
 - The method `types.Signature.TypeParams` returns the type parameters for
   a function.

**Constraint Interfaces**: type parameter constraints are interfaces, expressed
via an interface type expression. Interfaces that are only used in constraint
position are permitted new embedded elements composed of tilde expressions
(`~T`) and unions (`A | B | ~C`). The new builtin interface type `comparable`
is implemented by types for which `==` and `!=` are valid. As a special case,
the `interface` keyword may be omitted from constraint expressions if it may be
implied (in which case we say the interface is _implicit_).

_New APIs_:
 - The constant `token.TILDE` is used to represent tilde expressions as an
   `ast.UnaryExpr`.
 - Union expressions are represented as an `ast.BinaryExpr` using `|`. This
   means that `ast.BinaryExpr` may now be both a type and value expression.
 - The method `types.Interface.IsImplicit` reports whether the `interface`
   keyword was elided from this interface.
 - The method `types.Interface.MarkImplicit` marks an interface as being
   implicit.
 - The method `types.Interface.IsComparable` reports whether every type in an
   interface's type set is comparable.
 - The method `types.Interface.IsMethodSet` reports whether an interface is
   defined entirely by its methods (has no _specific types_).
 - The type `types.Union` is a type that represents an embedded union
   expression in an interface. May only appear as an embedded element in
   interfaces.
 - The type `types.Term` represents a (possibly tilde) term of a union.

**Instantiation**: generic types and functions may be _instantiated_ to create
non-generic types and functions by providing _type arguments_ (`var x T[int]`).
Function type arguments may be _inferred_ via function arguments, or via
type parameter constraints.

_New APIs_:
 - The type `ast.IndexListExpr` holds index expressions with multiple indices,
   as occurs in instantiation expressions with multiple type arguments, or in
   receivers with multiple type parameters.
 - The function `types.Instantiate` instantiates a generic type with type arguments.
 - The type `types.Context` is an opaque instantiation context that may be
   shared to reduce duplicate instances.
 - The field `types.Config.Context` holds a shared `Context` to use for
   instantiation while type-checking.
 - The type `types.TypeList` holds a list of types.
 - The type `types.ArgumentError` holds an error associated with a specific
   argument index. Used to represent instantiation errors.
 - The field `types.Info.Instances` maps instantiated identifiers to information
   about the resulting type instance.
 - The type `types.Instance` holds information about a type or function
   instance.
 - The method `types.Named.TypeArgs` reports the type arguments used to
   instantiate a named type.

# Examples

The following examples demonstrate the new APIs above, and discuss their
properties. All examples are runnable, contained in subdirectories of the
directory holding this README.

## Generic types

### Type parameter lists

Suppose we want to understand the generic library below, which defines a generic
`Pair`, a constraint interface `Constraint`, and a generic function `MakePair`.

%include findtypeparams/main.go input -

We can use the new `TypeParams` fields in `ast.TypeSpec` and `ast.FuncType` to
access the syntax of the type parameter list. From there, we can access type
parameter types in at least three ways:
 - by looking up type parameter definitions in `types.Info`
 - by calling `TypeParams()` on `types.Named` or `types.Signature`
 - by looking up type parameter objects in the declaration scope. Note that
   there now may be a scope associated with an `ast.TypeSpec` node.

%include findtypeparams/main.go print -

This program produces the following output. Note that not every type spec has
a scope.

%include findtypeparams/main.go output -

### Methods on generic types

**TODO**

## Constraint Interfaces

### New interface elements

**TODO**

### Implicit interfaces

**TODO**

### Type sets

**TODO**

## Instantiation

### Finding instantiated types

**TODO**

### Creating new instantiated types

**TODO**

### Using a shared context

**TODO**

# Updating tools while building at older Go versions

In the examples above, we can see how a lot of the new APIs integrate with
existing usage of `go/ast` or `go/types`. However, most tools still need to
build at older Go versions, and handling the new language constructs in-line
will break builds at older Go versions.

For this purpose, the `x/exp/typeparams` package provides functions and types
that proxy the new APIs (with stub implementations at older Go versions).
**NOTE**: does not yet exist -- see
[golang/go#50447](https://go.dev/issues/50447) for more information.

# Further help

If you're working on updating a tool to support generics, and need help, please
feel free to reach out for help in any of the following ways:
 - Via the [golang-tools](https://groups.google.com/g/golang-tools) mailing list.
 - Directly to me via email (`rfindley@google.com`).
 - For bugs, you can [file an issue](https://github.com/golang/go/issues/new/choose).
