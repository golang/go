# Checking Go Package API Compatibility

The `apidiff` tool in this directory determines whether two versions of the same
package are compatible. The goal is to help the developer make an informed
choice of semantic version after they have changed the code of their module.

`apidiff` reports two kinds of changes: incompatible ones, which require
incrementing the major part of the semantic version, and compatible ones, which
require a minor version increment. If no API changes are reported but there are
code changes that could affect client code, then the patch version should
be incremented.

Because `apidiff` ignores package import paths, it may be used to display API
differences between any two packages, not just different versions of the same
package.

The current version of `apidiff` compares only packages, not modules.


## Compatibility Desiderata

Any tool that checks compatibility can offer only an approximation. No tool can
detect behavioral changes; and even if it could, whether a behavioral change is
a breaking change or not depends on many factors, such as whether it closes a
security hole or fixes a bug. Even a change that causes some code to fail to
compile may not be considered a breaking change by the developers or their
users. It may only affect code marked as experimental or unstable, for
example, or the break may only manifest in unlikely cases.

For a tool to be useful, its notion of compatibility must be relaxed enough to
allow reasonable changes, like adding a field to a struct, but strict enough to
catch significant breaking changes. A tool that is too lax will miss important
incompatibilities, and users will stop trusting it; one that is too strict may
generate so much noise that users will ignore it.

To a first approximation, this tool reports a change as incompatible if it could
cause client code to stop compiling. But `apidiff` ignores five ways in which
code may fail to compile after a change. Three of them are mentioned in the
[Go 1 Compatibility Guarantee](https://golang.org/doc/go1compat).

### Unkeyed Struct Literals

Code that uses an unkeyed struct literal would fail to compile if a field was
added to the struct, making any such addition an incompatible change. An example:

```
// old
type Point struct { X, Y int }

// new
type Point struct { X, Y, Z int }

// client
p := pkg.Point{1, 2} // fails in new because there are more fields than expressions
```
Here and below, we provide three snippets: the code in the old version of the
package, the code in the new version, and the code written in a client of the package,
which refers to it by the name `pkg`. The client code compiles against the old
code but not the new.

### Embedding and Shadowing

Adding an exported field to a struct can break code that embeds that struct,
because the newly added field may conflict with an identically named field
at the same struct depth. A selector referring to the latter would become
ambiguous and thus erroneous.


```
// old
type Point struct { X, Y int }

// new
type Point struct { X, Y, Z int }

// client
type z struct { Z int }

var v struct {
    pkg.Point
    z
}

_ = v.Z // fails in new
```
In the new version, the last line fails to compile because there are two embedded `Z`
fields at the same depth, one from `z` and one from `pkg.Point`.


### Using an Identical Type Externally

If it is possible for client code to write a type expression representing the
underlying type of a defined type in a package, then external code can use it in
assignments involving the package type, making any change to that type incompatible.
```
// old
type Point struct { X, Y int }

// new
type Point struct { X, Y, Z int }

// client
var p struct { X, Y int } = pkg.Point{} // fails in new because of Point's extra field
```
Here, the external code could have used the provided name `Point`, but chose not
to. I'll have more to say about this and related examples later.

### unsafe.Sizeof and Friends

Since `unsafe.Sizeof`, `unsafe.Offsetof` and `unsafe.Alignof` are constant
expressions, they can be used in an array type literal:

```
// old
type S struct{ X int }

// new
type S struct{ X, y int }

// client
var a [unsafe.Sizeof(pkg.S{})]int = [8]int{} // fails in new because S's size is not 8
```
Use of these operations could make many changes to a type potentially incompatible.


### Type Switches

A package change that merges two different types (with same underlying type)
into a single new type may break type switches in clients that refer to both
original types:

```
// old
type T1 int
type T2 int

// new
type T1 int
type T2 = T1

// client
switch x.(type) {
case T1:
case T2:
} // fails with new because two cases have the same type
```
This sort of incompatibility is sufficiently esoteric to ignore; the tool allows
merging types.

## First Attempt at a Definition

Our first attempt at defining compatibility captures the idea that all the
exported names in the old package must have compatible equivalents in the new
package.

A new package is compatible with an old one if and only if:
- For every exported package-level name in the old package, the same name is
  declared in the new at package level, and
- the names denote the same kind of object (e.g. both are variables), and
- the types of the objects are compatible.

We will work out the details (and make some corrections) below, but it is clear
already that we will need to determine what makes two types compatible. And
whatever the definition of type compatibility, it's certainly true that if two
types are the same, they are compatible. So we will need to decide what makes an
old and new type the same. We will call this sameness relation _correspondence_.

## Type Correspondence

Go already has a definition of when two types are the same:
[type identity](https://golang.org/ref/spec#Type_identity).
But identity isn't adequate for our purpose: it says that two defined
types are identical if they arise from the same definition, but it's unclear
what "same" means when talking about two different packages (or two versions of
a single package).

The obvious change to the definition of identity is to require that old and new
[defined types](https://golang.org/ref/spec#Type_definitions)
have the same name instead. But that doesn't work either, for two
reasons. First, type aliases can equate two defined types with different names:

```
// old
type E int

// new
type t int
type E = t
```
Second, an unexported type can be renamed:

```
// old
type u1 int
var V u1

// new
type u2 int
var V u2
```
Here, even though `u1` and `u2` are unexported, their exported fields and
methods are visible to clients, so they are part of the API. But since the name
`u1` is not visible to clients, it can be changed compatibly. We say that `u1`
and `u2` are _exposed_: a type is exposed if a client package can declare variables of that type.

We will say that an old defined type _corresponds_ to a new one if they have the
same name, or one can be renamed to the other without otherwise changing the
API. In the first example above, old `E` and new `t` correspond. In the second,
old `u1` and new `u2` correspond.

Two or more old defined types can correspond to a single new type: we consider
"merging" two types into one to be a compatible change. As mentioned above,
code that uses both names in a type switch will fail, but we deliberately ignore
this case. However, a single old type can correspond to only one new type.

So far, we've explained what correspondence means for defined types. To extend
the definition to all types, we parallel the language's definition of type
identity. So, for instance, an old and a new slice type correspond if their
element types correspond.

## Definition of Compatibility

We can now present the definition of compatibility used by `apidiff`.

### Package Compatibility

> A new package is compatible with an old one if:
>1. Each exported name in the old package's scope also appears in the new
>package's scope, and the object (constant, variable, function or type) denoted
>by that name in the old package is compatible with the object denoted by the
>name in the new package, and
>2. For every exposed type that implements an exposed interface in the old package,
> its corresponding type should implement the corresponding interface in the new package.
>
>Otherwise the packages are incompatible.

As an aside, the tool also finds exported names in the new package that are not
exported in the old, and marks them as compatible changes.

Clause 2 is discussed further in "Whole-Package Compatibility."

### Object Compatibility

This section provides compatibility rules for constants, variables, functions
and types.

#### Constants

>A new exported constant is compatible with an old one of the same name if and only if
>1. Their types correspond, and
>2. Their values are identical.

It is tempting to allow changing a typed constant to an untyped one. That may
seem harmless, but it can break code like this:

```
// old
const C int64 = 1

// new
const C = 1

// client
var x = C          // old type is int64, new is int
var y int64 = x // fails with new: different types in assignment
```

A change to the value of a constant can break compatiblity if the value is used
in an array type:

```
// old
const C = 1

// new
const C = 2

// client
var a [C]int = [1]int{} // fails with new because [2]int and [1]int are different types
```
Changes to constant values are rare, and determining whether they are compatible
or not is better left to the user, so the tool reports them.

#### Variables

>A new exported variable is compatible with an old one of the same name if and
>only if their types correspond.

Correspondence doesn't look past names, so this rule does not prevent adding a
field to `MyStruct` if the package declares `var V MyStruct`. It does, however, mean that

```
var V struct { X int }
```
is incompatible with
```
var V struct { X, Y int }
```
I discuss this at length below in the section "Compatibility, Types and Names."

#### Functions

>A new exported function or variable is compatible with an old function of the
>same name if and only if their types (signatures) correspond.

This rule captures the fact that, although many signature changes are compatible
for all call sites, none are compatible for assignment:

```
var v func(int) = pkg.F
```
Here, `F` must be of type `func(int)` and not, for instance, `func(...int)` or `func(interface{})`.

Note that the rule permits changing a function to a variable. This is a common
practice, usually done for test stubbing, and cannot break any code at compile
time.

#### Exported Types

> A new exported type is compatible with an old one if and only if their
> names are the same and their types correspond.

This rule seems far too strict. But, ignoring aliases for the moment, it demands only
that the old and new _defined_ types correspond. Consider:
```
// old
type T struct { X int }

// new
type T struct { X, Y int }
```
The addition of `Y` is a compatible change, because this rule does not require
that the struct literals have to correspond, only that the defined types
denoted by `T` must correspond. (Remember that correspondence stops at type
names.)

If one type is an alias that refers to the corresponding defined type, the
situation is the same:

```
// old
type T struct { X int }

// new
type u struct { X, Y int }
type T = u
```
Here, the only requirement is that old `T` corresponds to new `u`, not that the
struct types correspond. (We can't tell from this snippet that the old `T` and
the new `u` do correspond; that depends on whether `u` replaces `T` throughout
the API.)

However, the following change is incompatible, because the names do not
denote corresponding types:

```
// old
type T = struct { X int }

// new
type T = struct { X, Y int }
```
### Type Literal Compatibility

Only five kinds of types can differ compatibly: defined types, structs,
interfaces, channels and numeric types. We only consider the compatibility of
the last four when they are the underlying type of a defined type. See
"Compatibility, Types and Names" for a rationale.

We justify the compatibility rules by enumerating all the ways a type
can be used, and by showing that the allowed changes cannot break any code that
uses values of the type in those ways.

Values of all types can be used in assignments (including argument passing and
function return), but we do not require that old and new types are assignment
compatible. That is because we assume that the old and new packages are never
used together: any given binary will link in either the old package or the new.
So in describing how a type can be used in the sections below, we omit
assignment.

Any type can also be used in a type assertion or conversion. The changes we allow
below may affect the run-time behavior of these operations, but they cannot affect
whether they compile. The only such breaking change would be to change
the type `T` in an assertion `x.T` so that it no longer implements the interface
type of `x`; but the rules for interfaces below disallow that.

> A new type is compatible with an old one if and only if they correspond, or
> one of the cases below applies.

#### Defined Types

Other than assignment, the only ways to use a defined type are to access its
methods, or to make use of the properties of its underlying type. Rule 2 below
covers the latter, and rules 3 and 4 cover the former.

> A new defined type is compatible with an old one if and only if all of the
> following hold:
>1. They correspond.
>2. Their underlying types are compatible.
>3. The new exported value method set is a superset of the old.
>4. The new exported pointer method set is a superset of the old.

An exported method set is a method set with all unexported methods removed.
When comparing methods of a method set, we require identical names and
corresponding signatures.

Removing an exported method is clearly a breaking change. But removing an
unexported one (or changing its signature) can be breaking as well, if it
results in the type no longer implementing an interface. See "Whole-Package
Compatibility," below.

#### Channels

> A new channel type is compatible with an old one if
>  1. The element types correspond, and
>  2. Either the directions are the same, or the new type has no direction.

Other than assignment, the only ways to use values of a channel type are to send
and receive on them, to close them, and to use them as map keys. Changes to a
channel type cannot cause code that closes a channel or uses it as a map key to
fail to compile, so we need not consider those operations.

Rule 1 ensures that any operations on the values sent or received will compile.
Rule 2 captures the fact that any program that compiles with a directed channel
must use either only sends, or only receives, so allowing the other operation
by removing the channel direction cannot break any code.


#### Interfaces

> A new interface is compatible with an old one if and only if:
> 1. The old interface does not have an unexported method, and it corresponds
>    to the new interfaces (i.e. they have the same method set), or
> 2. The old interface has an unexported method and the new exported method set is a
>    superset of the old.

Other than assignment, the only ways to use an interface are to implement it,
embed it, or call one of its methods. (Interface values can also be used as map
keys, but that cannot cause a compile-time error.)

Certainly, removing an exported method from an interface could break a client
call, so neither rule allows it.

Rule 1 also disallows adding a method to an interface without an existing unexported
method. Such an interface can be implemented in client code. If adding a method
were allowed, a type that implements the old interface could fail to implement
the new one:

```
type I interface { M1() }         // old
type I interface { M1(); M2() }   // new

// client
type t struct{}
func (t) M1() {}
var i pkg.I = t{} // fails with new, because t lacks M2
```

Rule 2 is based on the observation that if an interface has an unexported
method, the only way a client can implement it is to embed it.
Adding a method is compatible in this case, because the embedding struct will
continue to implement the interface. Adding a method also cannot break any call
sites, since no program that compiles could have any such call sites.

#### Structs

> A new struct is compatible with an old one if all of the following hold:
> 1. The new set of top-level exported fields is a superset of the old.
> 2. The new set of _selectable_ exported fields is a superset of the old.
> 3. If the old struct is comparable, so is the new one.

The set of selectable exported fields is the set of exported fields `F`
such that `x.F` is a valid selector expression for a value `x` of the struct
type. `F` may be at the top level of the struct, or it may be a field of an
embedded struct.

Two fields are the same if they have the same name and corresponding types.

Other than assignment, there are only four ways to use a struct: write a struct
literal, select a field, use a value of the struct as a map key, or compare two
values for equality. The first clause ensures that struct literals compile; the
second, that selections compile; and the third, that equality expressions and
map index expressions compile.

#### Numeric Types

> A new numeric type is compatible with an old one if and only if they are
> both unsigned integers, both signed integers, both floats or both complex
> types, and the new one is at least as large as the old on both 32-bit and
> 64-bit architectures.

Other than in assignments, numeric types appear in arithmetic and comparison
expressions. Since all arithmetic operations but shifts (see below) require that
operand types be identical, and by assumption the old and new types underly
defined types (see "Compatibility, Types and Names," below), there is no way for
client code to write an arithmetic expression that compiles with operands of the
old type but not the new.

Numeric types can also appear in type switches and type assertions. Again, since
the old and new types underly defined types, type switches and type assertions
that compiled using the old defined type will continue to compile with the new
defined type.

Going from an unsigned to a signed integer type is an incompatible change for
the sole reason that only an unsigned type can appear as the right operand of a
shift. If this rule is relaxed, then changes from an unsigned type to a larger
signed type would be compatible. See [this
issue](https://github.com/golang/go/issues/19113).

Only integer types can be used in bitwise and shift operations, and for indexing
slices and arrays. That is why switching from an integer to a floating-point
type--even one that can represent all values of the integer type--is an
incompatible change.


Conversions from floating-point to complex types or vice versa are not permitted
(the predeclared functions real, imag, and complex must be used instead). To
prevent valid floating-point or complex conversions from becoming invalid,
changing a floating-point type to a complex type or vice versa is considered an
incompatible change.

Although conversions between any two integer types are valid, assigning a
constant value to a variable of integer type that is too small to represent the
constant is not permitted. That is why the only compatible changes are to
a new type whose values are a superset of the old. The requirement that the new
set of values must include the old on both 32-bit and 64-bit machines allows
conversions from `int32` to `int` and from `int` to `int64`, but not the other
direction; and similarly for `uint`.

Changing a type to or from `uintptr` is considered an incompatible change. Since
its size is not specified, there is no way to know whether the new type's values
are a superset of the old type's.

## Whole-Package Compatibility

Some changes that are compatible for a single type are not compatible when the
package is considered as a whole. For example, if you remove an unexported
method on a defined type, it may no longer implement an interface of the
package. This can break client code:

```
// old
type T int
func (T) m() {}
type I interface { m() }

// new
type T int // no method m anymore

// client
var i pkg.I = pkg.T{} // fails with new because T lacks m
```

Similarly, adding a method to an interface can cause defined types
in the package to stop implementing it.

The second clause in the definition for package compatibility handles these
cases. To repeat:
> 2. For every exposed type that implements an exposed interface in the old package,
> its corresponding type should implement the corresponding interface in the new package.
Recall that a type is exposed if it is part of the package's API, even if it is
unexported.

Other incompatibilities that involve more than one type in the package can arise
whenever two types with identical underlying types exist in the old or new
package. Here, a change "splits" an identical underlying type into two, breaking
conversions:

```
// old
type B struct { X int }
type C struct { X int }

// new
type B struct { X int }
type C struct { X, Y int }

// client
var b B
_ = C(b) // fails with new: cannot convert B to C
```
Finally, changes that are compatible for the package in which they occur can
break downstream packages. That can happen even if they involve unexported
methods, thanks to embedding.

The definitions given here don't account for these sorts of problems.


## Compatibility, Types and Names 

The above definitions state that the only types that can differ compatibly are
defined types and the types that underly them. Changes to other type literals
are considered incompatible. For instance, it is considered an incompatible
change to add a field to the struct in this variable declaration:

```
var V struct { X int }
```
or this alias definition:
```
type T = struct { X int }
```

We make this choice to keep the definition of compatibility (relatively) simple.
A more precise definition could, for instance, distinguish between

```
func F(struct { X int })
```
where any changes to the struct are incompatible, and

```
func F(struct { X, u int })
```
where adding a field is compatible (since clients cannot write the signature,
and thus cannot assign `F` to a variable of the signature type). The definition
should then also allow other function signature changes that only require
call-site compatibility, like

```
func F(struct { X, u int }, ...int)
```
The result would be a much more complex definition with little benefit, since
the examples in this section rarely arise in practice.
