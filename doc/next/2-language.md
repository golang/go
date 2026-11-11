## Changes to the language {#language}

<!-- go.dev/issue/12854 -->

Go 1.28 allows [omitting the type](/ref/spec#Omitting_composite_literal_types)
of a [composite literal](/ref/spec#Composite_literals) whenever its type can be
[inferred](/issue/12854) from [assignment context](/ref/spec#Assignability).
Previously, a composite literal could omit its type only when appearing within
an enclosing array, slice, or map literal.

```go
type Point struct{ X, Y float64 }

var origin Point  = {}      // Point{}
var ptr    *Point = {1, 2}  // &Point{1, 2}
```

<!-- go.dev/issue/3939 -->

Starting with Go 1.28, the [conversion](/ref/spec#Conversions_to_and_from_a_string_type)
of an integer value `x` to a string type is valid only if `x`'s underlying type is `byte`
or `rune`, or if `x` is an untyped rune constant.
The ability to convert any integer value to a string type has been removed.
This new restriction lifts a long-existing `vet` check to the language level.
See [issue #3939](/issue/3939) for a discussion of this change.

```go
string('a')     // "a"
string(97)      // invalid: 97 is not a rune constant
```
