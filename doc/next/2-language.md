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
