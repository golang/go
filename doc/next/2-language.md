## Changes to the language {#language}

<!-- go.dev/issue/9859 -->

A key in a [struct literal](/ref/spec#Composite_literals) may be any
valid [field selector](/issue/9859) for the struct type, not just a
(top-level) field name of the struct.

<!-- go.dev/issue/77273 -->

Go 1.27 now supports [generic methods](/issue/77273):
a [method declaration](/ref/spec#Method_declarations) may declare its own
[type parameters](/ref/spec#Type_parameter_declarations).
This widely anticipated change allows adding generic functions within
the namespace of a particular data type where before one had to declare
such functions with a scope of the entire package.
Note that methods of [interfaces](/ref/spec#Interface_types) may not declare
type parameters nor can interface types be implemented by generic methods.

<!-- go.dev/issue/77245 -->

Function type inference has been [generalized](/issue/77245) to apply in all
contexts where a generic function is [assigned](/ref/spec#Assignability) to a
variable of (or converted to) a matching function type.
