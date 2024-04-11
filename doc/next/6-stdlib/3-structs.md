### New structs package


The new [structs](/pkg/structs) package provides
types for struct fields that modify properties of
the containing struct type such as memory layout.

In this release, the only such type is
[`HostLayout`](/pkg/structs#HostLayout)
which indicates that a structure with a field of that
type has a layout that conforms to host platform
expectations.