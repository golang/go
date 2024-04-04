### New unique package

The new [unique](/pkg/unique) package provides facilites for
canonicalizing values (like "interning" or "hash-consing").

Any value of comparable type may be canonicalized with the new
`Make[T]` function, which produces a reference to a canonical copy of
the value in the form of a `Handle[T]`.
Two `Handle[T]` are equal if and only if the values used to produce the
handles are equal, allowing programs to deduplicate values and reduce
their memory footprint.
Comparing two `Handle[T]` values is efficient, reducing down to a simple
pointer comparison.
