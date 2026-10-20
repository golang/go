# Lowering of M1 forms

Each block shows `.lgo` input, then the ejected `.go` output. Compiler-made names (`_logos_e0`, `_logos_site0`) vary by position.

## `v := f()?`

Input (enclosing function returns `(int, error)`):

```go
v := parse(s)?
```

Output:

```go
v, _logos_e0 := parse(s)
if _logos_e0 != nil {
	_logos_rt.TraceErr(_logos_e0, &_logos_site0)
	return 0, _logos_e0
}
```

The zero value matches the result type: `0`, `""`, `false`, `nil`, `T{}`, or `*new(T)` for a type parameter.

## `f()?` as a statement

Input:

```go
f()?
```

Output:

```go
if _logos_e0 := f(); _logos_e0 != nil {
	_logos_rt.TraceErr(_logos_e0, &_logos_site0)
	return _logos_e0
}
```

## `defer f()?`

Input:

```go
defer closeAll()?
```

Output:

```go
_logos_f0 := closeAll
defer func() {
	if _logos_e := _logos_f0(); _logos_e != nil {
		if _logos_err == nil {
			_logos_err = _logos_e
		} else {
			_logos_err = _logos_rt.Join(_logos_err, _logos_e)
		}
	}
}()
```

The function value and arguments are fixed where written. The function gets named results when it lacks them.

## `errdefer f()` and `errdefer f()?`

Input:

```go
errdefer tx.Rollback()?
```

Output (same capture as `defer f()?`, body guarded):

```go
_logos_f0 := tx.Rollback
defer func() {
	if !_logos_ok || _logos_err != nil {
		if _logos_e := _logos_f0(); _logos_e != nil {
			if _logos_err == nil {
				_logos_err = _logos_e
			} else {
				_logos_err = _logos_rt.Join(_logos_err, _logos_e)
			}
		}
	}
}()
```

`_logos_ok` starts false and turns true only on a clean `return`. It stays false on panic, so the cleanup still runs.

## Closed unions (M2)

A value-used union declaration ejects to its methods-only interface (terms dropped, methods kept) with a `// loGo/s union: A | B` comment naming the members:

```go
// input
type U interface {
	A | B
	M() string
}

// output: methods-only interface plus the member-list comment
```

`x.(U)` lowers to a membership check followed by a plain assertion. Comma-ok form yields `ok == false` for non-members; single-value form panics with `interface conversion: float64 is not a member of main.U`. A `case U:` clause expands to one case per member with `v := v.(U)` first in the clause body.

A type switch expression lowers to an immediately invoked function returning the chosen arm. When no arm matches, it panics with `logos: switch expression at <file>:<line> matched no arm`.

`type S struct : I` emits the struct plus one `var _ I = (*S)(nil)` assertion per listed interface.

## Options and nil flow (M3)

`*T?` stays the bare pointer with nil as none. Every other option lowers to `logos/rt.Option[T]` (a value plus a present flag), so a present nil payload stays distinct from absent. `a ?? b` evaluates `a` once and `b` only when `a` is none. `a?.f` evaluates `a` once and yields none when `a` is none.

A single-value map read `m[k]` lowers to a presence-checked lookup returning the element option, except for non-nil pointer elements, which keep the plain `m[k]` read with nil as none. `v, ok := m[k]`, stores, `++`, compound assignment, and `delete` lower unchanged.

Values from Go declarations are wrapped at each store into a declared non-nil location with a nil check that panics `logos: nil from Go value stored into non-nil *T at <file>:<line>`.

## Interface properties (M4)

Each property lowers to accessor methods on the declaring type: `Logos_GetF() T` on a value receiver for reads, `Logos_SetF(T)` on a pointer receiver for read-write properties. Reads and writes through the interface call those accessors. A compound update pins the receiver once:

```go
// input
c.HP -= 10

// output shape
_logos_r0 := c
_logos_r0.Logos_SetHP(_logos_r0.Logos_GetHP() - 10)
```

A method call on a struct-typed property copies out, calls by address, and writes back. Ejected output keeps the generated methods, so Go importers call `Logos_GetF` and `Logos_SetF` directly.

## Const views (M5)

Const views are erased: every `const T` type lowers to its plain `T`. No runtime check remains for reads, ranges, or const-receiver calls.

Zero-copy conversions lower to unsafe slice/string re-expression without allocation, only for string and const-composite provenance. A `const []byte` view of a mutable slice converts with a copy. `rt.Freeze` lowers to a plain deep-copy call returning the same pointer type.

## Error traces

`_logos_rt.TraceErr` records the call site for the current goroutine. `logos/rt.ErrorTrace()` returns the sites in propagation order (at most the last 8). A new error value starts a new trace. In ejected code tracing is a no-op.
