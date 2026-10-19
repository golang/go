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

## Error traces

`_logos_rt.TraceErr` records the call site for the current goroutine. `logos/rt.ErrorTrace()` returns the sites in propagation order (at most the last 8). A new error value starts a new trace. In ejected code tracing is a no-op.
