// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rt is the loGo/s support package: helpers the compiler's
// lowering calls, so lowered files import only logos/rt (plus unsafe
// where a milestone needs it).
package rt

import (
	"cmp"
	"context"
	"errors"
	"reflect"
	"slices"
	"strconv"
)

// Site records one error-propagation site: the file and line where a ?
// error path recorded the error. See TraceErr and ErrorTrace.
type Site struct {
	File string
	Line int
}

// DeferTry joins a deferred call result into an error result. A nil
// result leaves the target alone, a nil target takes the result
// as is, and two non-nil errors join with Join semantics.
func DeferTry(errp *error, e error) {
	if errp == nil {
		panic("logos: DeferTry with nil error target")
	}
	if e == nil {
		return
	}
	if *errp == nil {
		*errp = e
		return
	}
	*errp = Join(*errp, e)
}

// Join combines two errors, matching errors.Join semantics. The lowering
// keeps a nil result alone and calls Join only for two non-nil errors.
func Join(a, b error) error {
	return errors.Join(a, b)
}

// Context aliases context.Context so lowered code spells the group
// context without importing context itself.
type Context = context.Context

// MustNonNil checks a platform pointer stored into a declared non-nil
// location (SPEC T-202): it returns its argument, and panics with
// `logos: nil from Go value stored into non-nil *T at <file>:<line>`
// when the pointer is nil. name is the stored type as reflect.Type.String
// prints it, spelled by the lowering at compile time; an empty name marks
// a type that mentions a type parameter, which is named at run time. A nil
// site reports the position as unknown.
//
//go:noinline
func MustNonNil[P ~*E, E any](p P, name string, site *Site) P {
	if p == nil {
		panic(nilStoreMessage(TypeNameOr[P](name), site))
	}
	return p
}

func nilStoreMessage(name string, site *Site) string {
	at := "unknown"
	if site != nil {
		at = site.File + ":" + strconv.Itoa(site.Line)
	}
	return "logos: nil from Go value stored into non-nil " + name + " at " + at
}

// Option is the lowered form of a loGo/s option `T?` whose payload admits
// nil as a valid value (SPEC T-202, T-104.3): Present tells an absent key
// from a present nil payload, so a present nil slice reads as some with a
// nil payload, distinct from none. Options over a non-nil `*T` lower to
// the bare pointer instead, and an interface-backed union option with no
// nil-capable member lowers to the bare interface; neither uses this type.
type Option[T any] struct {
	Value   T
	Present bool
}

// SliceStableFunc sorts s in place, stably, by cmpFn. Collection chains
// lower sortBy through it so lowered files import only logos/rt.
func SliceStableFunc[S ~[]E, E any](s S, cmpFn func(a, b E) int) {
	slices.SortStableFunc(s, cmpFn)
}

// Compare orders x and y like cmp.Compare: ordered values compare by
// <, with NaN before every non-NaN float. Collection chain sortBy keys
// compare through it (SPEC T-104.6).
func Compare[T cmp.Ordered](x, y T) int {
	return cmp.Compare(x, y)
}

// Lookup reads key k from map m with explicit presence (SPEC T-104.3): an
// absent key gives none, a present zero value gives some, and a present
// nil payload gives some with a nil payload. Single-value map reads whose
// element needs presence lower to this call; non-nil pointer elements and
// interface-backed union elements with no nil-capable member keep the
// plain `m[k]` read with nil as none.
func Lookup[M ~map[K]V, K comparable, V any](m M, k K) Option[V] {
	v, ok := m[k]
	return Option[V]{Value: v, Present: ok}
}

// UnionAssertionError is the panic value of a failed single-value
// assertion to a closed union. When x.(U) fails because the dynamic type
// of x is not a member of U, it panics with
// &UnionAssertionError{Value: x, Union: "<pkg>.U"}. When u.(T) fails
// because T is not a member of U, it panics with the same error plus
// Asserted holding the name of T.
type UnionAssertionError struct {
	Value    any
	Union    string
	Asserted string
}

// Error formats the Go assertion message with the union in place of the
// asserted type: `interface conversion: <T> is not a member of <pkg>.U`,
// where <T> is Asserted when set, else the dynamic type of Value, or
// `nil` when Value is nil.
func (e *UnionAssertionError) Error() string {
	actual := "nil"
	if e.Asserted != "" {
		actual = e.Asserted
	} else if e.Value != nil {
		actual = reflect.TypeOf(e.Value).String()
	}
	return "interface conversion: " + actual + " is not a member of " + e.Union
}

// TypeNameOr returns name when the lowering spelled the name of T at
// compile time, and the name of T as Go prints it in an interface
// conversion panic otherwise: qualified by the package name, never by the
// spelling or the import alias used where T was written. The lowering
// passes an empty name only for a T that mentions a type parameter.
func TypeNameOr[T any](name string) string {
	if name != "" {
		return name
	}
	return reflect.TypeFor[T]().String()
}

func AppendInt(b []byte, x int64, base int) []byte {
	return strconv.AppendInt(b, x, base)
}

func AppendUint(b []byte, x uint64, base int) []byte {
	return strconv.AppendUint(b, x, base)
}

func AppendFloat(b []byte, f float64, fmt byte, prec, bitSize int) []byte {
	return strconv.AppendFloat(b, f, fmt, prec, bitSize)
}

func AppendBool(b []byte, x bool) []byte {
	return strconv.AppendBool(b, x)
}

type PanicFormatter func(...any) string

type stringer interface{ String() string }

// InterpError renders an interface operand of an interpolation that
// implements error: a nil interface gives `<nil>`, and a panic in Error
// gives `<nil>` for a nil pointer receiver and `%!v(PANIC=Error method:
// ...)` otherwise. The receiver is known only at run time.
func InterpError(v error, format PanicFormatter) (s string) {
	if v == nil {
		return "<nil>"
	}
	defer func() {
		if r := recover(); r != nil {
			if nilPointer(v) {
				s = "<nil>"
				return
			}
			s = methodPanicText("Error", r, format)
		}
	}()
	return v.Error()
}

// InterpErrorValue renders an operand whose static type is a concrete
// non-pointer type that implements error.
func InterpErrorValue[T error](v T, format PanicFormatter) (s string) {
	defer func() {
		if r := recover(); r != nil {
			s = methodPanicText("Error", r, format)
		}
	}()
	return v.Error()
}

// InterpErrorPtr renders an operand whose static type is a pointer type
// that implements error; a panic with a nil receiver gives `<nil>`.
func InterpErrorPtr[E any, P interface {
	*E
	error
}](v P, format PanicFormatter) (s string) {
	defer func() {
		if r := recover(); r != nil {
			if v == nil {
				s = "<nil>"
				return
			}
			s = methodPanicText("Error", r, format)
		}
	}()
	return v.Error()
}

// InterpStringer renders an interface operand of an interpolation that
// implements fmt.Stringer, like InterpError.
func InterpStringer(v stringer, format PanicFormatter) (s string) {
	if v == nil {
		return "<nil>"
	}
	defer func() {
		if r := recover(); r != nil {
			if nilPointer(v) {
				s = "<nil>"
				return
			}
			s = methodPanicText("String", r, format)
		}
	}()
	return v.String()
}

// InterpStringerValue renders an operand whose static type is a concrete
// non-pointer type that implements fmt.Stringer.
func InterpStringerValue[T stringer](v T, format PanicFormatter) (s string) {
	defer func() {
		if r := recover(); r != nil {
			s = methodPanicText("String", r, format)
		}
	}()
	return v.String()
}

// InterpStringerPtr renders an operand whose static type is a pointer
// type that implements fmt.Stringer; a panic with a nil receiver gives
// `<nil>`.
func InterpStringerPtr[E any, P interface {
	*E
	stringer
}](v P, format PanicFormatter) (s string) {
	defer func() {
		if r := recover(); r != nil {
			if v == nil {
				s = "<nil>"
				return
			}
			s = methodPanicText("String", r, format)
		}
	}()
	return v.String()
}

func nilPointer(v any) bool {
	rv := reflect.ValueOf(v)
	return rv.Kind() == reflect.Pointer && rv.IsNil()
}

func methodPanicText(method string, panicValue any, format PanicFormatter) string {
	return "%!v(PANIC=" + method + " method: " + format(panicValue) + ")"
}
