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
// when the value is nil. A nil site reports the position as unknown.
// Only pointer-shaped values can be nil; every other value passes
// through untouched.
func MustNonNil[T any](p T, site *Site) T {
	v := reflect.ValueOf(p)
	switch v.Kind() {
	case reflect.Ptr, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan, reflect.Interface, reflect.UnsafePointer:
		if v.IsNil() {
			at := "unknown"
			if site != nil {
				at = site.File + ":" + strconv.Itoa(site.Line)
			}
			panic("logos: nil from Go value stored into non-nil " + reflect.TypeOf(p).String() + " at " + at)
		}
	}
	return p
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
// Asserted holding TypeName[T]().
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

// TypeName returns the name of T as Go prints it in an interface
// conversion panic: qualified by the package name, never by the
// spelling or the import alias used where T was written.
func TypeName[T any]() string {
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

func InterpError(v error, format PanicFormatter) (s string) {
	if v == nil {
		return "<nil>"
	}
	return InterpErrorValue(v, format)
}

func InterpErrorValue[T error](v T, format PanicFormatter) (s string) {
	defer func() {
		if r := recover(); r != nil {
			s = methodPanicText(v, "Error", r, format)
		}
	}()
	return v.Error()
}

func InterpStringer(v stringer, format PanicFormatter) (s string) {
	if v == nil {
		return "<nil>"
	}
	return InterpStringerValue(v, format)
}

func InterpStringerValue[T stringer](v T, format PanicFormatter) (s string) {
	defer func() {
		if r := recover(); r != nil {
			s = methodPanicText(v, "String", r, format)
		}
	}()
	return v.String()
}

func methodPanicText(receiver any, method string, panicValue any, format PanicFormatter) string {
	if rv := reflect.ValueOf(receiver); rv.Kind() == reflect.Pointer && rv.IsNil() {
		return "<nil>"
	}
	return "%!v(PANIC=" + method + " method: " + format(panicValue) + ")"
}
