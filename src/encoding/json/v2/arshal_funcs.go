// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"errors"
	"fmt"
	"reflect"
	"sync"

	"encoding/json/internal"
	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
	"encoding/json/jsontext"
)

// SkipFunc may be returned by [MarshalToFunc] and [UnmarshalFromFunc] functions.
//
// Any function that returns SkipFunc must not cause observable side effects
// on the provided [jsontext.Encoder] or [jsontext.Decoder].
// For example, it is permissible to call [jsontext.Decoder.PeekKind],
// but not permissible to call [jsontext.Decoder.ReadToken] or
// [jsontext.Encoder.WriteToken] since such methods mutate the state.
var SkipFunc = errors.New("json: skip function")

var errSkipMutation = errors.New("must not read or write any tokens when skipping")
var errNonSingularValue = errors.New("must read or write exactly one value")

// Marshalers is a list of functions that may override the marshal behavior
// of specific types. Populate [WithMarshalers] to use it with
// [Marshal], [MarshalWrite], or [MarshalEncode].
// A nil *Marshalers is equivalent to an empty list.
// There are no exported fields or methods on Marshalers.
type Marshalers = typedMarshalers

// JoinMarshalers constructs a flattened list of marshal functions.
// If multiple functions in the list are applicable for a value of a given type,
// then those earlier in the list take precedence over those that come later.
// If a function returns [SkipFunc], then the next applicable function is called,
// otherwise the default marshaling behavior is used.
//
// For example:
//
//	m1 := JoinMarshalers(f1, f2)
//	m2 := JoinMarshalers(f0, m1, f3)     // equivalent to m3
//	m3 := JoinMarshalers(f0, f1, f2, f3) // equivalent to m2
func JoinMarshalers(ms ...*Marshalers) *Marshalers {
	return newMarshalers(ms...)
}

// Unmarshalers is a list of functions that may override the unmarshal behavior
// of specific types. Populate [WithUnmarshalers] to use it with
// [Unmarshal], [UnmarshalRead], or [UnmarshalDecode].
// A nil *Unmarshalers is equivalent to an empty list.
// There are no exported fields or methods on Unmarshalers.
type Unmarshalers = typedUnmarshalers

// JoinUnmarshalers constructs a flattened list of unmarshal functions.
// If multiple functions in the list are applicable for a value of a given type,
// then those earlier in the list take precedence over those that come later.
// If a function returns [SkipFunc], then the next applicable function is called,
// otherwise the default unmarshaling behavior is used.
//
// For example:
//
//	u1 := JoinUnmarshalers(f1, f2)
//	u2 := JoinUnmarshalers(f0, u1, f3)     // equivalent to u3
//	u3 := JoinUnmarshalers(f0, f1, f2, f3) // equivalent to u2
func JoinUnmarshalers(us ...*Unmarshalers) *Unmarshalers {
	return newUnmarshalers(us...)
}

type typedMarshalers = typedArshalers[jsontext.Encoder]
type typedUnmarshalers = typedArshalers[jsontext.Decoder]
type typedArshalers[Coder any] struct {
	nonComparable

	fncVals  []typedArshaler[Coder]
	fncCache sync.Map // map[reflect.Type]arshaler

	// fromAny reports whether any of Go types used to represent arbitrary JSON
	// (i.e., any, bool, string, float64, map[string]any, or []any) matches
	// any of the provided type-specific arshalers.
	//
	// This bit of information is needed in arshal_default.go to determine
	// whether to use the specialized logic in arshal_any.go to handle
	// the any interface type. The logic in arshal_any.go does not support
	// type-specific arshal functions, so we must avoid using that logic
	// if this is true.
	fromAny bool
}
type typedMarshaler = typedArshaler[jsontext.Encoder]
type typedUnmarshaler = typedArshaler[jsontext.Decoder]
type typedArshaler[Coder any] struct {
	typ     reflect.Type
	fnc     func(*Coder, addressableValue, *jsonopts.Struct) error
	maySkip bool
}

func newMarshalers(ms ...*Marshalers) *Marshalers       { return newTypedArshalers(ms...) }
func newUnmarshalers(us ...*Unmarshalers) *Unmarshalers { return newTypedArshalers(us...) }
func newTypedArshalers[Coder any](as ...*typedArshalers[Coder]) *typedArshalers[Coder] {
	var a typedArshalers[Coder]
	for _, a2 := range as {
		if a2 != nil {
			a.fncVals = append(a.fncVals, a2.fncVals...)
			a.fromAny = a.fromAny || a2.fromAny
		}
	}
	if len(a.fncVals) == 0 {
		return nil
	}
	return &a
}

func (a *typedArshalers[Coder]) lookup(fnc func(*Coder, addressableValue, *jsonopts.Struct) error, t reflect.Type) (func(*Coder, addressableValue, *jsonopts.Struct) error, bool) {
	if a == nil {
		return fnc, false
	}
	if v, ok := a.fncCache.Load(t); ok {
		if v == nil {
			return fnc, false
		}
		return v.(func(*Coder, addressableValue, *jsonopts.Struct) error), true
	}

	// Collect a list of arshalers that can be called for this type.
	// This list may be longer than 1 since some arshalers can be skipped.
	var fncs []func(*Coder, addressableValue, *jsonopts.Struct) error
	for _, fncVal := range a.fncVals {
		if !castableTo(t, fncVal.typ) {
			continue
		}
		fncs = append(fncs, fncVal.fnc)
		if !fncVal.maySkip {
			break // subsequent arshalers will never be called
		}
	}

	if len(fncs) == 0 {
		a.fncCache.Store(t, nil) // nil to indicate that no funcs found
		return fnc, false
	}

	// Construct an arshaler that may call every applicable arshaler.
	fncDefault := fnc
	fnc = func(c *Coder, v addressableValue, o *jsonopts.Struct) error {
		for _, fnc := range fncs {
			if err := fnc(c, v, o); err != SkipFunc {
				return err // may be nil or non-nil
			}
		}
		return fncDefault(c, v, o)
	}

	// Use the first stored so duplicate work can be garbage collected.
	v, _ := a.fncCache.LoadOrStore(t, fnc)
	return v.(func(*Coder, addressableValue, *jsonopts.Struct) error), true
}

// MarshalFunc constructs a type-specific marshaler that
// specifies how to marshal values of type T.
// T can be any type except a named pointer.
// The function is always provided with a non-nil pointer value
// if T is an interface or pointer type.
//
// The function must marshal exactly one JSON value.
// The value of T must not be retained outside the function call.
// It may not return [SkipFunc].
func MarshalFunc[T any](fn func(T) ([]byte, error)) *Marshalers {
	t := reflect.TypeFor[T]()
	assertCastableTo(t, true)
	typFnc := typedMarshaler{
		typ: t,
		fnc: func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
			val, err := fn(va.castTo(t).Interface().(T))
			if err != nil {
				err = wrapSkipFunc(err, "marshal function of type func(T) ([]byte, error)")
				if mo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
					return internal.NewMarshalerError(va.Addr().Interface(), err, "MarshalFunc") // unlike unmarshal, always wrapped
				}
				err = newMarshalErrorBefore(enc, t, err)
				return collapseSemanticErrors(err)
			}
			if err := enc.WriteValue(val); err != nil {
				if mo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
					return internal.NewMarshalerError(va.Addr().Interface(), err, "MarshalFunc") // unlike unmarshal, always wrapped
				}
				if isSyntacticError(err) {
					err = newMarshalErrorBefore(enc, t, err)
				}
				return err
			}
			return nil
		},
	}
	return &Marshalers{fncVals: []typedMarshaler{typFnc}, fromAny: castableToFromAny(t)}
}

// MarshalToFunc constructs a type-specific marshaler that
// specifies how to marshal values of type T.
// T can be any type except a named pointer.
// The function is always provided with a non-nil pointer value
// if T is an interface or pointer type.
//
// The function must marshal exactly one JSON value by calling write methods
// on the provided encoder. It may return [SkipFunc] such that marshaling can
// move on to the next marshal function. However, no mutable method calls may
// be called on the encoder if [SkipFunc] is returned.
// The pointer to [jsontext.Encoder] and the value of T
// must not be retained outside the function call.
func MarshalToFunc[T any](fn func(*jsontext.Encoder, T) error) *Marshalers {
	t := reflect.TypeFor[T]()
	assertCastableTo(t, true)
	typFnc := typedMarshaler{
		typ: t,
		fnc: func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
			xe := export.Encoder(enc)
			prevDepth, prevLength := xe.Tokens.DepthLength()
			xe.Flags.Set(jsonflags.WithinArshalCall | 1)
			err := fn(enc, va.castTo(t).Interface().(T))
			xe.Flags.Set(jsonflags.WithinArshalCall | 0)
			currDepth, currLength := xe.Tokens.DepthLength()
			if err == nil && (prevDepth != currDepth || prevLength+1 != currLength) {
				err = errNonSingularValue
			}
			if err != nil {
				if err == SkipFunc {
					if prevDepth == currDepth && prevLength == currLength {
						return SkipFunc
					}
					err = errSkipMutation
				}
				if mo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
					return internal.NewMarshalerError(va.Addr().Interface(), err, "MarshalToFunc") // unlike unmarshal, always wrapped
				}
				if !export.IsIOError(err) {
					err = newSemanticErrorWithPosition(enc, t, prevDepth, prevLength, err)
				}
				return err
			}
			return nil
		},
		maySkip: true,
	}
	return &Marshalers{fncVals: []typedMarshaler{typFnc}, fromAny: castableToFromAny(t)}
}

// UnmarshalFunc constructs a type-specific unmarshaler that
// specifies how to unmarshal values of type T.
// T must be an unnamed pointer or an interface type.
// The function is always provided with a non-nil pointer value.
//
// The function must unmarshal exactly one JSON value.
// The input []byte must not be mutated.
// The input []byte and value T must not be retained outside the function call.
// It may not return [SkipFunc].
func UnmarshalFunc[T any](fn func([]byte, T) error) *Unmarshalers {
	t := reflect.TypeFor[T]()
	assertCastableTo(t, false)
	typFnc := typedUnmarshaler{
		typ: t,
		fnc: func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
			val, err := dec.ReadValue()
			if err != nil {
				return err // must be a syntactic or I/O error
			}
			err = fn(val, va.castTo(t).Interface().(T))
			if err != nil {
				err = wrapSkipFunc(err, "unmarshal function of type func([]byte, T) error")
				if uo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
					return err // unlike marshal, never wrapped
				}
				err = newUnmarshalErrorAfter(dec, t, err)
				return collapseSemanticErrors(err)
			}
			return nil
		},
	}
	return &Unmarshalers{fncVals: []typedUnmarshaler{typFnc}, fromAny: castableToFromAny(t)}
}

// UnmarshalFromFunc constructs a type-specific unmarshaler that
// specifies how to unmarshal values of type T.
// T must be an unnamed pointer or an interface type.
// The function is always provided with a non-nil pointer value.
//
// The function must unmarshal exactly one JSON value by calling read methods
// on the provided decoder. It may return [SkipFunc] such that unmarshaling can
// move on to the next unmarshal function. However, no mutable method calls may
// be called on the decoder if [SkipFunc] is returned.
// The pointer to [jsontext.Decoder] and the value of T
// must not be retained outside the function call.
func UnmarshalFromFunc[T any](fn func(*jsontext.Decoder, T) error) *Unmarshalers {
	t := reflect.TypeFor[T]()
	assertCastableTo(t, false)
	typFnc := typedUnmarshaler{
		typ: t,
		fnc: func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
			xd := export.Decoder(dec)
			prevDepth, prevLength := xd.Tokens.DepthLength()
			xd.Flags.Set(jsonflags.WithinArshalCall | 1)
			err := fn(dec, va.castTo(t).Interface().(T))
			xd.Flags.Set(jsonflags.WithinArshalCall | 0)
			currDepth, currLength := xd.Tokens.DepthLength()
			if err == nil && (prevDepth != currDepth || prevLength+1 != currLength) {
				err = errNonSingularValue
			}
			if err != nil {
				if err == SkipFunc {
					if prevDepth == currDepth && prevLength == currLength {
						return SkipFunc
					}
					err = errSkipMutation
				}
				if uo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
					if err2 := xd.SkipUntil(prevDepth, prevLength+1); err2 != nil {
						return err2
					}
					return err // unlike marshal, never wrapped
				}
				if !isSyntacticError(err) && !export.IsIOError(err) {
					err = newSemanticErrorWithPosition(dec, t, prevDepth, prevLength, err)
				}
				return err
			}
			return nil
		},
		maySkip: true,
	}
	return &Unmarshalers{fncVals: []typedUnmarshaler{typFnc}, fromAny: castableToFromAny(t)}
}

// assertCastableTo asserts that "to" is a valid type to be casted to.
// These are the Go types that type-specific arshalers may operate upon.
//
// Let AllTypes be the universal set of all possible Go types.
// This function generally asserts that:
//
//	len([from for from in AllTypes if castableTo(from, to)]) > 0
//
// otherwise it panics.
//
// As a special-case if marshal is false, then we forbid any non-pointer or
// non-interface type since it is almost always a bug trying to unmarshal
// into something where the end-user caller did not pass in an addressable value
// since they will not observe the mutations.
func assertCastableTo(to reflect.Type, marshal bool) {
	switch to.Kind() {
	case reflect.Interface:
		return
	case reflect.Pointer:
		// Only allow unnamed pointers to be consistent with the fact that
		// taking the address of a value produces an unnamed pointer type.
		if to.Name() == "" {
			return
		}
	default:
		// Technically, non-pointer types are permissible for unmarshal.
		// However, they are often a bug since the receiver would be immutable.
		// Thus, only allow them for marshaling.
		if marshal {
			return
		}
	}
	if marshal {
		panic(fmt.Sprintf("input type %v must be an interface type, an unnamed pointer type, or a non-pointer type", to))
	} else {
		panic(fmt.Sprintf("input type %v must be an interface type or an unnamed pointer type", to))
	}
}

// castableTo checks whether values of type "from" can be casted to type "to".
// Nil pointer or interface "from" values are never considered castable.
//
// This function must be kept in sync with addressableValue.castTo.
func castableTo(from, to reflect.Type) bool {
	switch to.Kind() {
	case reflect.Interface:
		// TODO: This breaks when ordinary interfaces can have type sets
		// since interfaces now exist where only the value form of a type (T)
		// implements the interface, but not the pointer variant (*T).
		// See https://go.dev/issue/45346.
		return reflect.PointerTo(from).Implements(to)
	case reflect.Pointer:
		// Common case for unmarshaling.
		// From must be a concrete or interface type.
		return reflect.PointerTo(from) == to
	default:
		// Common case for marshaling.
		// From must be a concrete type.
		return from == to
	}
}

// castTo casts va to the specified type.
// If the type is an interface, then the underlying type will always
// be a non-nil pointer to a concrete type.
//
// Requirement: castableTo(va.Type(), to) must hold.
func (va addressableValue) castTo(to reflect.Type) reflect.Value {
	switch to.Kind() {
	case reflect.Interface:
		return va.Addr().Convert(to)
	case reflect.Pointer:
		return va.Addr()
	default:
		return va.Value
	}
}

// castableToFromAny reports whether "to" can be casted to from any
// of the dynamic types used to represent arbitrary JSON.
func castableToFromAny(to reflect.Type) bool {
	for _, from := range []reflect.Type{anyType, boolType, stringType, float64Type, mapStringAnyType, sliceAnyType} {
		if castableTo(from, to) {
			return true
		}
	}
	return false
}

func wrapSkipFunc(err error, what string) error {
	if err == SkipFunc {
		return errors.New(what + " cannot be skipped")
	}
	return err
}
