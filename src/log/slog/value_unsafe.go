// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"reflect"
	"unsafe"
)

// A Value can represent any Go value, but unlike type any,
// it can represent most small values without an allocation.
// The zero Value corresponds to nil.
type Value struct {
	// num holds the value for Kinds Int64, Uint64, Float64, Bool and Duration,
	// the string length for KindString, and nanoseconds since the epoch for KindTime.
	num uint64
	// If any is of type Kind, then the value is in num as described above.
	// If any is of type *time.Location, then the Kind is Time and time.Time value
	// can be constructed from the Unix nanos in num and the location (monotonic time
	// is not preserved).
	// If any is of type stringptr, then the Kind is String and the string value
	// consists of the length in num and the pointer in any.
	// Otherwise, the Kind is Any and any is the value.
	// (This implies that Attrs cannot store values of type Kind, *time.Location
	// or stringptr.)
	any any
}

type (
	stringptr unsafe.Pointer // used in Value.any when the Value is a string
	groupptr  unsafe.Pointer // used in Value.any when the Value is a []Attr
)

// Kind returns v's Kind.
func (v Value) Kind() Kind {
	switch x := v.any.(type) {
	case Kind:
		return x
	case stringptr:
		return KindString
	case timeLocation:
		return KindTime
	case groupptr:
		return KindGroup
	case LogValuer:
		return KindLogValuer
	case kind: // a kind is just a wrapper for a Kind
		return KindAny
	default:
		return KindAny
	}
}

// StringValue returns a new Value for a string.
func StringValue(value string) Value {
	hdr := (*reflect.StringHeader)(unsafe.Pointer(&value))
	return Value{num: uint64(hdr.Len), any: stringptr(hdr.Data)}
}

func (v Value) str() string {
	var s string
	hdr := (*reflect.StringHeader)(unsafe.Pointer(&s))
	hdr.Data = uintptr(v.any.(stringptr))
	hdr.Len = int(v.num)
	return s
}

// String returns Value's value as a string, formatted like fmt.Sprint. Unlike
// the methods Int64, Float64, and so on, which panic if v is of the
// wrong kind, String never panics.
func (v Value) String() string {
	if sp, ok := v.any.(stringptr); ok {
		// Inlining this code makes a huge difference.
		var s string
		hdr := (*reflect.StringHeader)(unsafe.Pointer(&s))
		hdr.Data = uintptr(sp)
		hdr.Len = int(v.num)
		return s
	}
	var buf []byte
	return string(v.append(buf))
}

func groupValue(as []Attr) Value {
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&as))
	return Value{num: uint64(hdr.Len), any: groupptr(hdr.Data)}
}

// group returns the Value's value as a []Attr.
// It panics if the Value's Kind is not KindGroup.
func (v Value) group() []Attr {
	if sp, ok := v.any.(groupptr); ok {
		return unsafe.Slice((*Attr)(sp), v.num)
	}
	panic("Group: bad kind")
}

func (v Value) uncheckedGroup() []Attr {
	return unsafe.Slice((*Attr)(v.any.(groupptr)), v.num)
}
