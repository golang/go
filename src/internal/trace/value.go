// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"unsafe"
)

// Value is a dynamically-typed value obtained from a trace.
type Value struct {
	kind    ValueKind
	pointer unsafe.Pointer
	scalar  uint64
}

// ValueKind is the type of a dynamically-typed value from a trace.
type ValueKind uint8

const (
	ValueBad ValueKind = iota
	ValueUint64
	ValueString
)

// Kind returns the ValueKind of the value.
//
// It represents the underlying structure of the value.
//
// New ValueKinds may be added in the future. Users of this type must be robust
// to that possibility.
func (v Value) Kind() ValueKind {
	return v.kind
}

// ToUint64 returns the uint64 value for a ValueUint64.
//
// Panics if this Value's Kind is not ValueUint64.
func (v Value) ToUint64() uint64 {
	if v.kind != ValueUint64 {
		panic("ToUint64 called on Value of a different Kind")
	}
	return v.scalar
}

// ToString returns the uint64 value for a ValueString.
//
// Panics if this Value's Kind is not ValueString.
func (v Value) ToString() string {
	if v.kind != ValueString {
		panic("ToString called on Value of a different Kind")
	}
	return unsafe.String((*byte)(v.pointer), int(v.scalar))
}

func uint64Value(x uint64) Value {
	return Value{kind: ValueUint64, scalar: x}
}

func stringValue(s string) Value {
	return Value{kind: ValueString, scalar: uint64(len(s)), pointer: unsafe.Pointer(unsafe.StringData(s))}
}

// String returns the string representation of the value.
func (v Value) String() string {
	switch v.Kind() {
	case ValueUint64:
		return fmt.Sprintf("Value{Uint64(%d)}", v.ToUint64())
	case ValueString:
		return fmt.Sprintf("Value{String(%s)}", v.ToString())
	}
	return "Value{Bad}"
}
