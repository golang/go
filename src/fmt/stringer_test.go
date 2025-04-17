// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	. "fmt"
	"testing"
)

type TI int
type TI8 int8
type TI16 int16
type TI32 int32
type TI64 int64
type TU uint
type TU8 uint8
type TU16 uint16
type TU32 uint32
type TU64 uint64
type TUI uintptr
type TF float64
type TF32 float32
type TF64 float64
type TB bool
type TS string

func (v TI) String() string   { return Sprintf("I: %d", int(v)) }
func (v TI8) String() string  { return Sprintf("I8: %d", int8(v)) }
func (v TI16) String() string { return Sprintf("I16: %d", int16(v)) }
func (v TI32) String() string { return Sprintf("I32: %d", int32(v)) }
func (v TI64) String() string { return Sprintf("I64: %d", int64(v)) }
func (v TU) String() string   { return Sprintf("U: %d", uint(v)) }
func (v TU8) String() string  { return Sprintf("U8: %d", uint8(v)) }
func (v TU16) String() string { return Sprintf("U16: %d", uint16(v)) }
func (v TU32) String() string { return Sprintf("U32: %d", uint32(v)) }
func (v TU64) String() string { return Sprintf("U64: %d", uint64(v)) }
func (v TUI) String() string  { return Sprintf("UI: %d", uintptr(v)) }
func (v TF) String() string   { return Sprintf("F: %f", float64(v)) }
func (v TF32) String() string { return Sprintf("F32: %f", float32(v)) }
func (v TF64) String() string { return Sprintf("F64: %f", float64(v)) }
func (v TB) String() string   { return Sprintf("B: %t", bool(v)) }
func (v TS) String() string   { return Sprintf("S: %q", string(v)) }

func check(t *testing.T, got, want string) {
	if got != want {
		t.Error(got, "!=", want)
	}
}

func TestStringer(t *testing.T) {
	s := Sprintf("%v %v %v %v %v", TI(0), TI8(1), TI16(2), TI32(3), TI64(4))
	check(t, s, "I: 0 I8: 1 I16: 2 I32: 3 I64: 4")
	s = Sprintf("%v %v %v %v %v %v", TU(5), TU8(6), TU16(7), TU32(8), TU64(9), TUI(10))
	check(t, s, "U: 5 U8: 6 U16: 7 U32: 8 U64: 9 UI: 10")
	s = Sprintf("%v %v %v", TF(1.0), TF32(2.0), TF64(3.0))
	check(t, s, "F: 1.000000 F32: 2.000000 F64: 3.000000")
	s = Sprintf("%v %v", TB(true), TS("x"))
	check(t, s, "B: true S: \"x\"")
}
