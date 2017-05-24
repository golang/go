// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"cmd/internal/obj/x86"
	"testing"
)

var CheckFunc = checkFunc
var PrintFunc = printFunc
var Opt = opt
var Deadcode = deadcode
var Copyelim = copyelim

func testConfig(t testing.TB) *Config {
	testCtxt := &obj.Link{Arch: &x86.Linkamd64}
	return NewConfig("amd64", DummyFrontend{t}, testCtxt, true)
}

func testConfigS390X(t testing.TB) *Config {
	return NewConfig("s390x", DummyFrontend{t}, obj.Linknew(&s390x.Links390x), true)
}

// DummyFrontend is a test-only frontend.
// It assumes 64 bit integers and pointers.
type DummyFrontend struct {
	t testing.TB
}

func (DummyFrontend) StringData(s string) interface{} {
	return nil
}

type dummyGCNode struct {
	typ  Type
	name string
}

func (d *dummyGCNode) Typ() Type {
	return d.typ
}
func (d *dummyGCNode) String() string {
	return d.name
}
func (d DummyFrontend) Auto(t Type) GCNode {
	return &dummyGCNode{typ: t, name: "dummy"}
}
func (d DummyFrontend) SplitString(s LocalSlot) (LocalSlot, LocalSlot) {
	return LocalSlot{s.N, d.TypeBytePtr(), s.Off}, LocalSlot{s.N, d.TypeInt(), s.Off + 8}
}
func (d DummyFrontend) SplitInterface(s LocalSlot) (LocalSlot, LocalSlot) {
	return LocalSlot{s.N, d.TypeBytePtr(), s.Off}, LocalSlot{s.N, d.TypeBytePtr(), s.Off + 8}
}
func (d DummyFrontend) SplitSlice(s LocalSlot) (LocalSlot, LocalSlot, LocalSlot) {
	return LocalSlot{s.N, s.Type.ElemType().PtrTo(), s.Off},
		LocalSlot{s.N, d.TypeInt(), s.Off + 8},
		LocalSlot{s.N, d.TypeInt(), s.Off + 16}
}
func (d DummyFrontend) SplitComplex(s LocalSlot) (LocalSlot, LocalSlot) {
	if s.Type.Size() == 16 {
		return LocalSlot{s.N, d.TypeFloat64(), s.Off}, LocalSlot{s.N, d.TypeFloat64(), s.Off + 8}
	}
	return LocalSlot{s.N, d.TypeFloat32(), s.Off}, LocalSlot{s.N, d.TypeFloat32(), s.Off + 4}
}
func (d DummyFrontend) SplitInt64(s LocalSlot) (LocalSlot, LocalSlot) {
	if s.Type.IsSigned() {
		return LocalSlot{s.N, d.TypeInt32(), s.Off + 4}, LocalSlot{s.N, d.TypeUInt32(), s.Off}
	}
	return LocalSlot{s.N, d.TypeUInt32(), s.Off + 4}, LocalSlot{s.N, d.TypeUInt32(), s.Off}
}
func (d DummyFrontend) SplitStruct(s LocalSlot, i int) LocalSlot {
	return LocalSlot{s.N, s.Type.FieldType(i), s.Off + s.Type.FieldOff(i)}
}
func (d DummyFrontend) SplitArray(s LocalSlot) LocalSlot {
	return LocalSlot{s.N, s.Type.ElemType(), s.Off}
}
func (DummyFrontend) Line(line int32) string {
	return "unknown.go:0"
}
func (DummyFrontend) AllocFrame(f *Func) {
}
func (DummyFrontend) Syslook(s string) interface{} {
	return DummySym(s)
}

func (d DummyFrontend) Logf(msg string, args ...interface{}) { d.t.Logf(msg, args...) }
func (d DummyFrontend) Log() bool                            { return true }

func (d DummyFrontend) Fatalf(line int32, msg string, args ...interface{}) { d.t.Fatalf(msg, args...) }
func (d DummyFrontend) Warnl(line int32, msg string, args ...interface{})  { d.t.Logf(msg, args...) }
func (d DummyFrontend) Debug_checknil() bool                               { return false }
func (d DummyFrontend) Debug_wb() bool                                     { return false }

func (d DummyFrontend) TypeBool() Type    { return TypeBool }
func (d DummyFrontend) TypeInt8() Type    { return TypeInt8 }
func (d DummyFrontend) TypeInt16() Type   { return TypeInt16 }
func (d DummyFrontend) TypeInt32() Type   { return TypeInt32 }
func (d DummyFrontend) TypeInt64() Type   { return TypeInt64 }
func (d DummyFrontend) TypeUInt8() Type   { return TypeUInt8 }
func (d DummyFrontend) TypeUInt16() Type  { return TypeUInt16 }
func (d DummyFrontend) TypeUInt32() Type  { return TypeUInt32 }
func (d DummyFrontend) TypeUInt64() Type  { return TypeUInt64 }
func (d DummyFrontend) TypeFloat32() Type { return TypeFloat32 }
func (d DummyFrontend) TypeFloat64() Type { return TypeFloat64 }
func (d DummyFrontend) TypeInt() Type     { return TypeInt64 }
func (d DummyFrontend) TypeUintptr() Type { return TypeUInt64 }
func (d DummyFrontend) TypeString() Type  { panic("unimplemented") }
func (d DummyFrontend) TypeBytePtr() Type { return TypeBytePtr }

func (d DummyFrontend) CanSSA(t Type) bool {
	// There are no un-SSAable types in dummy land.
	return true
}

type DummySym string

func (s DummySym) String() string { return string(s) }
