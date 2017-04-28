// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"cmd/internal/obj/x86"
	"cmd/internal/src"
	"fmt"
	"testing"
)

var CheckFunc = checkFunc
var Opt = opt
var Deadcode = deadcode
var Copyelim = copyelim

var testCtxts = map[string]*obj.Link{
	"amd64": obj.Linknew(&x86.Linkamd64),
	"s390x": obj.Linknew(&s390x.Links390x),
}

func testConfig(tb testing.TB) *Conf      { return testConfigArch(tb, "amd64") }
func testConfigS390X(tb testing.TB) *Conf { return testConfigArch(tb, "s390x") }

func testConfigArch(tb testing.TB, arch string) *Conf {
	ctxt, ok := testCtxts[arch]
	if !ok {
		tb.Fatalf("unknown arch %s", arch)
	}
	if ctxt.Arch.PtrSize != 8 {
		tb.Fatal("dummyTypes is 64-bit only")
	}
	c := &Conf{
		config: NewConfig(arch, dummyTypes, ctxt, true),
		tb:     tb,
	}
	return c
}

type Conf struct {
	config *Config
	tb     testing.TB
	fe     Frontend
}

func (c *Conf) Frontend() Frontend {
	if c.fe == nil {
		c.fe = DummyFrontend{t: c.tb, ctxt: c.config.ctxt}
	}
	return c.fe
}

// DummyFrontend is a test-only frontend.
// It assumes 64 bit integers and pointers.
type DummyFrontend struct {
	t    testing.TB
	ctxt *obj.Link
}

type DummyAuto struct {
	t *types.Type
	s string
}

func (d *DummyAuto) Typ() *types.Type {
	return d.t
}

func (d *DummyAuto) String() string {
	return d.s
}

func (DummyFrontend) StringData(s string) interface{} {
	return nil
}
func (DummyFrontend) Auto(pos src.XPos, t *types.Type) GCNode {
	return &DummyAuto{t: t, s: "aDummyAuto"}
}
func (d DummyFrontend) SplitString(s LocalSlot) (LocalSlot, LocalSlot) {
	return LocalSlot{s.N, dummyTypes.BytePtr, s.Off}, LocalSlot{s.N, dummyTypes.Int, s.Off + 8}
}
func (d DummyFrontend) SplitInterface(s LocalSlot) (LocalSlot, LocalSlot) {
	return LocalSlot{s.N, dummyTypes.BytePtr, s.Off}, LocalSlot{s.N, dummyTypes.BytePtr, s.Off + 8}
}
func (d DummyFrontend) SplitSlice(s LocalSlot) (LocalSlot, LocalSlot, LocalSlot) {
	return LocalSlot{s.N, s.Type.ElemType().PtrTo(), s.Off},
		LocalSlot{s.N, dummyTypes.Int, s.Off + 8},
		LocalSlot{s.N, dummyTypes.Int, s.Off + 16}
}
func (d DummyFrontend) SplitComplex(s LocalSlot) (LocalSlot, LocalSlot) {
	if s.Type.Size() == 16 {
		return LocalSlot{s.N, dummyTypes.Float64, s.Off}, LocalSlot{s.N, dummyTypes.Float64, s.Off + 8}
	}
	return LocalSlot{s.N, dummyTypes.Float32, s.Off}, LocalSlot{s.N, dummyTypes.Float32, s.Off + 4}
}
func (d DummyFrontend) SplitInt64(s LocalSlot) (LocalSlot, LocalSlot) {
	if s.Type.IsSigned() {
		return LocalSlot{s.N, dummyTypes.Int32, s.Off + 4}, LocalSlot{s.N, dummyTypes.UInt32, s.Off}
	}
	return LocalSlot{s.N, dummyTypes.UInt32, s.Off + 4}, LocalSlot{s.N, dummyTypes.UInt32, s.Off}
}
func (d DummyFrontend) SplitStruct(s LocalSlot, i int) LocalSlot {
	return LocalSlot{s.N, s.Type.FieldType(i), s.Off + s.Type.FieldOff(i)}
}
func (d DummyFrontend) SplitArray(s LocalSlot) LocalSlot {
	return LocalSlot{s.N, s.Type.ElemType(), s.Off}
}
func (DummyFrontend) Line(_ src.XPos) string {
	return "unknown.go:0"
}
func (DummyFrontend) AllocFrame(f *Func) {
}
func (d DummyFrontend) Syslook(s string) *obj.LSym {
	return d.ctxt.Lookup(s)
}
func (DummyFrontend) UseWriteBarrier() bool {
	return true // only writebarrier_test cares
}

func (d DummyFrontend) Logf(msg string, args ...interface{}) { d.t.Logf(msg, args...) }
func (d DummyFrontend) Log() bool                            { return true }

func (d DummyFrontend) Fatalf(_ src.XPos, msg string, args ...interface{}) { d.t.Fatalf(msg, args...) }
func (d DummyFrontend) Warnl(_ src.XPos, msg string, args ...interface{})  { d.t.Logf(msg, args...) }
func (d DummyFrontend) Debug_checknil() bool                               { return false }
func (d DummyFrontend) Debug_wb() bool                                     { return false }

var dummyTypes Types

func init() {
	// Initialize just enough of the universe and the types package to make our tests function.
	// TODO(josharian): move universe initialization to the types package,
	// so this test setup can share it.

	types.Tconv = func(t *types.Type, flag, mode, depth int) string {
		return t.Etype.String()
	}
	types.Sconv = func(s *types.Sym, flag, mode int) string {
		return "sym"
	}
	types.FormatSym = func(sym *types.Sym, s fmt.State, verb rune, mode int) {
		fmt.Fprintf(s, "sym")
	}
	types.FormatType = func(t *types.Type, s fmt.State, verb rune, mode int) {
		fmt.Fprintf(s, "%v", t.Etype)
	}
	types.Dowidth = func(t *types.Type) {}

	types.Tptr = types.TPTR64
	for _, typ := range [...]struct {
		width int64
		et    types.EType
	}{
		{1, types.TINT8},
		{1, types.TUINT8},
		{1, types.TBOOL},
		{2, types.TINT16},
		{2, types.TUINT16},
		{4, types.TINT32},
		{4, types.TUINT32},
		{4, types.TFLOAT32},
		{4, types.TFLOAT64},
		{8, types.TUINT64},
		{8, types.TINT64},
		{8, types.TINT},
		{8, types.TUINTPTR},
	} {
		t := types.New(typ.et)
		t.Width = typ.width
		t.Align = uint8(typ.width)
		types.Types[typ.et] = t
	}

	dummyTypes = Types{
		Bool:       types.Types[types.TBOOL],
		Int8:       types.Types[types.TINT8],
		Int16:      types.Types[types.TINT16],
		Int32:      types.Types[types.TINT32],
		Int64:      types.Types[types.TINT64],
		UInt8:      types.Types[types.TUINT8],
		UInt16:     types.Types[types.TUINT16],
		UInt32:     types.Types[types.TUINT32],
		UInt64:     types.Types[types.TUINT64],
		Float32:    types.Types[types.TFLOAT32],
		Float64:    types.Types[types.TFLOAT64],
		Int:        types.Types[types.TINT],
		Uintptr:    types.Types[types.TUINTPTR],
		String:     types.Types[types.TSTRING],
		BytePtr:    types.NewPtr(types.Types[types.TUINT8]),
		Int32Ptr:   types.NewPtr(types.Types[types.TINT32]),
		UInt32Ptr:  types.NewPtr(types.Types[types.TUINT32]),
		IntPtr:     types.NewPtr(types.Types[types.TINT]),
		UintptrPtr: types.NewPtr(types.Types[types.TUINTPTR]),
		Float32Ptr: types.NewPtr(types.Types[types.TFLOAT32]),
		Float64Ptr: types.NewPtr(types.Types[types.TFLOAT64]),
		BytePtrPtr: types.NewPtr(types.NewPtr(types.Types[types.TUINT8])),
	}
}

func (d DummyFrontend) DerefItab(sym *obj.LSym, off int64) *obj.LSym { return nil }

func (d DummyFrontend) CanSSA(t *types.Type) bool {
	// There are no un-SSAable types in dummy land.
	return true
}
