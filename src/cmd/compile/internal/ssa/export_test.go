// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"cmd/internal/obj/s390x"
	"cmd/internal/obj/x86"
	"cmd/internal/src"
	"testing"
)

var CheckFunc = checkFunc
var Opt = opt
var Deadcode = deadcode
var Copyelim = copyelim

var testCtxts = map[string]*obj.Link{
	"amd64": obj.Linknew(&x86.Linkamd64),
	"s390x": obj.Linknew(&s390x.Links390x),
	"arm64": obj.Linknew(&arm64.Linkarm64),
}

func testConfig(tb testing.TB) *Conf      { return testConfigArch(tb, "amd64") }
func testConfigS390X(tb testing.TB) *Conf { return testConfigArch(tb, "s390x") }
func testConfigARM64(tb testing.TB) *Conf { return testConfigArch(tb, "arm64") }

func testConfigArch(tb testing.TB, arch string) *Conf {
	ctxt, ok := testCtxts[arch]
	if !ok {
		tb.Fatalf("unknown arch %s", arch)
	}
	if ctxt.Arch.PtrSize != 8 {
		tb.Fatal("testTypes is 64-bit only")
	}
	c := &Conf{
		config: NewConfig(arch, testTypes, ctxt, true),
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
		c.fe = TestFrontend{t: c.tb, ctxt: c.config.ctxt}
	}
	return c.fe
}

// TestFrontend is a test-only frontend.
// It assumes 64 bit integers and pointers.
type TestFrontend struct {
	t    testing.TB
	ctxt *obj.Link
}

func (TestFrontend) StringData(s string) *obj.LSym {
	return nil
}
func (TestFrontend) Auto(pos src.XPos, t *types.Type) *ir.Name {
	n := ir.NewNameAt(pos, &types.Sym{Name: "aFakeAuto"})
	n.Class = ir.PAUTO
	return n
}
func (d TestFrontend) SplitSlot(parent *LocalSlot, suffix string, offset int64, t *types.Type) LocalSlot {
	return LocalSlot{N: parent.N, Type: t, Off: offset}
}
func (TestFrontend) Line(_ src.XPos) string {
	return "unknown.go:0"
}
func (TestFrontend) AllocFrame(f *Func) {
}
func (d TestFrontend) Syslook(s string) *obj.LSym {
	return d.ctxt.Lookup(s)
}
func (TestFrontend) UseWriteBarrier() bool {
	return true // only writebarrier_test cares
}
func (TestFrontend) SetWBPos(pos src.XPos) {
}

func (d TestFrontend) Logf(msg string, args ...interface{}) { d.t.Logf(msg, args...) }
func (d TestFrontend) Log() bool                            { return true }

func (d TestFrontend) Fatalf(_ src.XPos, msg string, args ...interface{}) { d.t.Fatalf(msg, args...) }
func (d TestFrontend) Warnl(_ src.XPos, msg string, args ...interface{})  { d.t.Logf(msg, args...) }
func (d TestFrontend) Debug_checknil() bool                               { return false }

func (d TestFrontend) MyImportPath() string {
	return "my/import/path"
}

var testTypes Types

func init() {
	// Initialize just enough of the universe and the types package to make our tests function.
	// TODO(josharian): move universe initialization to the types package,
	// so this test setup can share it.

	for _, typ := range [...]struct {
		width int64
		et    types.Kind
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
	testTypes.SetTypPtrs()
}

func (d TestFrontend) DerefItab(sym *obj.LSym, off int64) *obj.LSym { return nil }

func (d TestFrontend) CanSSA(t *types.Type) bool {
	// There are no un-SSAable types in test land.
	return true
}
