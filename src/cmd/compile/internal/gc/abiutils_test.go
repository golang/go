// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bufio"
	"cmd/compile/internal/base"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"cmd/internal/src"
	"os"
	"testing"
)

// AMD64 registers available:
// - integer: RAX, RBX, RCX, RDI, RSI, R8, R9, r10, R11
// - floating point: X0 - X14
var configAMD64 = ABIConfig{
	regAmounts: RegAmounts{
		intRegs:   9,
		floatRegs: 15,
	},
}

func TestMain(m *testing.M) {
	ssagen.Arch.LinkArch = &x86.Linkamd64
	ssagen.Arch.REGSP = x86.REGSP
	ssagen.Arch.MAXWIDTH = 1 << 50
	types.MaxWidth = ssagen.Arch.MAXWIDTH
	base.Ctxt = obj.Linknew(ssagen.Arch.LinkArch)
	base.Ctxt.DiagFunc = base.Errorf
	base.Ctxt.DiagFlush = base.FlushErrors
	base.Ctxt.Bso = bufio.NewWriter(os.Stdout)
	types.PtrSize = ssagen.Arch.LinkArch.PtrSize
	types.RegSize = ssagen.Arch.LinkArch.RegSize
	types.TypeLinkSym = func(t *types.Type) *obj.LSym {
		return reflectdata.TypeLinksym(t)
	}
	types.TypeLinkSym = func(t *types.Type) *obj.LSym {
		return reflectdata.TypeLinksym(t)
	}
	typecheck.Init()
	os.Exit(m.Run())
}

func TestABIUtilsBasic1(t *testing.T) {

	// func(x int32) int32
	i32 := types.Types[types.TINT32]
	ft := mkFuncType(nil, []*types.Type{i32}, []*types.Type{i32})

	// expected results
	exp := makeExpectedDump(`
		IN 0: R{ I0 } offset: -1 typ: int32
		OUT 0: R{ I0 } offset: -1 typ: int32
		intspill: 1 floatspill: 0 offsetToSpillArea: 0
`)

	abitest(t, ft, exp)
}

func TestABIUtilsBasic2(t *testing.T) {
	// func(x int32, y float64) (int32, float64, float64)
	i8 := types.Types[types.TINT8]
	i16 := types.Types[types.TINT16]
	i32 := types.Types[types.TINT32]
	i64 := types.Types[types.TINT64]
	f32 := types.Types[types.TFLOAT32]
	f64 := types.Types[types.TFLOAT64]
	c64 := types.Types[types.TCOMPLEX64]
	c128 := types.Types[types.TCOMPLEX128]
	ft := mkFuncType(nil,
		[]*types.Type{
			i8, i16, i32, i64,
			f32, f32, f64, f64,
			i8, i16, i32, i64,
			f32, f32, f64, f64,
			c128, c128, c128, c128, c64,
			i8, i16, i32, i64,
			i8, i16, i32, i64},
		[]*types.Type{i32, f64, f64})
	exp := makeExpectedDump(`
		IN 0: R{ I0 } offset: -1 typ: int8
		IN 1: R{ I1 } offset: -1 typ: int16
		IN 2: R{ I2 } offset: -1 typ: int32
		IN 3: R{ I3 } offset: -1 typ: int64
		IN 4: R{ F0 } offset: -1 typ: float32
		IN 5: R{ F1 } offset: -1 typ: float32
		IN 6: R{ F2 } offset: -1 typ: float64
		IN 7: R{ F3 } offset: -1 typ: float64
		IN 8: R{ I4 } offset: -1 typ: int8
		IN 9: R{ I5 } offset: -1 typ: int16
		IN 10: R{ I6 } offset: -1 typ: int32
		IN 11: R{ I7 } offset: -1 typ: int64
		IN 12: R{ F4 } offset: -1 typ: float32
		IN 13: R{ F5 } offset: -1 typ: float32
		IN 14: R{ F6 } offset: -1 typ: float64
		IN 15: R{ F7 } offset: -1 typ: float64
		IN 16: R{ F8 F9 } offset: -1 typ: complex128
		IN 17: R{ F10 F11 } offset: -1 typ: complex128
		IN 18: R{ F12 F13 } offset: -1 typ: complex128
		IN 19: R{ } offset: 0 typ: complex128
		IN 20: R{ F14 } offset: -1 typ: complex64
		IN 21: R{ I8 } offset: -1 typ: int8
		IN 22: R{ } offset: 16 typ: int16
		IN 23: R{ } offset: 20 typ: int32
		IN 24: R{ } offset: 24 typ: int64
		IN 25: R{ } offset: 32 typ: int8
		IN 26: R{ } offset: 34 typ: int16
		IN 27: R{ } offset: 36 typ: int32
		IN 28: R{ } offset: 40 typ: int64
		OUT 0: R{ I0 } offset: -1 typ: int32
		OUT 1: R{ F0 } offset: -1 typ: float64
		OUT 2: R{ F1 } offset: -1 typ: float64
		intspill: 9 floatspill: 15 offsetToSpillArea: 48
`)

	abitest(t, ft, exp)
}

func TestABIUtilsArrays(t *testing.T) {
	i32 := types.Types[types.TINT32]
	ae := types.NewArray(i32, 0)
	a1 := types.NewArray(i32, 1)
	a2 := types.NewArray(i32, 2)
	aa1 := types.NewArray(a1, 1)
	ft := mkFuncType(nil, []*types.Type{a1, ae, aa1, a2},
		[]*types.Type{a2, a1, ae, aa1})

	exp := makeExpectedDump(`
		IN 0: R{ I0 } offset: -1 typ: [1]int32
		IN 1: R{ } offset: 0 typ: [0]int32
		IN 2: R{ I1 } offset: -1 typ: [1][1]int32
		IN 3: R{ } offset: 0 typ: [2]int32
		OUT 0: R{ } offset: 8 typ: [2]int32
		OUT 1: R{ I0 } offset: -1 typ: [1]int32
		OUT 2: R{ } offset: 16 typ: [0]int32
		OUT 3: R{ I1 } offset: -1 typ: [1][1]int32
		intspill: 2 floatspill: 0 offsetToSpillArea: 16
`)

	abitest(t, ft, exp)
}

func TestABIUtilsStruct1(t *testing.T) {
	i8 := types.Types[types.TINT8]
	i16 := types.Types[types.TINT16]
	i32 := types.Types[types.TINT32]
	i64 := types.Types[types.TINT64]
	s := mkstruct([]*types.Type{i8, i8, mkstruct([]*types.Type{}), i8, i16})
	ft := mkFuncType(nil, []*types.Type{i8, s, i64},
		[]*types.Type{s, i8, i32})

	exp := makeExpectedDump(`
		IN 0: R{ I0 } offset: -1 typ: int8
		IN 1: R{ I1 I2 I3 I4 } offset: -1 typ: struct { int8; int8; struct {}; int8; int16 }
		IN 2: R{ I5 } offset: -1 typ: int64
		OUT 0: R{ I0 I1 I2 I3 } offset: -1 typ: struct { int8; int8; struct {}; int8; int16 }
		OUT 1: R{ I4 } offset: -1 typ: int8
		OUT 2: R{ I5 } offset: -1 typ: int32
		intspill: 6 floatspill: 0 offsetToSpillArea: 0
`)

	abitest(t, ft, exp)
}

func TestABIUtilsStruct2(t *testing.T) {
	f64 := types.Types[types.TFLOAT64]
	i64 := types.Types[types.TINT64]
	s := mkstruct([]*types.Type{i64, mkstruct([]*types.Type{})})
	fs := mkstruct([]*types.Type{f64, s, mkstruct([]*types.Type{})})
	ft := mkFuncType(nil, []*types.Type{s, s, fs},
		[]*types.Type{fs, fs})

	exp := makeExpectedDump(`
		IN 0: R{ I0 } offset: -1 typ: struct { int64; struct {} }
		IN 1: R{ I1 } offset: -1 typ: struct { int64; struct {} }
		IN 2: R{ I2 F0 } offset: -1 typ: struct { float64; struct { int64; struct {} }; struct {} }
		OUT 0: R{ I0 F0 } offset: -1 typ: struct { float64; struct { int64; struct {} }; struct {} }
		OUT 1: R{ I1 F1 } offset: -1 typ: struct { float64; struct { int64; struct {} }; struct {} }
		intspill: 3 floatspill: 1 offsetToSpillArea: 0
`)

	abitest(t, ft, exp)
}

func TestABIUtilsSliceString(t *testing.T) {
	i32 := types.Types[types.TINT32]
	sli32 := types.NewSlice(i32)
	str := types.New(types.TSTRING)
	i8 := types.Types[types.TINT8]
	i64 := types.Types[types.TINT64]
	ft := mkFuncType(nil, []*types.Type{sli32, i8, sli32, i8, str, i8, i64, sli32},
		[]*types.Type{str, i64, str, sli32})

	exp := makeExpectedDump(`
		IN 0: R{ I0 I1 I2 } offset: -1 typ: []int32
		IN 1: R{ I3 } offset: -1 typ: int8
		IN 2: R{ I4 I5 I6 } offset: -1 typ: []int32
		IN 3: R{ I7 } offset: -1 typ: int8
		IN 4: R{ } offset: 0 typ: string
		IN 5: R{ I8 } offset: -1 typ: int8
		IN 6: R{ } offset: 16 typ: int64
		IN 7: R{ } offset: 24 typ: []int32
		OUT 0: R{ I0 I1 } offset: -1 typ: string
		OUT 1: R{ I2 } offset: -1 typ: int64
		OUT 2: R{ I3 I4 } offset: -1 typ: string
		OUT 3: R{ I5 I6 I7 } offset: -1 typ: []int32
		intspill: 9 floatspill: 0 offsetToSpillArea: 48
`)

	abitest(t, ft, exp)
}

func TestABIUtilsMethod(t *testing.T) {
	i16 := types.Types[types.TINT16]
	i64 := types.Types[types.TINT64]
	f64 := types.Types[types.TFLOAT64]

	s1 := mkstruct([]*types.Type{i16, i16, i16})
	ps1 := types.NewPtr(s1)
	a7 := types.NewArray(ps1, 7)
	ft := mkFuncType(s1, []*types.Type{ps1, a7, f64, i16, i16, i16},
		[]*types.Type{a7, f64, i64})

	exp := makeExpectedDump(`
		IN 0: R{ I0 I1 I2 } offset: -1 typ: struct { int16; int16; int16 }
		IN 1: R{ I3 } offset: -1 typ: *struct { int16; int16; int16 }
		IN 2: R{ } offset: 0 typ: [7]*struct { int16; int16; int16 }
		IN 3: R{ F0 } offset: -1 typ: float64
		IN 4: R{ I4 } offset: -1 typ: int16
		IN 5: R{ I5 } offset: -1 typ: int16
		IN 6: R{ I6 } offset: -1 typ: int16
		OUT 0: R{ } offset: 56 typ: [7]*struct { int16; int16; int16 }
		OUT 1: R{ F0 } offset: -1 typ: float64
		OUT 2: R{ I0 } offset: -1 typ: int64
		intspill: 7 floatspill: 1 offsetToSpillArea: 112
`)

	abitest(t, ft, exp)
}

func TestABIUtilsInterfaces(t *testing.T) {
	ei := types.Types[types.TINTER] // interface{}
	pei := types.NewPtr(ei)         // *interface{}
	fldt := mkFuncType(types.FakeRecvType(), []*types.Type{},
		[]*types.Type{types.UntypedString})
	field := types.NewField(src.NoXPos, nil, fldt)
	// interface{ ...() string }
	nei := types.NewInterface(types.LocalPkg, []*types.Field{field})

	i16 := types.Types[types.TINT16]
	tb := types.Types[types.TBOOL]
	s1 := mkstruct([]*types.Type{i16, i16, tb})

	ft := mkFuncType(nil, []*types.Type{s1, ei, ei, nei, pei, nei, i16},
		[]*types.Type{ei, nei, pei})

	exp := makeExpectedDump(`
		IN 0: R{ I0 I1 I2 } offset: -1 typ: struct { int16; int16; bool }
		IN 1: R{ I3 I4 } offset: -1 typ: interface {}
		IN 2: R{ I5 I6 } offset: -1 typ: interface {}
		IN 3: R{ I7 I8 } offset: -1 typ: interface { () untyped string }
		IN 4: R{ } offset: 0 typ: *interface {}
		IN 5: R{ } offset: 8 typ: interface { () untyped string }
		IN 6: R{ } offset: 24 typ: int16
		OUT 0: R{ I0 I1 } offset: -1 typ: interface {}
		OUT 1: R{ I2 I3 } offset: -1 typ: interface { () untyped string }
		OUT 2: R{ I4 } offset: -1 typ: *interface {}
		intspill: 9 floatspill: 0 offsetToSpillArea: 32
`)

	abitest(t, ft, exp)
}
