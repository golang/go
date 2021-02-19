// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bufio"
	"cmd/compile/internal/abi"
	"cmd/compile/internal/base"
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
var configAMD64 = abi.NewABIConfig(9, 15)

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
	typecheck.InitUniverse()
	os.Exit(m.Run())
}

func TestABIUtilsBasic1(t *testing.T) {

	// func(x int32) int32
	i32 := types.Types[types.TINT32]
	ft := mkFuncType(nil, []*types.Type{i32}, []*types.Type{i32})

	// expected results
	exp := makeExpectedDump(`
        IN 0: R{ I0 } spilloffset: 0 typ: int32
        OUT 0: R{ I0 } spilloffset: -1 typ: int32
        offsetToSpillArea: 0 spillAreaSize: 8
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
        IN 0: R{ I0 } spilloffset: 0 typ: int8
        IN 1: R{ I1 } spilloffset: 2 typ: int16
        IN 2: R{ I2 } spilloffset: 4 typ: int32
        IN 3: R{ I3 } spilloffset: 8 typ: int64
        IN 4: R{ F0 } spilloffset: 16 typ: float32
        IN 5: R{ F1 } spilloffset: 20 typ: float32
        IN 6: R{ F2 } spilloffset: 24 typ: float64
        IN 7: R{ F3 } spilloffset: 32 typ: float64
        IN 8: R{ I4 } spilloffset: 40 typ: int8
        IN 9: R{ I5 } spilloffset: 42 typ: int16
        IN 10: R{ I6 } spilloffset: 44 typ: int32
        IN 11: R{ I7 } spilloffset: 48 typ: int64
        IN 12: R{ F4 } spilloffset: 56 typ: float32
        IN 13: R{ F5 } spilloffset: 60 typ: float32
        IN 14: R{ F6 } spilloffset: 64 typ: float64
        IN 15: R{ F7 } spilloffset: 72 typ: float64
        IN 16: R{ F8 F9 } spilloffset: 80 typ: complex128
        IN 17: R{ F10 F11 } spilloffset: 96 typ: complex128
        IN 18: R{ F12 F13 } spilloffset: 112 typ: complex128
        IN 19: R{ } offset: 0 typ: complex128
        IN 20: R{ } offset: 16 typ: complex64
        IN 21: R{ I8 } spilloffset: 128 typ: int8
        IN 22: R{ } offset: 24 typ: int16
        IN 23: R{ } offset: 28 typ: int32
        IN 24: R{ } offset: 32 typ: int64
        IN 25: R{ } offset: 40 typ: int8
        IN 26: R{ } offset: 42 typ: int16
        IN 27: R{ } offset: 44 typ: int32
        IN 28: R{ } offset: 48 typ: int64
        OUT 0: R{ I0 } spilloffset: -1 typ: int32
        OUT 1: R{ F0 } spilloffset: -1 typ: float64
        OUT 2: R{ F1 } spilloffset: -1 typ: float64
        offsetToSpillArea: 56 spillAreaSize: 136
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
        IN 0: R{ I0 } spilloffset: 0 typ: [1]int32
        IN 1: R{ } offset: 0 typ: [0]int32
        IN 2: R{ I1 } spilloffset: 4 typ: [1][1]int32
        IN 3: R{ } offset: 0 typ: [2]int32
        OUT 0: R{ } offset: 8 typ: [2]int32
        OUT 1: R{ I0 } spilloffset: -1 typ: [1]int32
        OUT 2: R{ } offset: 16 typ: [0]int32
        OUT 3: R{ I1 } spilloffset: -1 typ: [1][1]int32
        offsetToSpillArea: 16 spillAreaSize: 8
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
        IN 0: R{ I0 } spilloffset: 0 typ: int8
        IN 1: R{ I1 I2 I3 I4 } spilloffset: 2 typ: struct { int8; int8; struct {}; int8; int16 }
        IN 2: R{ I5 } spilloffset: 8 typ: int64
        OUT 0: R{ I0 I1 I2 I3 } spilloffset: -1 typ: struct { int8; int8; struct {}; int8; int16 }
        OUT 1: R{ I4 } spilloffset: -1 typ: int8
        OUT 2: R{ I5 } spilloffset: -1 typ: int32
        offsetToSpillArea: 0 spillAreaSize: 16
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
        IN 0: R{ I0 } spilloffset: 0 typ: struct { int64; struct {} }
        IN 1: R{ I1 } spilloffset: 16 typ: struct { int64; struct {} }
        IN 2: R{ I2 F0 } spilloffset: 32 typ: struct { float64; struct { int64; struct {} }; struct {} }
        OUT 0: R{ I0 F0 } spilloffset: -1 typ: struct { float64; struct { int64; struct {} }; struct {} }
        OUT 1: R{ I1 F1 } spilloffset: -1 typ: struct { float64; struct { int64; struct {} }; struct {} }
        offsetToSpillArea: 0 spillAreaSize: 64
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
        IN 0: R{ I0 I1 I2 } spilloffset: 0 typ: []int32
        IN 1: R{ I3 } spilloffset: 24 typ: int8
        IN 2: R{ I4 I5 I6 } spilloffset: 32 typ: []int32
        IN 3: R{ I7 } spilloffset: 56 typ: int8
        IN 4: R{ } offset: 0 typ: string
        IN 5: R{ I8 } spilloffset: 57 typ: int8
        IN 6: R{ } offset: 16 typ: int64
        IN 7: R{ } offset: 24 typ: []int32
        OUT 0: R{ I0 I1 } spilloffset: -1 typ: string
        OUT 1: R{ I2 } spilloffset: -1 typ: int64
        OUT 2: R{ I3 I4 } spilloffset: -1 typ: string
        OUT 3: R{ I5 I6 I7 } spilloffset: -1 typ: []int32
        offsetToSpillArea: 48 spillAreaSize: 64
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
        IN 0: R{ I0 I1 I2 } spilloffset: 0 typ: struct { int16; int16; int16 }
        IN 1: R{ I3 } spilloffset: 8 typ: *struct { int16; int16; int16 }
        IN 2: R{ } offset: 0 typ: [7]*struct { int16; int16; int16 }
        IN 3: R{ F0 } spilloffset: 16 typ: float64
        IN 4: R{ I4 } spilloffset: 24 typ: int16
        IN 5: R{ I5 } spilloffset: 26 typ: int16
        IN 6: R{ I6 } spilloffset: 28 typ: int16
        OUT 0: R{ } offset: 56 typ: [7]*struct { int16; int16; int16 }
        OUT 1: R{ F0 } spilloffset: -1 typ: float64
        OUT 2: R{ I0 } spilloffset: -1 typ: int64
        offsetToSpillArea: 112 spillAreaSize: 32
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
        IN 0: R{ I0 I1 I2 } spilloffset: 0 typ: struct { int16; int16; bool }
        IN 1: R{ I3 I4 } spilloffset: 8 typ: interface {}
        IN 2: R{ I5 I6 } spilloffset: 24 typ: interface {}
        IN 3: R{ I7 I8 } spilloffset: 40 typ: interface { () untyped string }
        IN 4: R{ } offset: 0 typ: *interface {}
        IN 5: R{ } offset: 8 typ: interface { () untyped string }
        IN 6: R{ } offset: 24 typ: int16
        OUT 0: R{ I0 I1 } spilloffset: -1 typ: interface {}
        OUT 1: R{ I2 I3 } spilloffset: -1 typ: interface { () untyped string }
        OUT 2: R{ I4 } spilloffset: -1 typ: *interface {}
        offsetToSpillArea: 32 spillAreaSize: 56
`)

	abitest(t, ft, exp)
}

func TestABINumParamRegs(t *testing.T) {
	i8 := types.Types[types.TINT8]
	i16 := types.Types[types.TINT16]
	i32 := types.Types[types.TINT32]
	i64 := types.Types[types.TINT64]
	f32 := types.Types[types.TFLOAT32]
	f64 := types.Types[types.TFLOAT64]
	c64 := types.Types[types.TCOMPLEX64]
	c128 := types.Types[types.TCOMPLEX128]

	s := mkstruct([]*types.Type{i8, i8, mkstruct([]*types.Type{}), i8, i16})
	a := types.NewArray(s, 3)

	nrtest(t, i8, 1)
	nrtest(t, i16, 1)
	nrtest(t, i32, 1)
	nrtest(t, i64, 1)
	nrtest(t, f32, 1)
	nrtest(t, f64, 1)
	nrtest(t, c64, 2)
	nrtest(t, c128, 2)
	nrtest(t, s, 4)
	nrtest(t, a, 12)

}
