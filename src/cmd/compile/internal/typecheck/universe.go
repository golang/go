// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

var (
	okfor [ir.OEND][]bool
	iscmp [ir.OEND]bool
)

var (
	okforeq    [types.NTYPE]bool
	okforadd   [types.NTYPE]bool
	okforand   [types.NTYPE]bool
	okfornone  [types.NTYPE]bool
	okforbool  [types.NTYPE]bool
	okforcap   [types.NTYPE]bool
	okforlen   [types.NTYPE]bool
	okforarith [types.NTYPE]bool
)

var basicTypes = [...]struct {
	name  string
	etype types.Kind
}{
	{"int8", types.TINT8},
	{"int16", types.TINT16},
	{"int32", types.TINT32},
	{"int64", types.TINT64},
	{"uint8", types.TUINT8},
	{"uint16", types.TUINT16},
	{"uint32", types.TUINT32},
	{"uint64", types.TUINT64},
	{"float32", types.TFLOAT32},
	{"float64", types.TFLOAT64},
	{"complex64", types.TCOMPLEX64},
	{"complex128", types.TCOMPLEX128},
	{"bool", types.TBOOL},
	{"string", types.TSTRING},
}

var typedefs = [...]struct {
	name     string
	etype    types.Kind
	sameas32 types.Kind
	sameas64 types.Kind
}{
	{"int", types.TINT, types.TINT32, types.TINT64},
	{"uint", types.TUINT, types.TUINT32, types.TUINT64},
	{"uintptr", types.TUINTPTR, types.TUINT32, types.TUINT64},
}

var builtinFuncs = [...]struct {
	name string
	op   ir.Op
}{
	{"append", ir.OAPPEND},
	{"cap", ir.OCAP},
	{"close", ir.OCLOSE},
	{"complex", ir.OCOMPLEX},
	{"copy", ir.OCOPY},
	{"delete", ir.ODELETE},
	{"imag", ir.OIMAG},
	{"len", ir.OLEN},
	{"make", ir.OMAKE},
	{"new", ir.ONEW},
	{"panic", ir.OPANIC},
	{"print", ir.OPRINT},
	{"println", ir.OPRINTN},
	{"real", ir.OREAL},
	{"recover", ir.ORECOVER},
}

var unsafeFuncs = [...]struct {
	name string
	op   ir.Op
}{
	{"Alignof", ir.OALIGNOF},
	{"Offsetof", ir.OOFFSETOF},
	{"Sizeof", ir.OSIZEOF},
}

// InitUniverse initializes the universe block.
func InitUniverse() {
	if types.PtrSize == 0 {
		base.Fatalf("typeinit before betypeinit")
	}

	types.SlicePtrOffset = 0
	types.SliceLenOffset = types.Rnd(types.SlicePtrOffset+int64(types.PtrSize), int64(types.PtrSize))
	types.SliceCapOffset = types.Rnd(types.SliceLenOffset+int64(types.PtrSize), int64(types.PtrSize))
	types.SliceSize = types.Rnd(types.SliceCapOffset+int64(types.PtrSize), int64(types.PtrSize))

	// string is same as slice wo the cap
	types.StringSize = types.Rnd(types.SliceLenOffset+int64(types.PtrSize), int64(types.PtrSize))

	for et := types.Kind(0); et < types.NTYPE; et++ {
		types.SimType[et] = et
	}

	types.Types[types.TANY] = types.New(types.TANY)
	types.Types[types.TINTER] = types.NewInterface(types.LocalPkg, nil)

	defBasic := func(kind types.Kind, pkg *types.Pkg, name string) *types.Type {
		sym := pkg.Lookup(name)
		n := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, sym)
		t := types.NewBasic(kind, n)
		n.SetType(t)
		sym.Def = n
		if kind != types.TANY {
			types.CalcSize(t)
		}
		return t
	}

	for _, s := range &basicTypes {
		types.Types[s.etype] = defBasic(s.etype, types.BuiltinPkg, s.name)
	}

	for _, s := range &typedefs {
		sameas := s.sameas32
		if types.PtrSize == 8 {
			sameas = s.sameas64
		}
		types.SimType[s.etype] = sameas

		types.Types[s.etype] = defBasic(s.etype, types.BuiltinPkg, s.name)
	}

	// We create separate byte and rune types for better error messages
	// rather than just creating type alias *types.Sym's for the uint8 and
	// int32 types. Hence, (bytetype|runtype).Sym.isAlias() is false.
	// TODO(gri) Should we get rid of this special case (at the cost
	// of less informative error messages involving bytes and runes)?
	// (Alternatively, we could introduce an OTALIAS node representing
	// type aliases, albeit at the cost of having to deal with it everywhere).
	types.ByteType = defBasic(types.TUINT8, types.BuiltinPkg, "byte")
	types.RuneType = defBasic(types.TINT32, types.BuiltinPkg, "rune")

	// error type
	s := types.BuiltinPkg.Lookup("error")
	n := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, s)
	types.ErrorType = types.NewNamed(n)
	types.ErrorType.SetUnderlying(makeErrorInterface())
	n.SetType(types.ErrorType)
	s.Def = n
	types.CalcSize(types.ErrorType)

	types.Types[types.TUNSAFEPTR] = defBasic(types.TUNSAFEPTR, ir.Pkgs.Unsafe, "Pointer")

	// simple aliases
	types.SimType[types.TMAP] = types.TPTR
	types.SimType[types.TCHAN] = types.TPTR
	types.SimType[types.TFUNC] = types.TPTR
	types.SimType[types.TUNSAFEPTR] = types.TPTR

	for _, s := range &builtinFuncs {
		s2 := types.BuiltinPkg.Lookup(s.name)
		def := NewName(s2)
		def.BuiltinOp = s.op
		s2.Def = def
	}

	for _, s := range &unsafeFuncs {
		s2 := ir.Pkgs.Unsafe.Lookup(s.name)
		def := NewName(s2)
		def.BuiltinOp = s.op
		s2.Def = def
	}

	s = types.BuiltinPkg.Lookup("true")
	s.Def = ir.NewConstAt(src.NoXPos, s, types.UntypedBool, constant.MakeBool(true))

	s = types.BuiltinPkg.Lookup("false")
	s.Def = ir.NewConstAt(src.NoXPos, s, types.UntypedBool, constant.MakeBool(false))

	s = Lookup("_")
	types.BlankSym = s
	s.Block = -100
	s.Def = NewName(s)
	types.Types[types.TBLANK] = types.New(types.TBLANK)
	ir.AsNode(s.Def).SetType(types.Types[types.TBLANK])
	ir.BlankNode = ir.AsNode(s.Def)
	ir.BlankNode.SetTypecheck(1)

	s = types.BuiltinPkg.Lookup("_")
	s.Block = -100
	s.Def = NewName(s)
	types.Types[types.TBLANK] = types.New(types.TBLANK)
	ir.AsNode(s.Def).SetType(types.Types[types.TBLANK])

	types.Types[types.TNIL] = types.New(types.TNIL)
	s = types.BuiltinPkg.Lookup("nil")
	nnil := NodNil()
	nnil.(*ir.NilExpr).SetSym(s)
	s.Def = nnil

	s = types.BuiltinPkg.Lookup("iota")
	s.Def = ir.NewIota(base.Pos, s)

	for et := types.TINT8; et <= types.TUINT64; et++ {
		types.IsInt[et] = true
	}
	types.IsInt[types.TINT] = true
	types.IsInt[types.TUINT] = true
	types.IsInt[types.TUINTPTR] = true

	types.IsFloat[types.TFLOAT32] = true
	types.IsFloat[types.TFLOAT64] = true

	types.IsComplex[types.TCOMPLEX64] = true
	types.IsComplex[types.TCOMPLEX128] = true

	// initialize okfor
	for et := types.Kind(0); et < types.NTYPE; et++ {
		if types.IsInt[et] || et == types.TIDEAL {
			okforeq[et] = true
			types.IsOrdered[et] = true
			okforarith[et] = true
			okforadd[et] = true
			okforand[et] = true
			ir.OKForConst[et] = true
			types.IsSimple[et] = true
		}

		if types.IsFloat[et] {
			okforeq[et] = true
			types.IsOrdered[et] = true
			okforadd[et] = true
			okforarith[et] = true
			ir.OKForConst[et] = true
			types.IsSimple[et] = true
		}

		if types.IsComplex[et] {
			okforeq[et] = true
			okforadd[et] = true
			okforarith[et] = true
			ir.OKForConst[et] = true
			types.IsSimple[et] = true
		}
	}

	types.IsSimple[types.TBOOL] = true

	okforadd[types.TSTRING] = true

	okforbool[types.TBOOL] = true

	okforcap[types.TARRAY] = true
	okforcap[types.TCHAN] = true
	okforcap[types.TSLICE] = true

	ir.OKForConst[types.TBOOL] = true
	ir.OKForConst[types.TSTRING] = true

	okforlen[types.TARRAY] = true
	okforlen[types.TCHAN] = true
	okforlen[types.TMAP] = true
	okforlen[types.TSLICE] = true
	okforlen[types.TSTRING] = true

	okforeq[types.TPTR] = true
	okforeq[types.TUNSAFEPTR] = true
	okforeq[types.TINTER] = true
	okforeq[types.TCHAN] = true
	okforeq[types.TSTRING] = true
	okforeq[types.TBOOL] = true
	okforeq[types.TMAP] = true    // nil only; refined in typecheck
	okforeq[types.TFUNC] = true   // nil only; refined in typecheck
	okforeq[types.TSLICE] = true  // nil only; refined in typecheck
	okforeq[types.TARRAY] = true  // only if element type is comparable; refined in typecheck
	okforeq[types.TSTRUCT] = true // only if all struct fields are comparable; refined in typecheck

	types.IsOrdered[types.TSTRING] = true

	for i := range okfor {
		okfor[i] = okfornone[:]
	}

	// binary
	okfor[ir.OADD] = okforadd[:]
	okfor[ir.OAND] = okforand[:]
	okfor[ir.OANDAND] = okforbool[:]
	okfor[ir.OANDNOT] = okforand[:]
	okfor[ir.ODIV] = okforarith[:]
	okfor[ir.OEQ] = okforeq[:]
	okfor[ir.OGE] = types.IsOrdered[:]
	okfor[ir.OGT] = types.IsOrdered[:]
	okfor[ir.OLE] = types.IsOrdered[:]
	okfor[ir.OLT] = types.IsOrdered[:]
	okfor[ir.OMOD] = okforand[:]
	okfor[ir.OMUL] = okforarith[:]
	okfor[ir.ONE] = okforeq[:]
	okfor[ir.OOR] = okforand[:]
	okfor[ir.OOROR] = okforbool[:]
	okfor[ir.OSUB] = okforarith[:]
	okfor[ir.OXOR] = okforand[:]
	okfor[ir.OLSH] = okforand[:]
	okfor[ir.ORSH] = okforand[:]

	// unary
	okfor[ir.OBITNOT] = okforand[:]
	okfor[ir.ONEG] = okforarith[:]
	okfor[ir.ONOT] = okforbool[:]
	okfor[ir.OPLUS] = okforarith[:]

	// special
	okfor[ir.OCAP] = okforcap[:]
	okfor[ir.OLEN] = okforlen[:]

	// comparison
	iscmp[ir.OLT] = true
	iscmp[ir.OGT] = true
	iscmp[ir.OGE] = true
	iscmp[ir.OLE] = true
	iscmp[ir.OEQ] = true
	iscmp[ir.ONE] = true
}

func makeErrorInterface() *types.Type {
	sig := types.NewSignature(types.NoPkg, fakeRecvField(), nil, nil, []*types.Field{
		types.NewField(src.NoXPos, nil, types.Types[types.TSTRING]),
	})
	method := types.NewField(src.NoXPos, Lookup("Error"), sig)
	return types.NewInterface(types.NoPkg, []*types.Field{method})
}

// DeclareUniverse makes the universe block visible within the current package.
func DeclareUniverse() {
	// Operationally, this is similar to a dot import of builtinpkg, except
	// that we silently skip symbols that are already declared in the
	// package block rather than emitting a redeclared symbol error.

	for _, s := range types.BuiltinPkg.Syms {
		if s.Def == nil {
			continue
		}
		s1 := Lookup(s.Name)
		if s1.Def != nil {
			continue
		}

		s1.Def = s.Def
		s1.Block = s.Block
	}
}
