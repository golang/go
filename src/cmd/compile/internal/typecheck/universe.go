// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"go/constant"

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
	{"Add", ir.OUNSAFEADD},
	{"Alignof", ir.OALIGNOF},
	{"Offsetof", ir.OOFFSETOF},
	{"Sizeof", ir.OSIZEOF},
	{"Slice", ir.OUNSAFESLICE},
}

// InitUniverse initializes the universe block.
func InitUniverse() {
	types.InitTypes(func(sym *types.Sym, typ *types.Type) types.Object {
		n := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, sym)
		n.SetType(typ)
		sym.Def = n
		return n
	})

	for _, s := range &builtinFuncs {
		s2 := types.BuiltinPkg.Lookup(s.name)
		def := NewName(s2)
		def.BuiltinOp = s.op
		s2.Def = def
	}

	for _, s := range &unsafeFuncs {
		s2 := types.UnsafePkg.Lookup(s.name)
		def := NewName(s2)
		def.BuiltinOp = s.op
		s2.Def = def
	}

	s := types.BuiltinPkg.Lookup("true")
	s.Def = ir.NewConstAt(src.NoXPos, s, types.UntypedBool, constant.MakeBool(true))

	s = types.BuiltinPkg.Lookup("false")
	s.Def = ir.NewConstAt(src.NoXPos, s, types.UntypedBool, constant.MakeBool(false))

	s = Lookup("_")
	types.BlankSym = s
	s.Def = NewName(s)
	ir.AsNode(s.Def).SetType(types.Types[types.TBLANK])
	ir.BlankNode = ir.AsNode(s.Def)
	ir.BlankNode.SetTypecheck(1)

	s = types.BuiltinPkg.Lookup("_")
	s.Def = NewName(s)
	ir.AsNode(s.Def).SetType(types.Types[types.TBLANK])

	s = types.BuiltinPkg.Lookup("nil")
	nnil := NodNil()
	nnil.(*ir.NilExpr).SetSym(s)
	s.Def = nnil

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
	}
}
