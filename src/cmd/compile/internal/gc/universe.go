// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(gri) This file should probably become part of package types.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

var basicTypes = [...]struct {
	name  string
	etype types.EType
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
	etype    types.EType
	sameas32 types.EType
	sameas64 types.EType
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

// isBuiltinFuncName reports whether name matches a builtin function
// name.
func isBuiltinFuncName(name string) bool {
	for _, fn := range &builtinFuncs {
		if fn.name == name {
			return true
		}
	}
	return false
}

var unsafeFuncs = [...]struct {
	name string
	op   ir.Op
}{
	{"Alignof", ir.OALIGNOF},
	{"Offsetof", ir.OOFFSETOF},
	{"Sizeof", ir.OSIZEOF},
}

// initUniverse initializes the universe block.
func initUniverse() {
	lexinit()
	typeinit()
	lexinit1()
}

// lexinit initializes known symbols and the basic types.
func lexinit() {
	for _, s := range &basicTypes {
		etype := s.etype
		if int(etype) >= len(types.Types) {
			base.Fatalf("lexinit: %s bad etype", s.name)
		}
		s2 := ir.BuiltinPkg.Lookup(s.name)
		t := types.Types[etype]
		if t == nil {
			t = types.New(etype)
			t.Sym = s2
			if etype != types.TANY && etype != types.TSTRING {
				dowidth(t)
			}
			types.Types[etype] = t
		}
		s2.Def = ir.AsTypesNode(typenod(t))
		ir.AsNode(s2.Def).Name = new(ir.Name)
	}

	for _, s := range &builtinFuncs {
		s2 := ir.BuiltinPkg.Lookup(s.name)
		s2.Def = ir.AsTypesNode(NewName(s2))
		ir.AsNode(s2.Def).SetSubOp(s.op)
	}

	for _, s := range &unsafeFuncs {
		s2 := unsafepkg.Lookup(s.name)
		s2.Def = ir.AsTypesNode(NewName(s2))
		ir.AsNode(s2.Def).SetSubOp(s.op)
	}

	types.UntypedString = types.New(types.TSTRING)
	types.UntypedBool = types.New(types.TBOOL)
	types.Types[types.TANY] = types.New(types.TANY)

	s := ir.BuiltinPkg.Lookup("true")
	s.Def = ir.AsTypesNode(nodbool(true))
	ir.AsNode(s.Def).Sym = lookup("true")
	ir.AsNode(s.Def).Name = new(ir.Name)
	ir.AsNode(s.Def).Type = types.UntypedBool

	s = ir.BuiltinPkg.Lookup("false")
	s.Def = ir.AsTypesNode(nodbool(false))
	ir.AsNode(s.Def).Sym = lookup("false")
	ir.AsNode(s.Def).Name = new(ir.Name)
	ir.AsNode(s.Def).Type = types.UntypedBool

	s = lookup("_")
	s.Block = -100
	s.Def = ir.AsTypesNode(NewName(s))
	types.Types[types.TBLANK] = types.New(types.TBLANK)
	ir.AsNode(s.Def).Type = types.Types[types.TBLANK]
	ir.BlankNode = ir.AsNode(s.Def)

	s = ir.BuiltinPkg.Lookup("_")
	s.Block = -100
	s.Def = ir.AsTypesNode(NewName(s))
	types.Types[types.TBLANK] = types.New(types.TBLANK)
	ir.AsNode(s.Def).Type = types.Types[types.TBLANK]

	types.Types[types.TNIL] = types.New(types.TNIL)
	s = ir.BuiltinPkg.Lookup("nil")
	s.Def = ir.AsTypesNode(nodnil())
	ir.AsNode(s.Def).Sym = s
	ir.AsNode(s.Def).Name = new(ir.Name)

	s = ir.BuiltinPkg.Lookup("iota")
	s.Def = ir.AsTypesNode(ir.Nod(ir.OIOTA, nil, nil))
	ir.AsNode(s.Def).Sym = s
	ir.AsNode(s.Def).Name = new(ir.Name)
}

func typeinit() {
	if Widthptr == 0 {
		base.Fatalf("typeinit before betypeinit")
	}

	for et := types.EType(0); et < types.NTYPE; et++ {
		simtype[et] = et
	}

	types.Types[types.TPTR] = types.New(types.TPTR)
	dowidth(types.Types[types.TPTR])

	t := types.New(types.TUNSAFEPTR)
	types.Types[types.TUNSAFEPTR] = t
	t.Sym = unsafepkg.Lookup("Pointer")
	t.Sym.Def = ir.AsTypesNode(typenod(t))
	ir.AsNode(t.Sym.Def).Name = new(ir.Name)
	dowidth(types.Types[types.TUNSAFEPTR])

	for et := types.TINT8; et <= types.TUINT64; et++ {
		isInt[et] = true
	}
	isInt[types.TINT] = true
	isInt[types.TUINT] = true
	isInt[types.TUINTPTR] = true

	isFloat[types.TFLOAT32] = true
	isFloat[types.TFLOAT64] = true

	isComplex[types.TCOMPLEX64] = true
	isComplex[types.TCOMPLEX128] = true

	// initialize okfor
	for et := types.EType(0); et < types.NTYPE; et++ {
		if isInt[et] || et == types.TIDEAL {
			okforeq[et] = true
			okforcmp[et] = true
			okforarith[et] = true
			okforadd[et] = true
			okforand[et] = true
			ir.OKForConst[et] = true
			issimple[et] = true
		}

		if isFloat[et] {
			okforeq[et] = true
			okforcmp[et] = true
			okforadd[et] = true
			okforarith[et] = true
			ir.OKForConst[et] = true
			issimple[et] = true
		}

		if isComplex[et] {
			okforeq[et] = true
			okforadd[et] = true
			okforarith[et] = true
			ir.OKForConst[et] = true
			issimple[et] = true
		}
	}

	issimple[types.TBOOL] = true

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

	okforcmp[types.TSTRING] = true

	var i int
	for i = 0; i < len(okfor); i++ {
		okfor[i] = okfornone[:]
	}

	// binary
	okfor[ir.OADD] = okforadd[:]
	okfor[ir.OAND] = okforand[:]
	okfor[ir.OANDAND] = okforbool[:]
	okfor[ir.OANDNOT] = okforand[:]
	okfor[ir.ODIV] = okforarith[:]
	okfor[ir.OEQ] = okforeq[:]
	okfor[ir.OGE] = okforcmp[:]
	okfor[ir.OGT] = okforcmp[:]
	okfor[ir.OLE] = okforcmp[:]
	okfor[ir.OLT] = okforcmp[:]
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

	types.Types[types.TINTER] = types.New(types.TINTER) // empty interface

	// simple aliases
	simtype[types.TMAP] = types.TPTR
	simtype[types.TCHAN] = types.TPTR
	simtype[types.TFUNC] = types.TPTR
	simtype[types.TUNSAFEPTR] = types.TPTR

	slicePtrOffset = 0
	sliceLenOffset = Rnd(slicePtrOffset+int64(Widthptr), int64(Widthptr))
	sliceCapOffset = Rnd(sliceLenOffset+int64(Widthptr), int64(Widthptr))
	sizeofSlice = Rnd(sliceCapOffset+int64(Widthptr), int64(Widthptr))

	// string is same as slice wo the cap
	sizeofString = Rnd(sliceLenOffset+int64(Widthptr), int64(Widthptr))

	dowidth(types.Types[types.TSTRING])
	dowidth(types.UntypedString)
}

func makeErrorInterface() *types.Type {
	sig := functypefield(fakeRecvField(), nil, []*types.Field{
		types.NewField(src.NoXPos, nil, types.Types[types.TSTRING]),
	})

	method := types.NewField(src.NoXPos, lookup("Error"), sig)

	t := types.New(types.TINTER)
	t.SetInterface([]*types.Field{method})
	return t
}

func lexinit1() {
	// error type
	s := ir.BuiltinPkg.Lookup("error")
	types.Errortype = makeErrorInterface()
	types.Errortype.Sym = s
	types.Errortype.Orig = makeErrorInterface()
	s.Def = ir.AsTypesNode(typenod(types.Errortype))
	dowidth(types.Errortype)

	// We create separate byte and rune types for better error messages
	// rather than just creating type alias *types.Sym's for the uint8 and
	// int32 types. Hence, (bytetype|runtype).Sym.isAlias() is false.
	// TODO(gri) Should we get rid of this special case (at the cost
	// of less informative error messages involving bytes and runes)?
	// (Alternatively, we could introduce an OTALIAS node representing
	// type aliases, albeit at the cost of having to deal with it everywhere).

	// byte alias
	s = ir.BuiltinPkg.Lookup("byte")
	types.Bytetype = types.New(types.TUINT8)
	types.Bytetype.Sym = s
	s.Def = ir.AsTypesNode(typenod(types.Bytetype))
	ir.AsNode(s.Def).Name = new(ir.Name)
	dowidth(types.Bytetype)

	// rune alias
	s = ir.BuiltinPkg.Lookup("rune")
	types.Runetype = types.New(types.TINT32)
	types.Runetype.Sym = s
	s.Def = ir.AsTypesNode(typenod(types.Runetype))
	ir.AsNode(s.Def).Name = new(ir.Name)
	dowidth(types.Runetype)

	// backend-dependent builtin types (e.g. int).
	for _, s := range &typedefs {
		s1 := ir.BuiltinPkg.Lookup(s.name)

		sameas := s.sameas32
		if Widthptr == 8 {
			sameas = s.sameas64
		}

		simtype[s.etype] = sameas

		t := types.New(s.etype)
		t.Sym = s1
		types.Types[s.etype] = t
		s1.Def = ir.AsTypesNode(typenod(t))
		ir.AsNode(s1.Def).Name = new(ir.Name)
		s1.Origpkg = ir.BuiltinPkg

		dowidth(t)
	}
}

// finishUniverse makes the universe block visible within the current package.
func finishUniverse() {
	// Operationally, this is similar to a dot import of builtinpkg, except
	// that we silently skip symbols that are already declared in the
	// package block rather than emitting a redeclared symbol error.

	for _, s := range ir.BuiltinPkg.Syms {
		if s.Def == nil {
			continue
		}
		s1 := lookup(s.Name)
		if s1.Def != nil {
			continue
		}

		s1.Def = s.Def
		s1.Block = s.Block
	}

	nodfp = NewName(lookup(".fp"))
	nodfp.Type = types.Types[types.TINT32]
	nodfp.SetClass(ir.PPARAM)
	nodfp.Name.SetUsed(true)
}
