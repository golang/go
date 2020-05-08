// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(gri) This file should probably become part of package types.

package gc

import "cmd/compile/internal/types"

// builtinpkg is a fake package that declares the universe block.
var builtinpkg *types.Pkg

var basicTypes = [...]struct {
	name  string
	etype types.EType
}{
	{"int8", TINT8},
	{"int16", TINT16},
	{"int32", TINT32},
	{"int64", TINT64},
	{"uint8", TUINT8},
	{"uint16", TUINT16},
	{"uint32", TUINT32},
	{"uint64", TUINT64},
	{"float32", TFLOAT32},
	{"float64", TFLOAT64},
	{"complex64", TCOMPLEX64},
	{"complex128", TCOMPLEX128},
	{"bool", TBOOL},
	{"string", TSTRING},
}

var typedefs = [...]struct {
	name     string
	etype    types.EType
	sameas32 types.EType
	sameas64 types.EType
}{
	{"int", TINT, TINT32, TINT64},
	{"uint", TUINT, TUINT32, TUINT64},
	{"uintptr", TUINTPTR, TUINT32, TUINT64},
}

var builtinFuncs = [...]struct {
	name string
	op   Op
}{
	{"append", OAPPEND},
	{"cap", OCAP},
	{"close", OCLOSE},
	{"complex", OCOMPLEX},
	{"copy", OCOPY},
	{"delete", ODELETE},
	{"imag", OIMAG},
	{"len", OLEN},
	{"make", OMAKE},
	{"new", ONEW},
	{"panic", OPANIC},
	{"print", OPRINT},
	{"println", OPRINTN},
	{"real", OREAL},
	{"recover", ORECOVER},
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
	op   Op
}{
	{"Alignof", OALIGNOF},
	{"Offsetof", OOFFSETOF},
	{"Sizeof", OSIZEOF},
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
			Fatalf("lexinit: %s bad etype", s.name)
		}
		s2 := builtinpkg.Lookup(s.name)
		t := types.Types[etype]
		if t == nil {
			t = types.New(etype)
			t.Sym = s2
			if etype != TANY && etype != TSTRING {
				dowidth(t)
			}
			types.Types[etype] = t
		}
		s2.Def = asTypesNode(typenod(t))
		asNode(s2.Def).Name = new(Name)
	}

	for _, s := range &builtinFuncs {
		s2 := builtinpkg.Lookup(s.name)
		s2.Def = asTypesNode(newname(s2))
		asNode(s2.Def).SetSubOp(s.op)
	}

	for _, s := range &unsafeFuncs {
		s2 := unsafepkg.Lookup(s.name)
		s2.Def = asTypesNode(newname(s2))
		asNode(s2.Def).SetSubOp(s.op)
	}

	types.Idealstring = types.New(TSTRING)
	types.Idealbool = types.New(TBOOL)
	types.Types[TANY] = types.New(TANY)

	s := builtinpkg.Lookup("true")
	s.Def = asTypesNode(nodbool(true))
	asNode(s.Def).Sym = lookup("true")
	asNode(s.Def).Name = new(Name)
	asNode(s.Def).Type = types.Idealbool

	s = builtinpkg.Lookup("false")
	s.Def = asTypesNode(nodbool(false))
	asNode(s.Def).Sym = lookup("false")
	asNode(s.Def).Name = new(Name)
	asNode(s.Def).Type = types.Idealbool

	s = lookup("_")
	s.Block = -100
	s.Def = asTypesNode(newname(s))
	types.Types[TBLANK] = types.New(TBLANK)
	asNode(s.Def).Type = types.Types[TBLANK]
	nblank = asNode(s.Def)

	s = builtinpkg.Lookup("_")
	s.Block = -100
	s.Def = asTypesNode(newname(s))
	types.Types[TBLANK] = types.New(TBLANK)
	asNode(s.Def).Type = types.Types[TBLANK]

	types.Types[TNIL] = types.New(TNIL)
	s = builtinpkg.Lookup("nil")
	var v Val
	v.U = new(NilVal)
	s.Def = asTypesNode(nodlit(v))
	asNode(s.Def).Sym = s
	asNode(s.Def).Name = new(Name)

	s = builtinpkg.Lookup("iota")
	s.Def = asTypesNode(nod(OIOTA, nil, nil))
	asNode(s.Def).Sym = s
	asNode(s.Def).Name = new(Name)
}

func typeinit() {
	if Widthptr == 0 {
		Fatalf("typeinit before betypeinit")
	}

	for et := types.EType(0); et < NTYPE; et++ {
		simtype[et] = et
	}

	types.Types[TPTR] = types.New(TPTR)
	dowidth(types.Types[TPTR])

	t := types.New(TUNSAFEPTR)
	types.Types[TUNSAFEPTR] = t
	t.Sym = unsafepkg.Lookup("Pointer")
	t.Sym.Def = asTypesNode(typenod(t))
	asNode(t.Sym.Def).Name = new(Name)
	dowidth(types.Types[TUNSAFEPTR])

	for et := TINT8; et <= TUINT64; et++ {
		isInt[et] = true
	}
	isInt[TINT] = true
	isInt[TUINT] = true
	isInt[TUINTPTR] = true

	isFloat[TFLOAT32] = true
	isFloat[TFLOAT64] = true

	isComplex[TCOMPLEX64] = true
	isComplex[TCOMPLEX128] = true

	// initialize okfor
	for et := types.EType(0); et < NTYPE; et++ {
		if isInt[et] || et == TIDEAL {
			okforeq[et] = true
			okforcmp[et] = true
			okforarith[et] = true
			okforadd[et] = true
			okforand[et] = true
			okforconst[et] = true
			issimple[et] = true
			minintval[et] = new(Mpint)
			maxintval[et] = new(Mpint)
		}

		if isFloat[et] {
			okforeq[et] = true
			okforcmp[et] = true
			okforadd[et] = true
			okforarith[et] = true
			okforconst[et] = true
			issimple[et] = true
			minfltval[et] = newMpflt()
			maxfltval[et] = newMpflt()
		}

		if isComplex[et] {
			okforeq[et] = true
			okforadd[et] = true
			okforarith[et] = true
			okforconst[et] = true
			issimple[et] = true
		}
	}

	issimple[TBOOL] = true

	okforadd[TSTRING] = true

	okforbool[TBOOL] = true

	okforcap[TARRAY] = true
	okforcap[TCHAN] = true
	okforcap[TSLICE] = true

	okforconst[TBOOL] = true
	okforconst[TSTRING] = true

	okforlen[TARRAY] = true
	okforlen[TCHAN] = true
	okforlen[TMAP] = true
	okforlen[TSLICE] = true
	okforlen[TSTRING] = true

	okforeq[TPTR] = true
	okforeq[TUNSAFEPTR] = true
	okforeq[TINTER] = true
	okforeq[TCHAN] = true
	okforeq[TSTRING] = true
	okforeq[TBOOL] = true
	okforeq[TMAP] = true    // nil only; refined in typecheck
	okforeq[TFUNC] = true   // nil only; refined in typecheck
	okforeq[TSLICE] = true  // nil only; refined in typecheck
	okforeq[TARRAY] = true  // only if element type is comparable; refined in typecheck
	okforeq[TSTRUCT] = true // only if all struct fields are comparable; refined in typecheck

	okforcmp[TSTRING] = true

	var i int
	for i = 0; i < len(okfor); i++ {
		okfor[i] = okfornone[:]
	}

	// binary
	okfor[OADD] = okforadd[:]
	okfor[OAND] = okforand[:]
	okfor[OANDAND] = okforbool[:]
	okfor[OANDNOT] = okforand[:]
	okfor[ODIV] = okforarith[:]
	okfor[OEQ] = okforeq[:]
	okfor[OGE] = okforcmp[:]
	okfor[OGT] = okforcmp[:]
	okfor[OLE] = okforcmp[:]
	okfor[OLT] = okforcmp[:]
	okfor[OMOD] = okforand[:]
	okfor[OMUL] = okforarith[:]
	okfor[ONE] = okforeq[:]
	okfor[OOR] = okforand[:]
	okfor[OOROR] = okforbool[:]
	okfor[OSUB] = okforarith[:]
	okfor[OXOR] = okforand[:]
	okfor[OLSH] = okforand[:]
	okfor[ORSH] = okforand[:]

	// unary
	okfor[OBITNOT] = okforand[:]
	okfor[ONEG] = okforarith[:]
	okfor[ONOT] = okforbool[:]
	okfor[OPLUS] = okforarith[:]

	// special
	okfor[OCAP] = okforcap[:]
	okfor[OLEN] = okforlen[:]

	// comparison
	iscmp[OLT] = true
	iscmp[OGT] = true
	iscmp[OGE] = true
	iscmp[OLE] = true
	iscmp[OEQ] = true
	iscmp[ONE] = true

	maxintval[TINT8].SetString("0x7f")
	minintval[TINT8].SetString("-0x80")
	maxintval[TINT16].SetString("0x7fff")
	minintval[TINT16].SetString("-0x8000")
	maxintval[TINT32].SetString("0x7fffffff")
	minintval[TINT32].SetString("-0x80000000")
	maxintval[TINT64].SetString("0x7fffffffffffffff")
	minintval[TINT64].SetString("-0x8000000000000000")

	maxintval[TUINT8].SetString("0xff")
	maxintval[TUINT16].SetString("0xffff")
	maxintval[TUINT32].SetString("0xffffffff")
	maxintval[TUINT64].SetString("0xffffffffffffffff")

	// f is valid float if min < f < max.  (min and max are not themselves valid.)
	maxfltval[TFLOAT32].SetString("33554431p103") // 2^24-1 p (127-23) + 1/2 ulp
	minfltval[TFLOAT32].SetString("-33554431p103")
	maxfltval[TFLOAT64].SetString("18014398509481983p970") // 2^53-1 p (1023-52) + 1/2 ulp
	minfltval[TFLOAT64].SetString("-18014398509481983p970")

	maxfltval[TCOMPLEX64] = maxfltval[TFLOAT32]
	minfltval[TCOMPLEX64] = minfltval[TFLOAT32]
	maxfltval[TCOMPLEX128] = maxfltval[TFLOAT64]
	minfltval[TCOMPLEX128] = minfltval[TFLOAT64]

	types.Types[TINTER] = types.New(TINTER) // empty interface

	// simple aliases
	simtype[TMAP] = TPTR
	simtype[TCHAN] = TPTR
	simtype[TFUNC] = TPTR
	simtype[TUNSAFEPTR] = TPTR

	slicePtrOffset = 0
	sliceLenOffset = Rnd(slicePtrOffset+int64(Widthptr), int64(Widthptr))
	sliceCapOffset = Rnd(sliceLenOffset+int64(Widthptr), int64(Widthptr))
	sizeofSlice = Rnd(sliceCapOffset+int64(Widthptr), int64(Widthptr))

	// string is same as slice wo the cap
	sizeofString = Rnd(sliceLenOffset+int64(Widthptr), int64(Widthptr))

	dowidth(types.Types[TSTRING])
	dowidth(types.Idealstring)
}

func makeErrorInterface() *types.Type {
	field := types.NewField()
	field.Type = types.Types[TSTRING]
	f := functypefield(fakeRecvField(), nil, []*types.Field{field})

	field = types.NewField()
	field.Sym = lookup("Error")
	field.Type = f

	t := types.New(TINTER)
	t.SetInterface([]*types.Field{field})
	return t
}

func lexinit1() {
	// error type
	s := builtinpkg.Lookup("error")
	types.Errortype = makeErrorInterface()
	types.Errortype.Sym = s
	types.Errortype.Orig = makeErrorInterface()
	s.Def = asTypesNode(typenod(types.Errortype))
	dowidth(types.Errortype)

	// We create separate byte and rune types for better error messages
	// rather than just creating type alias *types.Sym's for the uint8 and
	// int32 types. Hence, (bytetype|runtype).Sym.isAlias() is false.
	// TODO(gri) Should we get rid of this special case (at the cost
	// of less informative error messages involving bytes and runes)?
	// (Alternatively, we could introduce an OTALIAS node representing
	// type aliases, albeit at the cost of having to deal with it everywhere).

	// byte alias
	s = builtinpkg.Lookup("byte")
	types.Bytetype = types.New(TUINT8)
	types.Bytetype.Sym = s
	s.Def = asTypesNode(typenod(types.Bytetype))
	asNode(s.Def).Name = new(Name)
	dowidth(types.Bytetype)

	// rune alias
	s = builtinpkg.Lookup("rune")
	types.Runetype = types.New(TINT32)
	types.Runetype.Sym = s
	s.Def = asTypesNode(typenod(types.Runetype))
	asNode(s.Def).Name = new(Name)
	dowidth(types.Runetype)

	// backend-dependent builtin types (e.g. int).
	for _, s := range &typedefs {
		s1 := builtinpkg.Lookup(s.name)

		sameas := s.sameas32
		if Widthptr == 8 {
			sameas = s.sameas64
		}

		simtype[s.etype] = sameas
		minfltval[s.etype] = minfltval[sameas]
		maxfltval[s.etype] = maxfltval[sameas]
		minintval[s.etype] = minintval[sameas]
		maxintval[s.etype] = maxintval[sameas]

		t := types.New(s.etype)
		t.Sym = s1
		types.Types[s.etype] = t
		s1.Def = asTypesNode(typenod(t))
		asNode(s1.Def).Name = new(Name)
		s1.Origpkg = builtinpkg

		dowidth(t)
	}
}

// finishUniverse makes the universe block visible within the current package.
func finishUniverse() {
	// Operationally, this is similar to a dot import of builtinpkg, except
	// that we silently skip symbols that are already declared in the
	// package block rather than emitting a redeclared symbol error.

	for _, s := range builtinpkg.Syms {
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

	nodfp = newname(lookup(".fp"))
	nodfp.Type = types.Types[TINT32]
	nodfp.SetClass(PPARAM)
	nodfp.Name.SetUsed(true)
}
