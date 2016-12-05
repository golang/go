// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

// builtinpkg is a fake package that declares the universe block.
var builtinpkg *Pkg

var itable *Type // distinguished *byte

var basicTypes = [...]struct {
	name  string
	etype EType
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
	etype    EType
	width    *int
	sameas32 EType
	sameas64 EType
}{
	{"int", TINT, &Widthint, TINT32, TINT64},
	{"uint", TUINT, &Widthint, TUINT32, TUINT64},
	{"uintptr", TUINTPTR, &Widthptr, TUINT32, TUINT64},
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
	for _, s := range basicTypes {
		etype := s.etype
		if int(etype) >= len(Types) {
			Fatalf("lexinit: %s bad etype", s.name)
		}
		s2 := Pkglookup(s.name, builtinpkg)
		t := Types[etype]
		if t == nil {
			t = typ(etype)
			t.Sym = s2
			if etype != TANY && etype != TSTRING {
				dowidth(t)
			}
			Types[etype] = t
		}
		s2.Def = typenod(t)
		s2.Def.Name = new(Name)
	}

	for _, s := range builtinFuncs {
		// TODO(marvin): Fix Node.EType type union.
		s2 := Pkglookup(s.name, builtinpkg)
		s2.Def = nod(ONAME, nil, nil)
		s2.Def.Sym = s2
		s2.Def.Etype = EType(s.op)
	}

	for _, s := range unsafeFuncs {
		s2 := Pkglookup(s.name, unsafepkg)
		s2.Def = nod(ONAME, nil, nil)
		s2.Def.Sym = s2
		s2.Def.Etype = EType(s.op)
	}

	idealstring = typ(TSTRING)
	idealbool = typ(TBOOL)
	Types[TANY] = typ(TANY)

	s := Pkglookup("true", builtinpkg)
	s.Def = nodbool(true)
	s.Def.Sym = lookup("true")
	s.Def.Name = new(Name)
	s.Def.Type = idealbool

	s = Pkglookup("false", builtinpkg)
	s.Def = nodbool(false)
	s.Def.Sym = lookup("false")
	s.Def.Name = new(Name)
	s.Def.Type = idealbool

	s = lookup("_")
	s.Block = -100
	s.Def = nod(ONAME, nil, nil)
	s.Def.Sym = s
	Types[TBLANK] = typ(TBLANK)
	s.Def.Type = Types[TBLANK]
	nblank = s.Def

	s = Pkglookup("_", builtinpkg)
	s.Block = -100
	s.Def = nod(ONAME, nil, nil)
	s.Def.Sym = s
	Types[TBLANK] = typ(TBLANK)
	s.Def.Type = Types[TBLANK]

	Types[TNIL] = typ(TNIL)
	s = Pkglookup("nil", builtinpkg)
	var v Val
	v.U = new(NilVal)
	s.Def = nodlit(v)
	s.Def.Sym = s
	s.Def.Name = new(Name)

	s = Pkglookup("iota", builtinpkg)
	s.Def = nod(OIOTA, nil, nil)
	s.Def.Sym = s
	s.Def.Name = new(Name)
}

func typeinit() {
	if Widthptr == 0 {
		Fatalf("typeinit before betypeinit")
	}

	for et := EType(0); et < NTYPE; et++ {
		simtype[et] = et
	}

	Types[TPTR32] = typ(TPTR32)
	dowidth(Types[TPTR32])

	Types[TPTR64] = typ(TPTR64)
	dowidth(Types[TPTR64])

	t := typ(TUNSAFEPTR)
	Types[TUNSAFEPTR] = t
	t.Sym = Pkglookup("Pointer", unsafepkg)
	t.Sym.Def = typenod(t)
	t.Sym.Def.Name = new(Name)
	dowidth(Types[TUNSAFEPTR])

	Tptr = TPTR32
	if Widthptr == 8 {
		Tptr = TPTR64
	}

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

	isforw[TFORW] = true

	// initialize okfor
	for et := EType(0); et < NTYPE; et++ {
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

	okforeq[TPTR32] = true
	okforeq[TPTR64] = true
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
	okfor[OHMUL] = okforarith[:]
	okfor[OMUL] = okforarith[:]
	okfor[ONE] = okforeq[:]
	okfor[OOR] = okforand[:]
	okfor[OOROR] = okforbool[:]
	okfor[OSUB] = okforarith[:]
	okfor[OXOR] = okforand[:]
	okfor[OLSH] = okforand[:]
	okfor[ORSH] = okforand[:]

	// unary
	okfor[OCOM] = okforand[:]

	okfor[OMINUS] = okforarith[:]
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

	// for walk to use in error messages
	Types[TFUNC] = functype(nil, nil, nil)

	// types used in front end
	// types[TNIL] got set early in lexinit
	Types[TIDEAL] = typ(TIDEAL)

	Types[TINTER] = typ(TINTER)

	// simple aliases
	simtype[TMAP] = Tptr

	simtype[TCHAN] = Tptr
	simtype[TFUNC] = Tptr
	simtype[TUNSAFEPTR] = Tptr

	array_array = int(Rnd(0, int64(Widthptr)))
	array_nel = int(Rnd(int64(array_array)+int64(Widthptr), int64(Widthint)))
	array_cap = int(Rnd(int64(array_nel)+int64(Widthint), int64(Widthint)))
	sizeof_Array = int(Rnd(int64(array_cap)+int64(Widthint), int64(Widthptr)))

	// string is same as slice wo the cap
	sizeof_String = int(Rnd(int64(array_nel)+int64(Widthint), int64(Widthptr)))

	dowidth(Types[TSTRING])
	dowidth(idealstring)

	itable = typPtr(Types[TUINT8])
}

func makeErrorInterface() *Type {
	field := newField()
	field.Type = Types[TSTRING]
	f := functypefield(fakethisfield(), nil, []*Field{field})

	field = newField()
	field.Sym = lookup("Error")
	field.Type = f

	t := typ(TINTER)
	t.SetFields([]*Field{field})
	return t
}

func lexinit1() {
	// error type
	s := Pkglookup("error", builtinpkg)
	errortype = makeErrorInterface()
	errortype.Sym = s
	// TODO: If we can prove that it's safe to set errortype.Orig here
	// than we don't need the special errortype/errorInterface case in
	// bexport.go. See also issue #15920.
	// errortype.Orig = makeErrorInterface()
	s.Def = typenod(errortype)

	// byte alias
	s = Pkglookup("byte", builtinpkg)
	bytetype = typ(TUINT8)
	bytetype.Sym = s
	s.Def = typenod(bytetype)
	s.Def.Name = new(Name)

	// rune alias
	s = Pkglookup("rune", builtinpkg)
	runetype = typ(TINT32)
	runetype.Sym = s
	s.Def = typenod(runetype)
	s.Def.Name = new(Name)

	// backend-dependent builtin types (e.g. int).
	for _, s := range typedefs {
		s1 := Pkglookup(s.name, builtinpkg)

		sameas := s.sameas32
		if *s.width == 8 {
			sameas = s.sameas64
		}

		simtype[s.etype] = sameas
		minfltval[s.etype] = minfltval[sameas]
		maxfltval[s.etype] = maxfltval[sameas]
		minintval[s.etype] = minintval[sameas]
		maxintval[s.etype] = maxintval[sameas]

		t := typ(s.etype)
		t.Sym = s1
		Types[s.etype] = t
		s1.Def = typenod(t)
		s1.Def.Name = new(Name)
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

	nodfp = nod(ONAME, nil, nil)
	nodfp.Type = Types[TINT32]
	nodfp.Xoffset = 0
	nodfp.Class = PPARAM
	nodfp.Sym = lookup(".fp")
}
