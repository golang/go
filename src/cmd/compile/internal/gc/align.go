// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "cmd/internal/obj"

// machine size and rounding alignment is dictated around
// the size of a pointer, set in betypeinit (see ../amd64/galign.go).
var defercalc int

func Rnd(o int64, r int64) int64 {
	if r < 1 || r > 8 || r&(r-1) != 0 {
		Fatalf("rnd %d", r)
	}
	return (o + r - 1) &^ (r - 1)
}

func offmod(t *Type) {
	o := int32(0)
	for f := t.Type; f != nil; f = f.Down {
		if f.Etype != TFIELD {
			Fatalf("offmod: not TFIELD: %v", Tconv(f, obj.FmtLong))
		}
		f.Width = int64(o)
		o += int32(Widthptr)
		if int64(o) >= Thearch.MAXWIDTH {
			Yyerror("interface too large")
			o = int32(Widthptr)
		}
	}
}

func widstruct(errtype *Type, t *Type, o int64, flag int) int64 {
	starto := o
	maxalign := int32(flag)
	if maxalign < 1 {
		maxalign = 1
	}
	lastzero := int64(0)
	var w int64
	for f := t.Type; f != nil; f = f.Down {
		if f.Etype != TFIELD {
			Fatalf("widstruct: not TFIELD: %v", Tconv(f, obj.FmtLong))
		}
		if f.Type == nil {
			// broken field, just skip it so that other valid fields
			// get a width.
			continue
		}

		dowidth(f.Type)
		if int32(f.Type.Align) > maxalign {
			maxalign = int32(f.Type.Align)
		}
		if f.Type.Width < 0 {
			Fatalf("invalid width %d", f.Type.Width)
		}
		w = f.Type.Width
		if f.Type.Align > 0 {
			o = Rnd(o, int64(f.Type.Align))
		}
		f.Width = o // really offset for TFIELD
		if f.Nname != nil {
			// this same stackparam logic is in addrescapes
			// in typecheck.go.  usually addrescapes runs after
			// widstruct, in which case we could drop this,
			// but function closure functions are the exception.
			if f.Nname.Name.Param.Stackparam != nil {
				f.Nname.Name.Param.Stackparam.Xoffset = o
				f.Nname.Xoffset = 0
			} else {
				f.Nname.Xoffset = o
			}
		}

		if w == 0 {
			lastzero = o
		}
		o += w
		if o >= Thearch.MAXWIDTH {
			Yyerror("type %v too large", Tconv(errtype, obj.FmtLong))
			o = 8 // small but nonzero
		}
	}

	// For nonzero-sized structs which end in a zero-sized thing, we add
	// an extra byte of padding to the type.  This padding ensures that
	// taking the address of the zero-sized thing can't manufacture a
	// pointer to the next object in the heap.  See issue 9401.
	if flag == 1 && o > starto && o == lastzero {
		o++
	}

	// final width is rounded
	if flag != 0 {
		o = Rnd(o, int64(maxalign))
	}
	t.Align = uint8(maxalign)

	// type width only includes back to first field's offset
	t.Width = o - starto

	return o
}

func dowidth(t *Type) {
	if Widthptr == 0 {
		Fatalf("dowidth without betypeinit")
	}

	if t == nil {
		return
	}

	if t.Width > 0 {
		if t.Align == 0 {
			// See issue 11354
			Fatalf("zero alignment with nonzero size %v", t)
		}
		return
	}

	if t.Width == -2 {
		lno := int(lineno)
		lineno = int32(t.Lineno)
		if !t.Broke {
			t.Broke = true
			Yyerror("invalid recursive type %v", t)
		}

		t.Width = 0
		lineno = int32(lno)
		return
	}

	// break infinite recursion if the broken recursive type
	// is referenced again
	if t.Broke && t.Width == 0 {
		return
	}

	// defer checkwidth calls until after we're done
	defercalc++

	lno := int(lineno)
	lineno = int32(t.Lineno)
	t.Width = -2
	t.Align = 0

	et := t.Etype
	switch et {
	case TFUNC, TCHAN, TMAP, TSTRING:
		break

	// simtype == 0 during bootstrap
	default:
		if Simtype[t.Etype] != 0 {
			et = Simtype[t.Etype]
		}
	}

	w := int64(0)
	switch et {
	default:
		Fatalf("dowidth: unknown type: %v", t)

	// compiler-specific stuff
	case TINT8, TUINT8, TBOOL:
		// bool is int8
		w = 1

	case TINT16, TUINT16:
		w = 2

	case TINT32, TUINT32, TFLOAT32:
		w = 4

	case TINT64, TUINT64, TFLOAT64, TCOMPLEX64:
		w = 8
		t.Align = uint8(Widthreg)

	case TCOMPLEX128:
		w = 16
		t.Align = uint8(Widthreg)

	case TPTR32:
		w = 4
		checkwidth(t.Type)

	case TPTR64:
		w = 8
		checkwidth(t.Type)

	case TUNSAFEPTR:
		w = int64(Widthptr)

	case TINTER: // implemented as 2 pointers
		w = 2 * int64(Widthptr)

		t.Align = uint8(Widthptr)
		offmod(t)

	case TCHAN: // implemented as pointer
		w = int64(Widthptr)

		checkwidth(t.Type)

		// make fake type to check later to
		// trigger channel argument check.
		t1 := typ(TCHANARGS)

		t1.Type = t
		checkwidth(t1)

	case TCHANARGS:
		t1 := t.Type
		dowidth(t.Type) // just in case
		if t1.Type.Width >= 1<<16 {
			Yyerror("channel element type too large (>64kB)")
		}
		t.Width = 1

	case TMAP: // implemented as pointer
		w = int64(Widthptr)

		checkwidth(t.Type)
		checkwidth(t.Down)

	case TFORW: // should have been filled in
		if !t.Broke {
			Yyerror("invalid recursive type %v", t)
		}
		w = 1 // anything will do

	// dummy type; should be replaced before use.
	case TANY:
		if Debug['A'] == 0 {
			Fatalf("dowidth any")
		}
		w = 1 // anything will do

	case TSTRING:
		if sizeof_String == 0 {
			Fatalf("early dowidth string")
		}
		w = int64(sizeof_String)
		t.Align = uint8(Widthptr)

	case TARRAY:
		if t.Type == nil {
			break
		}
		if t.Bound >= 0 {
			dowidth(t.Type)
			if t.Type.Width != 0 {
				cap := (uint64(Thearch.MAXWIDTH) - 1) / uint64(t.Type.Width)
				if uint64(t.Bound) > cap {
					Yyerror("type %v larger than address space", Tconv(t, obj.FmtLong))
				}
			}

			w = t.Bound * t.Type.Width
			t.Align = t.Type.Align
		} else if t.Bound == -1 {
			w = int64(sizeof_Array)
			checkwidth(t.Type)
			t.Align = uint8(Widthptr)
		} else if t.Bound == -100 {
			if !t.Broke {
				Yyerror("use of [...] array outside of array literal")
				t.Broke = true
			}
		} else {
			Fatalf("dowidth %v", t) // probably [...]T
		}

	case TSTRUCT:
		if t.Funarg {
			Fatalf("dowidth fn struct %v", t)
		}
		w = widstruct(t, t, 0, 1)

	// make fake type to check later to
	// trigger function argument computation.
	case TFUNC:
		t1 := typ(TFUNCARGS)

		t1.Type = t
		checkwidth(t1)

		// width of func type is pointer
		w = int64(Widthptr)

	// function is 3 cated structures;
	// compute their widths as side-effect.
	case TFUNCARGS:
		t1 := t.Type

		w = widstruct(t.Type, *getthis(t1), 0, 0)
		w = widstruct(t.Type, *getinarg(t1), w, Widthreg)
		w = widstruct(t.Type, *Getoutarg(t1), w, Widthreg)
		t1.Argwid = w
		if w%int64(Widthreg) != 0 {
			Warn("bad type %v %d\n", t1, w)
		}
		t.Align = 1
	}

	if Widthptr == 4 && w != int64(int32(w)) {
		Yyerror("type %v too large", t)
	}

	t.Width = w
	if t.Align == 0 {
		if w > 8 || w&(w-1) != 0 {
			Fatalf("invalid alignment for %v", t)
		}
		t.Align = uint8(w)
	}

	lineno = int32(lno)

	if defercalc == 1 {
		resumecheckwidth()
	} else {
		defercalc--
	}
}

// when a type's width should be known, we call checkwidth
// to compute it.  during a declaration like
//
//	type T *struct { next T }
//
// it is necessary to defer the calculation of the struct width
// until after T has been initialized to be a pointer to that struct.
// similarly, during import processing structs may be used
// before their definition.  in those situations, calling
// defercheckwidth() stops width calculations until
// resumecheckwidth() is called, at which point all the
// checkwidths that were deferred are executed.
// dowidth should only be called when the type's size
// is needed immediately.  checkwidth makes sure the
// size is evaluated eventually.
type TypeList struct {
	t    *Type
	next *TypeList
}

var tlfree *TypeList

var tlq *TypeList

func checkwidth(t *Type) {
	if t == nil {
		return
	}

	// function arg structs should not be checked
	// outside of the enclosing function.
	if t.Funarg {
		Fatalf("checkwidth %v", t)
	}

	if defercalc == 0 {
		dowidth(t)
		return
	}

	if t.Deferwidth {
		return
	}
	t.Deferwidth = true

	l := tlfree
	if l != nil {
		tlfree = l.next
	} else {
		l = new(TypeList)
	}

	l.t = t
	l.next = tlq
	tlq = l
}

func defercheckwidth() {
	// we get out of sync on syntax errors, so don't be pedantic.
	if defercalc != 0 && nerrors == 0 {
		Fatalf("defercheckwidth")
	}
	defercalc = 1
}

func resumecheckwidth() {
	if defercalc == 0 {
		Fatalf("resumecheckwidth")
	}
	for l := tlq; l != nil; l = tlq {
		l.t.Deferwidth = false
		tlq = l.next
		dowidth(l.t)
		l.next = tlfree
		tlfree = l
	}

	defercalc = 0
}

var itable *Type // distinguished *byte

func typeinit() {
	if Widthptr == 0 {
		Fatalf("typeinit before betypeinit")
	}

	for et := EType(0); et < NTYPE; et++ {
		Simtype[et] = et
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
		Isint[et] = true
	}
	Isint[TINT] = true
	Isint[TUINT] = true
	Isint[TUINTPTR] = true

	Isfloat[TFLOAT32] = true
	Isfloat[TFLOAT64] = true

	Iscomplex[TCOMPLEX64] = true
	Iscomplex[TCOMPLEX128] = true

	Isptr[TPTR32] = true
	Isptr[TPTR64] = true

	isforw[TFORW] = true

	Issigned[TINT] = true
	Issigned[TINT8] = true
	Issigned[TINT16] = true
	Issigned[TINT32] = true
	Issigned[TINT64] = true

	// initialize okfor
	for et := EType(0); et < NTYPE; et++ {
		if Isint[et] || et == TIDEAL {
			okforeq[et] = true
			okforcmp[et] = true
			okforarith[et] = true
			okforadd[et] = true
			okforand[et] = true
			okforconst[et] = true
			issimple[et] = true
			Minintval[et] = new(Mpint)
			Maxintval[et] = new(Mpint)
		}

		if Isfloat[et] {
			okforeq[et] = true
			okforcmp[et] = true
			okforadd[et] = true
			okforarith[et] = true
			okforconst[et] = true
			issimple[et] = true
			minfltval[et] = newMpflt()
			maxfltval[et] = newMpflt()
		}

		if Iscomplex[et] {
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

	okforconst[TBOOL] = true
	okforconst[TSTRING] = true

	okforlen[TARRAY] = true
	okforlen[TCHAN] = true
	okforlen[TMAP] = true
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
	okforeq[TARRAY] = true  // nil slice only; refined in typecheck
	okforeq[TSTRUCT] = true // it's complicated; refined in typecheck

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

	mpatofix(Maxintval[TINT8], "0x7f")
	mpatofix(Minintval[TINT8], "-0x80")
	mpatofix(Maxintval[TINT16], "0x7fff")
	mpatofix(Minintval[TINT16], "-0x8000")
	mpatofix(Maxintval[TINT32], "0x7fffffff")
	mpatofix(Minintval[TINT32], "-0x80000000")
	mpatofix(Maxintval[TINT64], "0x7fffffffffffffff")
	mpatofix(Minintval[TINT64], "-0x8000000000000000")

	mpatofix(Maxintval[TUINT8], "0xff")
	mpatofix(Maxintval[TUINT16], "0xffff")
	mpatofix(Maxintval[TUINT32], "0xffffffff")
	mpatofix(Maxintval[TUINT64], "0xffffffffffffffff")

	// f is valid float if min < f < max.  (min and max are not themselves valid.)
	mpatoflt(maxfltval[TFLOAT32], "33554431p103") // 2^24-1 p (127-23) + 1/2 ulp
	mpatoflt(minfltval[TFLOAT32], "-33554431p103")
	mpatoflt(maxfltval[TFLOAT64], "18014398509481983p970") // 2^53-1 p (1023-52) + 1/2 ulp
	mpatoflt(minfltval[TFLOAT64], "-18014398509481983p970")

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
	Simtype[TMAP] = Tptr

	Simtype[TCHAN] = Tptr
	Simtype[TFUNC] = Tptr
	Simtype[TUNSAFEPTR] = Tptr

	// pick up the backend thearch.typedefs
	for i = range Thearch.Typedefs {
		s := Lookup(Thearch.Typedefs[i].Name)
		s1 := Pkglookup(Thearch.Typedefs[i].Name, builtinpkg)

		etype := Thearch.Typedefs[i].Etype
		if int(etype) >= len(Types) {
			Fatalf("typeinit: %s bad etype", s.Name)
		}
		sameas := Thearch.Typedefs[i].Sameas
		if int(sameas) >= len(Types) {
			Fatalf("typeinit: %s bad sameas", s.Name)
		}
		Simtype[etype] = sameas
		minfltval[etype] = minfltval[sameas]
		maxfltval[etype] = maxfltval[sameas]
		Minintval[etype] = Minintval[sameas]
		Maxintval[etype] = Maxintval[sameas]

		t = Types[etype]
		if t != nil {
			Fatalf("typeinit: %s already defined", s.Name)
		}

		t = typ(etype)
		t.Sym = s1

		dowidth(t)
		Types[etype] = t
		s1.Def = typenod(t)
		s1.Def.Name = new(Name)
	}

	Array_array = int(Rnd(0, int64(Widthptr)))
	Array_nel = int(Rnd(int64(Array_array)+int64(Widthptr), int64(Widthint)))
	Array_cap = int(Rnd(int64(Array_nel)+int64(Widthint), int64(Widthint)))
	sizeof_Array = int(Rnd(int64(Array_cap)+int64(Widthint), int64(Widthptr)))

	// string is same as slice wo the cap
	sizeof_String = int(Rnd(int64(Array_nel)+int64(Widthint), int64(Widthptr)))

	dowidth(Types[TSTRING])
	dowidth(idealstring)

	itable = typ(Tptr)
	itable.Type = Types[TUINT8]
}

// compute total size of f's in/out arguments.
func Argsize(t *Type) int {
	var save Iter
	var x int64

	w := int64(0)

	fp := Structfirst(&save, Getoutarg(t))
	for fp != nil {
		x = fp.Width + fp.Type.Width
		if x > w {
			w = x
		}
		fp = structnext(&save)
	}

	fp = funcfirst(&save, t)
	for fp != nil {
		x = fp.Width + fp.Type.Width
		if x > w {
			w = x
		}
		fp = funcnext(&save)
	}

	w = (w + int64(Widthptr) - 1) &^ (int64(Widthptr) - 1)
	if int64(int(w)) != w {
		Fatalf("argsize too big")
	}
	return int(w)
}
