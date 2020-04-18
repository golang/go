// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"sort"
)

// sizeCalculationDisabled indicates whether it is safe
// to calculate Types' widths and alignments. See dowidth.
var sizeCalculationDisabled bool

// machine size and rounding alignment is dictated around
// the size of a pointer, set in betypeinit (see ../amd64/galign.go).
var defercalc int

func Rnd(o int64, r int64) int64 {
	if r < 1 || r > 8 || r&(r-1) != 0 {
		Fatalf("rnd %d", r)
	}
	return (o + r - 1) &^ (r - 1)
}

// expandiface computes the method set for interface type t by
// expanding embedded interfaces.
func expandiface(t *types.Type) {
	seen := make(map[*types.Sym]*types.Field)
	var methods []*types.Field

	addMethod := func(m *types.Field, explicit bool) {
		switch prev := seen[m.Sym]; {
		case prev == nil:
			seen[m.Sym] = m
		case langSupported(1, 14, t.Pkg()) && !explicit && types.Identical(m.Type, prev.Type):
			return
		default:
			yyerrorl(m.Pos, "duplicate method %s", m.Sym.Name)
		}
		methods = append(methods, m)
	}

	for _, m := range t.Methods().Slice() {
		if m.Sym == nil {
			continue
		}

		checkwidth(m.Type)
		addMethod(m, true)
	}

	for _, m := range t.Methods().Slice() {
		if m.Sym != nil {
			continue
		}

		if !m.Type.IsInterface() {
			yyerrorl(m.Pos, "interface contains embedded non-interface %v", m.Type)
			m.SetBroke(true)
			t.SetBroke(true)
			// Add to fields so that error messages
			// include the broken embedded type when
			// printing t.
			// TODO(mdempsky): Revisit this.
			methods = append(methods, m)
			continue
		}

		// Embedded interface: duplicate all methods
		// (including broken ones, if any) and add to t's
		// method set.
		for _, t1 := range m.Type.Fields().Slice() {
			f := types.NewField()
			f.Pos = m.Pos // preserve embedding position
			f.Sym = t1.Sym
			f.Type = t1.Type
			f.SetBroke(t1.Broke())
			addMethod(f, false)
		}
	}

	sort.Sort(methcmp(methods))

	if int64(len(methods)) >= thearch.MAXWIDTH/int64(Widthptr) {
		yyerror("interface too large")
	}
	for i, m := range methods {
		m.Offset = int64(i) * int64(Widthptr)
	}

	// Access fields directly to avoid recursively calling dowidth
	// within Type.Fields().
	t.Extra.(*types.Interface).Fields.Set(methods)
}

func widstruct(errtype *types.Type, t *types.Type, o int64, flag int) int64 {
	starto := o
	maxalign := int32(flag)
	if maxalign < 1 {
		maxalign = 1
	}
	lastzero := int64(0)
	for _, f := range t.Fields().Slice() {
		if f.Type == nil {
			// broken field, just skip it so that other valid fields
			// get a width.
			continue
		}

		dowidth(f.Type)
		if int32(f.Type.Align) > maxalign {
			maxalign = int32(f.Type.Align)
		}
		if f.Type.Align > 0 {
			o = Rnd(o, int64(f.Type.Align))
		}
		f.Offset = o
		if n := asNode(f.Nname); n != nil {
			// addrescapes has similar code to update these offsets.
			// Usually addrescapes runs after widstruct,
			// in which case we could drop this,
			// but function closure functions are the exception.
			// NOTE(rsc): This comment may be stale.
			// It's possible the ordering has changed and this is
			// now the common case. I'm not sure.
			if n.Name.Param.Stackcopy != nil {
				n.Name.Param.Stackcopy.Xoffset = o
				n.Xoffset = 0
			} else {
				n.Xoffset = o
			}
		}

		w := f.Type.Width
		if w < 0 {
			Fatalf("invalid width %d", f.Type.Width)
		}
		if w == 0 {
			lastzero = o
		}
		o += w
		maxwidth := thearch.MAXWIDTH
		// On 32-bit systems, reflect tables impose an additional constraint
		// that each field start offset must fit in 31 bits.
		if maxwidth < 1<<32 {
			maxwidth = 1<<31 - 1
		}
		if o >= maxwidth {
			yyerror("type %L too large", errtype)
			o = 8 // small but nonzero
		}
	}

	// For nonzero-sized structs which end in a zero-sized thing, we add
	// an extra byte of padding to the type. This padding ensures that
	// taking the address of the zero-sized thing can't manufacture a
	// pointer to the next object in the heap. See issue 9401.
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

// dowidth calculates and stores the size and alignment for t.
// If sizeCalculationDisabled is set, and the size/alignment
// have not already been calculated, it calls Fatal.
// This is used to prevent data races in the back end.
func dowidth(t *types.Type) {
	// Calling dowidth when typecheck tracing enabled is not safe.
	// See issue #33658.
	if enableTrace && skipDowidthForTracing {
		return
	}
	if Widthptr == 0 {
		Fatalf("dowidth without betypeinit")
	}

	if t == nil {
		return
	}

	if t.Width == -2 {
		if !t.Broke() {
			t.SetBroke(true)
			yyerrorl(asNode(t.Nod).Pos, "invalid recursive type %v", t)
		}

		t.Width = 0
		t.Align = 1
		return
	}

	if t.WidthCalculated() {
		return
	}

	if sizeCalculationDisabled {
		if t.Broke() {
			// break infinite recursion from Fatal call below
			return
		}
		t.SetBroke(true)
		Fatalf("width not calculated: %v", t)
	}

	// break infinite recursion if the broken recursive type
	// is referenced again
	if t.Broke() && t.Width == 0 {
		return
	}

	// defer checkwidth calls until after we're done
	defercheckwidth()

	lno := lineno
	if asNode(t.Nod) != nil {
		lineno = asNode(t.Nod).Pos
	}

	t.Width = -2
	t.Align = 0 // 0 means use t.Width, below

	et := t.Etype
	switch et {
	case TFUNC, TCHAN, TMAP, TSTRING:
		break

	// simtype == 0 during bootstrap
	default:
		if simtype[t.Etype] != 0 {
			et = simtype[t.Etype]
		}
	}

	var w int64
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

	case TINT64, TUINT64, TFLOAT64:
		w = 8
		t.Align = uint8(Widthreg)

	case TCOMPLEX64:
		w = 8
		t.Align = 4

	case TCOMPLEX128:
		w = 16
		t.Align = uint8(Widthreg)

	case TPTR:
		w = int64(Widthptr)
		checkwidth(t.Elem())

	case TUNSAFEPTR:
		w = int64(Widthptr)

	case TINTER: // implemented as 2 pointers
		w = 2 * int64(Widthptr)
		t.Align = uint8(Widthptr)
		expandiface(t)

	case TCHAN: // implemented as pointer
		w = int64(Widthptr)

		checkwidth(t.Elem())

		// make fake type to check later to
		// trigger channel argument check.
		t1 := types.NewChanArgs(t)
		checkwidth(t1)

	case TCHANARGS:
		t1 := t.ChanArgs()
		dowidth(t1) // just in case
		if t1.Elem().Width >= 1<<16 {
			yyerror("channel element type too large (>64kB)")
		}
		w = 1 // anything will do

	case TMAP: // implemented as pointer
		w = int64(Widthptr)
		checkwidth(t.Elem())
		checkwidth(t.Key())

	case TFORW: // should have been filled in
		if !t.Broke() {
			t.SetBroke(true)
			yyerror("invalid recursive type %v", t)
		}
		w = 1 // anything will do

	case TANY:
		// dummy type; should be replaced before use.
		Fatalf("dowidth any")

	case TSTRING:
		if sizeofString == 0 {
			Fatalf("early dowidth string")
		}
		w = sizeofString
		t.Align = uint8(Widthptr)

	case TARRAY:
		if t.Elem() == nil {
			break
		}

		dowidth(t.Elem())
		if t.Elem().Width != 0 {
			cap := (uint64(thearch.MAXWIDTH) - 1) / uint64(t.Elem().Width)
			if uint64(t.NumElem()) > cap {
				yyerror("type %L larger than address space", t)
			}
		}
		w = t.NumElem() * t.Elem().Width
		t.Align = t.Elem().Align

	case TSLICE:
		if t.Elem() == nil {
			break
		}
		w = sizeofSlice
		checkwidth(t.Elem())
		t.Align = uint8(Widthptr)

	case TSTRUCT:
		if t.IsFuncArgStruct() {
			Fatalf("dowidth fn struct %v", t)
		}
		w = widstruct(t, t, 0, 1)

	// make fake type to check later to
	// trigger function argument computation.
	case TFUNC:
		t1 := types.NewFuncArgs(t)
		checkwidth(t1)
		w = int64(Widthptr) // width of func type is pointer

	// function is 3 cated structures;
	// compute their widths as side-effect.
	case TFUNCARGS:
		t1 := t.FuncArgs()
		w = widstruct(t1, t1.Recvs(), 0, 0)
		w = widstruct(t1, t1.Params(), w, Widthreg)
		w = widstruct(t1, t1.Results(), w, Widthreg)
		t1.Extra.(*types.Func).Argwid = w
		if w%int64(Widthreg) != 0 {
			Warn("bad type %v %d\n", t1, w)
		}
		t.Align = 1
	}

	if Widthptr == 4 && w != int64(int32(w)) {
		yyerror("type %v too large", t)
	}

	t.Width = w
	if t.Align == 0 {
		if w == 0 || w > 8 || w&(w-1) != 0 {
			Fatalf("invalid alignment for %v", t)
		}
		t.Align = uint8(w)
	}

	lineno = lno

	resumecheckwidth()
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

var deferredTypeStack []*types.Type

func checkwidth(t *types.Type) {
	if t == nil {
		return
	}

	// function arg structs should not be checked
	// outside of the enclosing function.
	if t.IsFuncArgStruct() {
		Fatalf("checkwidth %v", t)
	}

	if defercalc == 0 {
		dowidth(t)
		return
	}

	// if type has not yet been pushed on deferredTypeStack yet, do it now
	if !t.Deferwidth() {
		t.SetDeferwidth(true)
		deferredTypeStack = append(deferredTypeStack, t)
	}
}

func defercheckwidth() {
	defercalc++
}

func resumecheckwidth() {
	if defercalc == 1 {
		for len(deferredTypeStack) > 0 {
			t := deferredTypeStack[len(deferredTypeStack)-1]
			deferredTypeStack = deferredTypeStack[:len(deferredTypeStack)-1]
			t.SetDeferwidth(false)
			dowidth(t)
		}
	}

	defercalc--
}
