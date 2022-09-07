// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"sort"

	"cmd/compile/internal/base"
	"cmd/internal/src"
)

var PtrSize int

var RegSize int

// Slices in the runtime are represented by three components:
//
//	type slice struct {
//		ptr unsafe.Pointer
//		len int
//		cap int
//	}
//
// Strings in the runtime are represented by two components:
//
//	type string struct {
//		ptr unsafe.Pointer
//		len int
//	}
//
// These variables are the offsets of fields and sizes of these structs.
var (
	SlicePtrOffset int64
	SliceLenOffset int64
	SliceCapOffset int64

	SliceSize  int64
	StringSize int64
)

var SkipSizeForTracing bool

// typePos returns the position associated with t.
// This is where t was declared or where it appeared as a type expression.
func typePos(t *Type) src.XPos {
	if pos := t.Pos(); pos.IsKnown() {
		return pos
	}
	base.Fatalf("bad type: %v", t)
	panic("unreachable")
}

// MaxWidth is the maximum size of a value on the target architecture.
var MaxWidth int64

// CalcSizeDisabled indicates whether it is safe
// to calculate Types' widths and alignments. See CalcSize.
var CalcSizeDisabled bool

// machine size and rounding alignment is dictated around
// the size of a pointer, set in gc.Main (see ../gc/main.go).
var defercalc int

// RoundUp rounds o to a multiple of r, r is a power of 2.
func RoundUp(o int64, r int64) int64 {
	if r < 1 || r > 8 || r&(r-1) != 0 {
		base.Fatalf("Round %d", r)
	}
	return (o + r - 1) &^ (r - 1)
}

// expandiface computes the method set for interface type t by
// expanding embedded interfaces.
func expandiface(t *Type) {
	seen := make(map[*Sym]*Field)
	var methods []*Field

	addMethod := func(m *Field, explicit bool) {
		switch prev := seen[m.Sym]; {
		case prev == nil:
			seen[m.Sym] = m
		case !explicit && Identical(m.Type, prev.Type):
			return
		default:
			base.ErrorfAt(m.Pos, "duplicate method %s", m.Sym.Name)
		}
		methods = append(methods, m)
	}

	{
		methods := t.Methods().Slice()
		sort.SliceStable(methods, func(i, j int) bool {
			mi, mj := methods[i], methods[j]

			// Sort embedded types by type name (if any).
			if mi.Sym == nil && mj.Sym == nil {
				return mi.Type.Sym().Less(mj.Type.Sym())
			}

			// Sort methods before embedded types.
			if mi.Sym == nil || mj.Sym == nil {
				return mi.Sym != nil
			}

			// Sort methods by symbol name.
			return mi.Sym.Less(mj.Sym)
		})
	}

	for _, m := range t.Methods().Slice() {
		if m.Sym == nil {
			continue
		}

		CheckSize(m.Type)
		addMethod(m, true)
	}

	for _, m := range t.Methods().Slice() {
		if m.Sym != nil || m.Type == nil {
			continue
		}

		if m.Type.IsUnion() {
			continue
		}

		// In 1.18, embedded types can be anything. In Go 1.17, we disallow
		// embedding anything other than interfaces. This requirement was caught
		// by types2 already, so allow non-interface here.
		if !m.Type.IsInterface() {
			continue
		}

		// Embedded interface: duplicate all methods
		// and add to t's method set.
		for _, t1 := range m.Type.AllMethods().Slice() {
			f := NewField(m.Pos, t1.Sym, t1.Type)
			addMethod(f, false)

			// Clear position after typechecking, for consistency with types2.
			f.Pos = src.NoXPos
		}

		// Clear position after typechecking, for consistency with types2.
		m.Pos = src.NoXPos
	}

	sort.Sort(MethodsByName(methods))

	if int64(len(methods)) >= MaxWidth/int64(PtrSize) {
		base.ErrorfAt(typePos(t), "interface too large")
	}
	for i, m := range methods {
		m.Offset = int64(i) * int64(PtrSize)
	}

	t.SetAllMethods(methods)
}

func calcStructOffset(errtype *Type, t *Type, o int64, flag int) int64 {
	// flag is 0 (receiver), 1 (actual struct), or RegSize (in/out parameters)
	isStruct := flag == 1
	starto := o
	maxalign := int32(flag)
	if maxalign < 1 {
		maxalign = 1
	}
	// Special case: sync/atomic.align64 is an empty struct we recognize
	// as a signal that the struct it contains must be 64-bit-aligned.
	//
	// This logic is duplicated in go/types and cmd/compile/internal/types2.
	if isStruct && t.NumFields() == 0 && t.Sym() != nil && t.Sym().Name == "align64" && isAtomicStdPkg(t.Sym().Pkg) {
		maxalign = 8
	}
	lastzero := int64(0)
	for _, f := range t.Fields().Slice() {
		if f.Type == nil {
			// broken field, just skip it so that other valid fields
			// get a width.
			continue
		}

		CalcSize(f.Type)
		// If type T contains a field F marked as not-in-heap,
		// then T must also be a not-in-heap type. Otherwise,
		// you could heap allocate T and then get a pointer F,
		// which would be a heap pointer to a not-in-heap type.
		if f.Type.NotInHeap() {
			t.SetNotInHeap(true)
		}
		if int32(f.Type.align) > maxalign {
			maxalign = int32(f.Type.align)
		}
		if f.Type.align > 0 {
			o = RoundUp(o, int64(f.Type.align))
		}
		if isStruct { // For receiver/args/results, do not set, it depends on ABI
			f.Offset = o
		}

		w := f.Type.width
		if w < 0 {
			base.Fatalf("invalid width %d", f.Type.width)
		}
		if w == 0 {
			lastzero = o
		}
		o += w
		maxwidth := MaxWidth
		// On 32-bit systems, reflect tables impose an additional constraint
		// that each field start offset must fit in 31 bits.
		if maxwidth < 1<<32 {
			maxwidth = 1<<31 - 1
		}
		if o >= maxwidth {
			base.ErrorfAt(typePos(errtype), "type %L too large", errtype)
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
		o = RoundUp(o, int64(maxalign))
	}
	t.align = uint8(maxalign)

	// type width only includes back to first field's offset
	t.width = o - starto

	return o
}

func isAtomicStdPkg(p *Pkg) bool {
	return (p.Prefix == "sync/atomic" || p.Prefix == `""` && base.Ctxt.Pkgpath == "sync/atomic") ||
		(p.Prefix == "runtime/internal/atomic" || p.Prefix == `""` && base.Ctxt.Pkgpath == "runtime/internal/atomic")
}

// CalcSize calculates and stores the size and alignment for t.
// If CalcSizeDisabled is set, and the size/alignment
// have not already been calculated, it calls Fatal.
// This is used to prevent data races in the back end.
func CalcSize(t *Type) {
	// Calling CalcSize when typecheck tracing enabled is not safe.
	// See issue #33658.
	if base.EnableTrace && SkipSizeForTracing {
		return
	}
	if PtrSize == 0 {
		// Assume this is a test.
		return
	}

	if t == nil {
		return
	}

	if t.width == -2 {
		t.width = 0
		t.align = 1
		base.Fatalf("invalid recursive type %v", t)
		return
	}

	if t.widthCalculated() {
		return
	}

	if CalcSizeDisabled {
		base.Fatalf("width not calculated: %v", t)
	}

	// defer CheckSize calls until after we're done
	DeferCheckSize()

	lno := base.Pos
	if pos := t.Pos(); pos.IsKnown() {
		base.Pos = pos
	}

	t.width = -2
	t.align = 0 // 0 means use t.Width, below

	et := t.Kind()
	switch et {
	case TFUNC, TCHAN, TMAP, TSTRING:
		break

	// SimType == 0 during bootstrap
	default:
		if SimType[t.Kind()] != 0 {
			et = SimType[t.Kind()]
		}
	}

	var w int64
	switch et {
	default:
		base.Fatalf("CalcSize: unknown type: %v", t)

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
		t.align = uint8(RegSize)

	case TCOMPLEX64:
		w = 8
		t.align = 4

	case TCOMPLEX128:
		w = 16
		t.align = uint8(RegSize)

	case TPTR:
		w = int64(PtrSize)
		CheckSize(t.Elem())

	case TUNSAFEPTR:
		w = int64(PtrSize)

	case TINTER: // implemented as 2 pointers
		w = 2 * int64(PtrSize)
		t.align = uint8(PtrSize)
		expandiface(t)

	case TUNION:
		// Always part of an interface for now, so size/align don't matter.
		// Pretend a union is represented like an interface.
		w = 2 * int64(PtrSize)
		t.align = uint8(PtrSize)

	case TCHAN: // implemented as pointer
		w = int64(PtrSize)

		CheckSize(t.Elem())

		// Make fake type to trigger channel element size check after
		// any top-level recursive type has been completed.
		t1 := NewChanArgs(t)
		CheckSize(t1)

	case TCHANARGS:
		t1 := t.ChanArgs()
		CalcSize(t1) // just in case
		// Make sure size of t1.Elem() is calculated at this point. We can
		// use CalcSize() here rather than CheckSize(), because the top-level
		// (possibly recursive) type will have been calculated before the fake
		// chanargs is handled.
		CalcSize(t1.Elem())
		if t1.Elem().width >= 1<<16 {
			base.Errorf("channel element type too large (>64kB)")
		}
		w = 1 // anything will do

	case TMAP: // implemented as pointer
		w = int64(PtrSize)
		CheckSize(t.Elem())
		CheckSize(t.Key())

	case TFORW: // should have been filled in
		base.Fatalf("invalid recursive type %v", t)
		w = 1 // anything will do

	case TANY:
		// not a real type; should be replaced before use.
		base.Fatalf("CalcSize any")

	case TSTRING:
		if StringSize == 0 {
			base.Fatalf("early CalcSize string")
		}
		w = StringSize
		t.align = uint8(PtrSize)

	case TARRAY:
		if t.Elem() == nil {
			break
		}

		CalcSize(t.Elem())
		t.SetNotInHeap(t.Elem().NotInHeap())
		if t.Elem().width != 0 {
			cap := (uint64(MaxWidth) - 1) / uint64(t.Elem().width)
			if uint64(t.NumElem()) > cap {
				base.Errorf("type %L larger than address space", t)
			}
		}
		w = t.NumElem() * t.Elem().width
		t.align = t.Elem().align

	case TSLICE:
		if t.Elem() == nil {
			break
		}
		w = SliceSize
		CheckSize(t.Elem())
		t.align = uint8(PtrSize)

	case TSTRUCT:
		if t.IsFuncArgStruct() {
			base.Fatalf("CalcSize fn struct %v", t)
		}
		// Recognize and mark runtime/internal/sys.nih as not-in-heap.
		if sym := t.Sym(); sym != nil && sym.Pkg.Path == "runtime/internal/sys" && sym.Name == "nih" {
			t.SetNotInHeap(true)
		}
		w = calcStructOffset(t, t, 0, 1)

	// make fake type to check later to
	// trigger function argument computation.
	case TFUNC:
		t1 := NewFuncArgs(t)
		CheckSize(t1)
		w = int64(PtrSize) // width of func type is pointer

	// function is 3 cated structures;
	// compute their widths as side-effect.
	case TFUNCARGS:
		t1 := t.FuncArgs()
		w = calcStructOffset(t1, t1.Recvs(), 0, 0)
		w = calcStructOffset(t1, t1.Params(), w, RegSize)
		w = calcStructOffset(t1, t1.Results(), w, RegSize)
		t1.extra.(*Func).Argwid = w
		if w%int64(RegSize) != 0 {
			base.Warn("bad type %v %d\n", t1, w)
		}
		t.align = 1

	case TTYPEPARAM:
		// TODO(danscales) - remove when we eliminate the need
		// to do CalcSize in noder2 (which shouldn't be needed in the noder)
		w = int64(PtrSize)
	}

	if PtrSize == 4 && w != int64(int32(w)) {
		base.Errorf("type %v too large", t)
	}

	t.width = w
	if t.align == 0 {
		if w == 0 || w > 8 || w&(w-1) != 0 {
			base.Fatalf("invalid alignment for %v", t)
		}
		t.align = uint8(w)
	}

	base.Pos = lno

	ResumeCheckSize()
}

// CalcStructSize calculates the size of s,
// filling in s.Width and s.Align,
// even if size calculation is otherwise disabled.
func CalcStructSize(s *Type) {
	s.width = calcStructOffset(s, s, 0, 1) // sets align
}

// RecalcSize is like CalcSize, but recalculates t's size even if it
// has already been calculated before. It does not recalculate other
// types.
func RecalcSize(t *Type) {
	t.align = 0
	CalcSize(t)
}

func (t *Type) widthCalculated() bool {
	return t.align > 0
}

// when a type's width should be known, we call CheckSize
// to compute it.  during a declaration like
//
//	type T *struct { next T }
//
// it is necessary to defer the calculation of the struct width
// until after T has been initialized to be a pointer to that struct.
// similarly, during import processing structs may be used
// before their definition.  in those situations, calling
// DeferCheckSize() stops width calculations until
// ResumeCheckSize() is called, at which point all the
// CalcSizes that were deferred are executed.
// CalcSize should only be called when the type's size
// is needed immediately.  CheckSize makes sure the
// size is evaluated eventually.

var deferredTypeStack []*Type

func CheckSize(t *Type) {
	if t == nil {
		return
	}

	// function arg structs should not be checked
	// outside of the enclosing function.
	if t.IsFuncArgStruct() {
		base.Fatalf("CheckSize %v", t)
	}

	if defercalc == 0 {
		CalcSize(t)
		return
	}

	// if type has not yet been pushed on deferredTypeStack yet, do it now
	if !t.Deferwidth() {
		t.SetDeferwidth(true)
		deferredTypeStack = append(deferredTypeStack, t)
	}
}

func DeferCheckSize() {
	defercalc++
}

func ResumeCheckSize() {
	if defercalc == 1 {
		for len(deferredTypeStack) > 0 {
			t := deferredTypeStack[len(deferredTypeStack)-1]
			deferredTypeStack = deferredTypeStack[:len(deferredTypeStack)-1]
			t.SetDeferwidth(false)
			CalcSize(t)
		}
	}

	defercalc--
}

// PtrDataSize returns the length in bytes of the prefix of t
// containing pointer data. Anything after this offset is scalar data.
//
// PtrDataSize is only defined for actual Go types. It's an error to
// use it on compiler-internal types (e.g., TSSA, TRESULTS).
func PtrDataSize(t *Type) int64 {
	switch t.Kind() {
	case TBOOL, TINT8, TUINT8, TINT16, TUINT16, TINT32,
		TUINT32, TINT64, TUINT64, TINT, TUINT,
		TUINTPTR, TCOMPLEX64, TCOMPLEX128, TFLOAT32, TFLOAT64:
		return 0

	case TPTR:
		if t.Elem().NotInHeap() {
			return 0
		}
		return int64(PtrSize)

	case TUNSAFEPTR, TFUNC, TCHAN, TMAP:
		return int64(PtrSize)

	case TSTRING:
		// struct { byte *str; intgo len; }
		return int64(PtrSize)

	case TINTER:
		// struct { Itab *tab;	void *data; } or
		// struct { Type *type; void *data; }
		// Note: see comment in typebits.Set
		return 2 * int64(PtrSize)

	case TSLICE:
		if t.Elem().NotInHeap() {
			return 0
		}
		// struct { byte *array; uintgo len; uintgo cap; }
		return int64(PtrSize)

	case TARRAY:
		if t.NumElem() == 0 {
			return 0
		}
		// t.NumElem() > 0
		size := PtrDataSize(t.Elem())
		if size == 0 {
			return 0
		}
		return (t.NumElem()-1)*t.Elem().Size() + size

	case TSTRUCT:
		// Find the last field that has pointers, if any.
		fs := t.Fields().Slice()
		for i := len(fs) - 1; i >= 0; i-- {
			if size := PtrDataSize(fs[i].Type); size > 0 {
				return fs[i].Offset + size
			}
		}
		return 0

	case TSSA:
		if t != TypeInt128 {
			base.Fatalf("PtrDataSize: unexpected ssa type %v", t)
		}
		return 0

	default:
		base.Fatalf("PtrDataSize: unexpected type, %v", t)
		return 0
	}
}
