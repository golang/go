// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "fmt"

const (
	// These values are known by runtime.
	ANOEQ = iota
	AMEM0
	AMEM8
	AMEM16
	AMEM32
	AMEM64
	AMEM128
	ASTRING
	AINTER
	ANILINTER
	AFLOAT32
	AFLOAT64
	ACPLX64
	ACPLX128
	AMEM = 100
)

func algtype(t *Type) int {
	a := algtype1(t, nil)
	if a == AMEM {
		switch t.Width {
		case 0:
			return AMEM0
		case 1:
			return AMEM8
		case 2:
			return AMEM16
		case 4:
			return AMEM32
		case 8:
			return AMEM64
		case 16:
			return AMEM128
		}
	}

	return a
}

func algtype1(t *Type, bad **Type) int {
	if bad != nil {
		*bad = nil
	}
	if t.Broke {
		return AMEM
	}
	if t.Noalg {
		return ANOEQ
	}

	switch t.Etype {
	// will be defined later.
	case TANY, TFORW:
		*bad = t

		return -1

	case TINT8,
		TUINT8,
		TINT16,
		TUINT16,
		TINT32,
		TUINT32,
		TINT64,
		TUINT64,
		TINT,
		TUINT,
		TUINTPTR,
		TBOOL,
		TPTR32,
		TPTR64,
		TCHAN,
		TUNSAFEPTR:
		return AMEM

	case TFUNC, TMAP:
		if bad != nil {
			*bad = t
		}
		return ANOEQ

	case TFLOAT32:
		return AFLOAT32

	case TFLOAT64:
		return AFLOAT64

	case TCOMPLEX64:
		return ACPLX64

	case TCOMPLEX128:
		return ACPLX128

	case TSTRING:
		return ASTRING

	case TINTER:
		if isnilinter(t) {
			return ANILINTER
		}
		return AINTER

	case TARRAY:
		if Isslice(t) {
			if bad != nil {
				*bad = t
			}
			return ANOEQ
		}

		a := algtype1(t.Type, bad)
		if a == ANOEQ || a == AMEM {
			if a == ANOEQ && bad != nil {
				*bad = t
			}
			return a
		}

		switch t.Bound {
		case 0:
			// We checked above that the element type is comparable.
			return AMEM
		case 1:
			// Single-element array is same as its lone element.
			return a
		}

		return -1 // needs special compare

	case TSTRUCT:
		if t.Type != nil && t.Type.Down == nil && !isblanksym(t.Type.Sym) {
			// One-field struct is same as that one field alone.
			return algtype1(t.Type.Type, bad)
		}

		ret := AMEM
		var a int
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			// All fields must be comparable.
			a = algtype1(t1.Type, bad)

			if a == ANOEQ {
				return ANOEQ
			}

			// Blank fields, padded fields, fields with non-memory
			// equality need special compare.
			if a != AMEM || isblanksym(t1.Sym) || ispaddedfield(t1, t.Width) {
				ret = -1
				continue
			}
		}

		return ret
	}

	Fatalf("algtype1: unexpected type %v", t)
	return 0
}

// Generate a helper function to compute the hash of a value of type t.
func genhash(sym *Sym, t *Type) {
	if Debug['r'] != 0 {
		fmt.Printf("genhash %v %v\n", sym, t)
	}

	lineno = 1 // less confusing than end of input
	dclcontext = PEXTERN
	markdcl()

	// func sym(p *T, h uintptr) uintptr
	fn := Nod(ODCLFUNC, nil, nil)

	fn.Func.Nname = newname(sym)
	fn.Func.Nname.Class = PFUNC
	tfn := Nod(OTFUNC, nil, nil)
	fn.Func.Nname.Name.Param.Ntype = tfn

	n := Nod(ODCLFIELD, newname(Lookup("p")), typenod(Ptrto(t)))
	tfn.List = list(tfn.List, n)
	np := n.Left
	n = Nod(ODCLFIELD, newname(Lookup("h")), typenod(Types[TUINTPTR]))
	tfn.List = list(tfn.List, n)
	nh := n.Left
	n = Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])) // return value
	tfn.Rlist = list(tfn.Rlist, n)

	funchdr(fn)
	typecheck(&fn.Func.Nname.Name.Param.Ntype, Etype)

	// genhash is only called for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch t.Etype {
	default:
		Fatalf("genhash %v", t)

	case TARRAY:
		if Isslice(t) {
			Fatalf("genhash %v", t)
		}

		// An array of pure memory would be handled by the
		// standard algorithm, so the element type must not be
		// pure memory.
		hashel := hashfor(t.Type)

		n := Nod(ORANGE, nil, Nod(OIND, np, nil))
		ni := newname(Lookup("i"))
		ni.Type = Types[TINT]
		n.List = list1(ni)
		n.Colas = true
		colasdefn(n.List, n)
		ni = n.List.N

		// h = hashel(&p[i], h)
		call := Nod(OCALL, hashel, nil)

		nx := Nod(OINDEX, np, ni)
		nx.Bounded = true
		na := Nod(OADDR, nx, nil)
		na.Etype = 1 // no escape to heap
		call.List = list(call.List, na)
		call.List = list(call.List, nh)
		n.Nbody = list(n.Nbody, Nod(OAS, nh, call))

		fn.Nbody = list(fn.Nbody, n)

	// Walk the struct using memhash for runs of AMEM
	// and calling specific hash functions for the others.
	case TSTRUCT:
		var call *Node
		var nx *Node
		var na *Node
		var hashel *Node

		t1 := t.Type
		for {
			first, size, next := memrun(t, t1)
			t1 = next

			// Run memhash for fields up to this one.
			if first != nil {
				hashel = hashmem(first.Type)

				// h = hashel(&p.first, size, h)
				call = Nod(OCALL, hashel, nil)

				nx = Nod(OXDOT, np, newname(first.Sym)) // TODO: fields from other packages?
				na = Nod(OADDR, nx, nil)
				na.Etype = 1 // no escape to heap
				call.List = list(call.List, na)
				call.List = list(call.List, nh)
				call.List = list(call.List, Nodintconst(size))
				fn.Nbody = list(fn.Nbody, Nod(OAS, nh, call))
			}

			if t1 == nil {
				break
			}
			if isblanksym(t1.Sym) {
				t1 = t1.Down
				continue
			}
			if algtype1(t1.Type, nil) == AMEM {
				// Our memory run might have been stopped by padding or a blank field.
				// If the next field is memory-ish, it could be the start of a new run.
				continue
			}

			hashel = hashfor(t1.Type)
			call = Nod(OCALL, hashel, nil)
			nx = Nod(OXDOT, np, newname(t1.Sym)) // TODO: fields from other packages?
			na = Nod(OADDR, nx, nil)
			na.Etype = 1 // no escape to heap
			call.List = list(call.List, na)
			call.List = list(call.List, nh)
			fn.Nbody = list(fn.Nbody, Nod(OAS, nh, call))

			t1 = t1.Down
		}
	}

	r := Nod(ORETURN, nil, nil)
	r.List = list(r.List, nh)
	fn.Nbody = list(fn.Nbody, r)

	if Debug['r'] != 0 {
		dumplist("genhash body", fn.Nbody)
	}

	funcbody(fn)
	Curfn = fn
	fn.Func.Dupok = true
	typecheck(&fn, Etop)
	typechecklist(fn.Nbody, Etop)
	Curfn = nil

	// Disable safemode while compiling this code: the code we
	// generate internally can refer to unsafe.Pointer.
	// In this case it can happen if we need to generate an ==
	// for a struct containing a reflect.Value, which itself has
	// an unexported field of type unsafe.Pointer.
	old_safemode := safemode

	safemode = 0
	funccompile(fn)
	safemode = old_safemode
}

func hashfor(t *Type) *Node {
	var sym *Sym

	a := algtype1(t, nil)
	switch a {
	case AMEM:
		Fatalf("hashfor with AMEM type")

	case AINTER:
		sym = Pkglookup("interhash", Runtimepkg)

	case ANILINTER:
		sym = Pkglookup("nilinterhash", Runtimepkg)

	case ASTRING:
		sym = Pkglookup("strhash", Runtimepkg)

	case AFLOAT32:
		sym = Pkglookup("f32hash", Runtimepkg)

	case AFLOAT64:
		sym = Pkglookup("f64hash", Runtimepkg)

	case ACPLX64:
		sym = Pkglookup("c64hash", Runtimepkg)

	case ACPLX128:
		sym = Pkglookup("c128hash", Runtimepkg)

	default:
		sym = typesymprefix(".hash", t)
	}

	n := newname(sym)
	n.Class = PFUNC
	tfn := Nod(OTFUNC, nil, nil)
	tfn.List = list(tfn.List, Nod(ODCLFIELD, nil, typenod(Ptrto(t))))
	tfn.List = list(tfn.List, Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	tfn.Rlist = list(tfn.Rlist, Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	typecheck(&tfn, Etype)
	n.Type = tfn.Type
	return n
}

// geneq generates a helper function to
// check equality of two values of type t.
func geneq(sym *Sym, t *Type) {
	if Debug['r'] != 0 {
		fmt.Printf("geneq %v %v\n", sym, t)
	}

	lineno = 1 // less confusing than end of input
	dclcontext = PEXTERN
	markdcl()

	// func sym(p, q *T) bool
	fn := Nod(ODCLFUNC, nil, nil)

	fn.Func.Nname = newname(sym)
	fn.Func.Nname.Class = PFUNC
	tfn := Nod(OTFUNC, nil, nil)
	fn.Func.Nname.Name.Param.Ntype = tfn

	n := Nod(ODCLFIELD, newname(Lookup("p")), typenod(Ptrto(t)))
	tfn.List = list(tfn.List, n)
	np := n.Left
	n = Nod(ODCLFIELD, newname(Lookup("q")), typenod(Ptrto(t)))
	tfn.List = list(tfn.List, n)
	nq := n.Left
	n = Nod(ODCLFIELD, nil, typenod(Types[TBOOL]))
	tfn.Rlist = list(tfn.Rlist, n)

	funchdr(fn)

	// geneq is only called for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch t.Etype {
	default:
		Fatalf("geneq %v", t)

	case TARRAY:
		if Isslice(t) {
			Fatalf("geneq %v", t)
		}

		// An array of pure memory would be handled by the
		// standard memequal, so the element type must not be
		// pure memory.  Even if we unrolled the range loop,
		// each iteration would be a function call, so don't bother
		// unrolling.
		nrange := Nod(ORANGE, nil, Nod(OIND, np, nil))

		ni := newname(Lookup("i"))
		ni.Type = Types[TINT]
		nrange.List = list1(ni)
		nrange.Colas = true
		colasdefn(nrange.List, nrange)
		ni = nrange.List.N

		// if p[i] != q[i] { return false }
		nx := Nod(OINDEX, np, ni)

		nx.Bounded = true
		ny := Nod(OINDEX, nq, ni)
		ny.Bounded = true

		nif := Nod(OIF, nil, nil)
		nif.Left = Nod(ONE, nx, ny)
		r := Nod(ORETURN, nil, nil)
		r.List = list(r.List, Nodbool(false))
		nif.Nbody = list(nif.Nbody, r)
		nrange.Nbody = list(nrange.Nbody, nif)
		fn.Nbody = list(fn.Nbody, nrange)

		// return true
		ret := Nod(ORETURN, nil, nil)
		ret.List = list(ret.List, Nodbool(true))
		fn.Nbody = list(fn.Nbody, ret)

	// Walk the struct using memequal for runs of AMEM
	// and calling specific equality tests for the others.
	// Skip blank-named fields.
	case TSTRUCT:
		var conjuncts []*Node

		t1 := t.Type
		for {
			first, size, next := memrun(t, t1)
			t1 = next

			// Run memequal for fields up to this one.
			// TODO(rsc): All the calls to newname are wrong for
			// cross-package unexported fields.
			if first != nil {
				if first.Down == t1 {
					conjuncts = append(conjuncts, eqfield(np, nq, newname(first.Sym)))
				} else if first.Down.Down == t1 {
					conjuncts = append(conjuncts, eqfield(np, nq, newname(first.Sym)))
					first = first.Down
					if !isblanksym(first.Sym) {
						conjuncts = append(conjuncts, eqfield(np, nq, newname(first.Sym)))
					}
				} else {
					// More than two fields: use memequal.
					conjuncts = append(conjuncts, eqmem(np, nq, newname(first.Sym), size))
				}
			}

			if t1 == nil {
				break
			}
			if isblanksym(t1.Sym) {
				t1 = t1.Down
				continue
			}
			if algtype1(t1.Type, nil) == AMEM {
				// Our memory run might have been stopped by padding or a blank field.
				// If the next field is memory-ish, it could be the start of a new run.
				continue
			}

			// Check this field, which is not just memory.
			conjuncts = append(conjuncts, eqfield(np, nq, newname(t1.Sym)))
			t1 = t1.Down
		}

		var and *Node
		switch len(conjuncts) {
		case 0:
			and = Nodbool(true)
		case 1:
			and = conjuncts[0]
		default:
			and = Nod(OANDAND, conjuncts[0], conjuncts[1])
			for _, conjunct := range conjuncts[2:] {
				and = Nod(OANDAND, and, conjunct)
			}
		}

		ret := Nod(ORETURN, nil, nil)
		ret.List = list(ret.List, and)
		fn.Nbody = list(fn.Nbody, ret)
	}

	if Debug['r'] != 0 {
		dumplist("geneq body", fn.Nbody)
	}

	funcbody(fn)
	Curfn = fn
	fn.Func.Dupok = true
	typecheck(&fn, Etop)
	typechecklist(fn.Nbody, Etop)
	Curfn = nil

	// Disable safemode while compiling this code: the code we
	// generate internally can refer to unsafe.Pointer.
	// In this case it can happen if we need to generate an ==
	// for a struct containing a reflect.Value, which itself has
	// an unexported field of type unsafe.Pointer.
	old_safemode := safemode
	safemode = 0

	// Disable checknils while compiling this code.
	// We are comparing a struct or an array,
	// neither of which can be nil, and our comparisons
	// are shallow.
	Disable_checknil++

	funccompile(fn)

	safemode = old_safemode
	Disable_checknil--
}

// eqfield returns the node
// 	p.field == q.field
func eqfield(p *Node, q *Node, field *Node) *Node {
	nx := Nod(OXDOT, p, field)
	ny := Nod(OXDOT, q, field)
	ne := Nod(OEQ, nx, ny)
	return ne
}

// eqmem returns the node
// 	memequal(&p.field, &q.field [, size])
func eqmem(p *Node, q *Node, field *Node, size int64) *Node {
	var needsize int

	nx := Nod(OADDR, Nod(OXDOT, p, field), nil)
	nx.Etype = 1 // does not escape
	ny := Nod(OADDR, Nod(OXDOT, q, field), nil)
	ny.Etype = 1 // does not escape
	typecheck(&nx, Erv)
	typecheck(&ny, Erv)

	call := Nod(OCALL, eqmemfunc(size, nx.Type.Type, &needsize), nil)
	call.List = list(call.List, nx)
	call.List = list(call.List, ny)
	if needsize != 0 {
		call.List = list(call.List, Nodintconst(size))
	}

	return call
}

func eqmemfunc(size int64, type_ *Type, needsize *int) *Node {
	var fn *Node

	switch size {
	default:
		fn = syslook("memequal", 1)
		*needsize = 1

	case 1, 2, 4, 8, 16:
		buf := fmt.Sprintf("memequal%d", int(size)*8)
		fn = syslook(buf, 1)
		*needsize = 0
	}

	substArgTypes(fn, type_, type_)
	return fn
}

// memrun finds runs of struct fields for which memory-only algs are appropriate.
// t is the parent struct type, and field is the field at which to start.
// first is the first field in the memory run.
// size is the length in bytes of the memory included in the run.
// next is the next field after the memory run.
func memrun(t *Type, field *Type) (first *Type, size int64, next *Type) {
	var offend int64
	for {
		if field == nil || algtype1(field.Type, nil) != AMEM || isblanksym(field.Sym) {
			break
		}
		offend = field.Width + field.Type.Width
		if first == nil {
			first = field
		}

		// If it's a memory field but it's padded, stop here.
		if ispaddedfield(field, t.Width) {
			field = field.Down
			break
		}
		field = field.Down
	}
	if first != nil {
		size = offend - first.Width // first.Width is offset
	}
	return first, size, field
}

// ispaddedfield reports whether the given field
// is followed by padding. For the case where t is
// the last field, total gives the size of the enclosing struct.
func ispaddedfield(t *Type, total int64) bool {
	if t.Etype != TFIELD {
		Fatalf("ispaddedfield called non-field %v", t)
	}
	if t.Down == nil {
		return t.Width+t.Type.Width != total
	}
	return t.Width+t.Type.Width != t.Down.Width
}
