// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "fmt"

// AlgKind describes the kind of algorithms used for comparing and
// hashing a Type.
type AlgKind int

const (
	// These values are known by runtime.
	ANOEQ AlgKind = iota
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

	// Type can be compared/hashed as regular memory.
	AMEM AlgKind = 100

	// Type needs special comparison/hashing functions.
	ASPECIAL AlgKind = -1
)

// IsComparable reports whether t is a comparable type.
func (t *Type) IsComparable() bool {
	a, _ := algtype1(t)
	return a != ANOEQ
}

// IsRegularMemory reports whether t can be compared/hashed as regular memory.
func (t *Type) IsRegularMemory() bool {
	a, _ := algtype1(t)
	return a == AMEM
}

// IncomparableField returns an incomparable Field of struct Type t, if any.
func (t *Type) IncomparableField() *Field {
	for _, f := range t.FieldSlice() {
		if !f.Type.IsComparable() {
			return f
		}
	}
	return nil
}

// algtype is like algtype1, except it returns the fixed-width AMEMxx variants
// instead of the general AMEM kind when possible.
func algtype(t *Type) AlgKind {
	a, _ := algtype1(t)
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

// algtype1 returns the AlgKind used for comparing and hashing Type t.
// If it returns ANOEQ, it also returns the component type of t that
// makes it incomparable.
func algtype1(t *Type) (AlgKind, *Type) {
	if t.Broke {
		return AMEM, nil
	}
	if t.Noalg {
		return ANOEQ, t
	}

	switch t.Etype {
	case TANY, TFORW:
		// will be defined later.
		return ANOEQ, t

	case TINT8, TUINT8, TINT16, TUINT16,
		TINT32, TUINT32, TINT64, TUINT64,
		TINT, TUINT, TUINTPTR,
		TBOOL, TPTR32, TPTR64,
		TCHAN, TUNSAFEPTR:
		return AMEM, nil

	case TFUNC, TMAP:
		return ANOEQ, t

	case TFLOAT32:
		return AFLOAT32, nil

	case TFLOAT64:
		return AFLOAT64, nil

	case TCOMPLEX64:
		return ACPLX64, nil

	case TCOMPLEX128:
		return ACPLX128, nil

	case TSTRING:
		return ASTRING, nil

	case TINTER:
		if t.IsEmptyInterface() {
			return ANILINTER, nil
		}
		return AINTER, nil

	case TSLICE:
		return ANOEQ, t

	case TARRAY:
		a, bad := algtype1(t.Elem())
		switch a {
		case AMEM:
			return AMEM, nil
		case ANOEQ:
			return ANOEQ, bad
		}

		switch t.NumElem() {
		case 0:
			// We checked above that the element type is comparable.
			return AMEM, nil
		case 1:
			// Single-element array is same as its lone element.
			return a, nil
		}

		return ASPECIAL, nil

	case TSTRUCT:
		fields := t.FieldSlice()

		// One-field struct is same as that one field alone.
		if len(fields) == 1 && !isblanksym(fields[0].Sym) {
			return algtype1(fields[0].Type)
		}

		ret := AMEM
		for i, f := range fields {
			// All fields must be comparable.
			a, bad := algtype1(f.Type)
			if a == ANOEQ {
				return ANOEQ, bad
			}

			// Blank fields, padded fields, fields with non-memory
			// equality need special compare.
			if a != AMEM || isblanksym(f.Sym) || ispaddedfield(t, i) {
				ret = ASPECIAL
			}
		}

		return ret, nil
	}

	Fatalf("algtype1: unexpected type %v", t)
	return 0, nil
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
	tfn.List.Append(n)
	np := n.Left
	n = Nod(ODCLFIELD, newname(Lookup("h")), typenod(Types[TUINTPTR]))
	tfn.List.Append(n)
	nh := n.Left
	n = Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])) // return value
	tfn.Rlist.Append(n)

	funchdr(fn)
	fn.Func.Nname.Name.Param.Ntype = typecheck(fn.Func.Nname.Name.Param.Ntype, Etype)

	// genhash is only called for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch t.Etype {
	default:
		Fatalf("genhash %v", t)

	case TARRAY:
		// An array of pure memory would be handled by the
		// standard algorithm, so the element type must not be
		// pure memory.
		hashel := hashfor(t.Elem())

		n := Nod(ORANGE, nil, Nod(OIND, np, nil))
		ni := newname(Lookup("i"))
		ni.Type = Types[TINT]
		n.List.Set1(ni)
		n.Colas = true
		colasdefn(n.List.Slice(), n)
		ni = n.List.First()

		// h = hashel(&p[i], h)
		call := Nod(OCALL, hashel, nil)

		nx := Nod(OINDEX, np, ni)
		nx.Bounded = true
		na := Nod(OADDR, nx, nil)
		na.Etype = 1 // no escape to heap
		call.List.Append(na)
		call.List.Append(nh)
		n.Nbody.Append(Nod(OAS, nh, call))

		fn.Nbody.Append(n)

	case TSTRUCT:
		// Walk the struct using memhash for runs of AMEM
		// and calling specific hash functions for the others.
		for i, fields := 0, t.FieldSlice(); i < len(fields); {
			f := fields[i]

			// Skip blank fields.
			if isblanksym(f.Sym) {
				i++
				continue
			}

			// Hash non-memory fields with appropriate hash function.
			if !f.Type.IsRegularMemory() {
				hashel := hashfor(f.Type)
				call := Nod(OCALL, hashel, nil)
				nx := NodSym(OXDOT, np, f.Sym) // TODO: fields from other packages?
				na := Nod(OADDR, nx, nil)
				na.Etype = 1 // no escape to heap
				call.List.Append(na)
				call.List.Append(nh)
				fn.Nbody.Append(Nod(OAS, nh, call))
				i++
				continue
			}

			// Otherwise, hash a maximal length run of raw memory.
			size, next := memrun(t, i)

			// h = hashel(&p.first, size, h)
			hashel := hashmem(f.Type)
			call := Nod(OCALL, hashel, nil)
			nx := NodSym(OXDOT, np, f.Sym) // TODO: fields from other packages?
			na := Nod(OADDR, nx, nil)
			na.Etype = 1 // no escape to heap
			call.List.Append(na)
			call.List.Append(nh)
			call.List.Append(Nodintconst(size))
			fn.Nbody.Append(Nod(OAS, nh, call))

			i = next
		}
	}

	r := Nod(ORETURN, nil, nil)
	r.List.Append(nh)
	fn.Nbody.Append(r)

	if Debug['r'] != 0 {
		dumplist("genhash body", fn.Nbody)
	}

	funcbody(fn)
	Curfn = fn
	fn.Func.Dupok = true
	fn = typecheck(fn, Etop)
	typecheckslice(fn.Nbody.Slice(), Etop)
	Curfn = nil
	popdcl()
	testdclstack()

	// Disable safemode while compiling this code: the code we
	// generate internally can refer to unsafe.Pointer.
	// In this case it can happen if we need to generate an ==
	// for a struct containing a reflect.Value, which itself has
	// an unexported field of type unsafe.Pointer.
	old_safemode := safemode
	safemode = false

	Disable_checknil++
	funccompile(fn)
	Disable_checknil--

	safemode = old_safemode
}

func hashfor(t *Type) *Node {
	var sym *Sym

	switch a, _ := algtype1(t); a {
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
	tfn.List.Append(Nod(ODCLFIELD, nil, typenod(Ptrto(t))))
	tfn.List.Append(Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	tfn.Rlist.Append(Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	tfn = typecheck(tfn, Etype)
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
	tfn.List.Append(n)
	np := n.Left
	n = Nod(ODCLFIELD, newname(Lookup("q")), typenod(Ptrto(t)))
	tfn.List.Append(n)
	nq := n.Left
	n = Nod(ODCLFIELD, nil, typenod(Types[TBOOL]))
	tfn.Rlist.Append(n)

	funchdr(fn)
	fn.Func.Nname.Name.Param.Ntype = typecheck(fn.Func.Nname.Name.Param.Ntype, Etype)

	// geneq is only called for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch t.Etype {
	default:
		Fatalf("geneq %v", t)

	case TARRAY:
		// An array of pure memory would be handled by the
		// standard memequal, so the element type must not be
		// pure memory. Even if we unrolled the range loop,
		// each iteration would be a function call, so don't bother
		// unrolling.
		nrange := Nod(ORANGE, nil, Nod(OIND, np, nil))

		ni := newname(Lookup("i"))
		ni.Type = Types[TINT]
		nrange.List.Set1(ni)
		nrange.Colas = true
		colasdefn(nrange.List.Slice(), nrange)
		ni = nrange.List.First()

		// if p[i] != q[i] { return false }
		nx := Nod(OINDEX, np, ni)

		nx.Bounded = true
		ny := Nod(OINDEX, nq, ni)
		ny.Bounded = true

		nif := Nod(OIF, nil, nil)
		nif.Left = Nod(ONE, nx, ny)
		r := Nod(ORETURN, nil, nil)
		r.List.Append(Nodbool(false))
		nif.Nbody.Append(r)
		nrange.Nbody.Append(nif)
		fn.Nbody.Append(nrange)

		// return true
		ret := Nod(ORETURN, nil, nil)
		ret.List.Append(Nodbool(true))
		fn.Nbody.Append(ret)

	case TSTRUCT:
		var cond *Node
		and := func(n *Node) {
			if cond == nil {
				cond = n
				return
			}
			cond = Nod(OANDAND, cond, n)
		}

		// Walk the struct using memequal for runs of AMEM
		// and calling specific equality tests for the others.
		for i, fields := 0, t.FieldSlice(); i < len(fields); {
			f := fields[i]

			// Skip blank-named fields.
			if isblanksym(f.Sym) {
				i++
				continue
			}

			// Compare non-memory fields with field equality.
			if !f.Type.IsRegularMemory() {
				and(eqfield(np, nq, f.Sym))
				i++
				continue
			}

			// Find maximal length run of memory-only fields.
			size, next := memrun(t, i)

			// TODO(rsc): All the calls to newname are wrong for
			// cross-package unexported fields.
			if s := fields[i:next]; len(s) <= 2 {
				// Two or fewer fields: use plain field equality.
				for _, f := range s {
					and(eqfield(np, nq, f.Sym))
				}
			} else {
				// More than two fields: use memequal.
				and(eqmem(np, nq, f.Sym, size))
			}
			i = next
		}

		if cond == nil {
			cond = Nodbool(true)
		}

		ret := Nod(ORETURN, nil, nil)
		ret.List.Append(cond)
		fn.Nbody.Append(ret)
	}

	if Debug['r'] != 0 {
		dumplist("geneq body", fn.Nbody)
	}

	funcbody(fn)
	Curfn = fn
	fn.Func.Dupok = true
	fn = typecheck(fn, Etop)
	typecheckslice(fn.Nbody.Slice(), Etop)
	Curfn = nil
	popdcl()
	testdclstack()

	// Disable safemode while compiling this code: the code we
	// generate internally can refer to unsafe.Pointer.
	// In this case it can happen if we need to generate an ==
	// for a struct containing a reflect.Value, which itself has
	// an unexported field of type unsafe.Pointer.
	old_safemode := safemode
	safemode = false

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
func eqfield(p *Node, q *Node, field *Sym) *Node {
	nx := NodSym(OXDOT, p, field)
	ny := NodSym(OXDOT, q, field)
	ne := Nod(OEQ, nx, ny)
	return ne
}

// eqmem returns the node
// 	memequal(&p.field, &q.field [, size])
func eqmem(p *Node, q *Node, field *Sym, size int64) *Node {
	nx := Nod(OADDR, NodSym(OXDOT, p, field), nil)
	nx.Etype = 1 // does not escape
	ny := Nod(OADDR, NodSym(OXDOT, q, field), nil)
	ny.Etype = 1 // does not escape
	nx = typecheck(nx, Erv)
	ny = typecheck(ny, Erv)

	fn, needsize := eqmemfunc(size, nx.Type.Elem())
	call := Nod(OCALL, fn, nil)
	call.List.Append(nx)
	call.List.Append(ny)
	if needsize {
		call.List.Append(Nodintconst(size))
	}

	return call
}

func eqmemfunc(size int64, t *Type) (fn *Node, needsize bool) {
	switch size {
	default:
		fn = syslook("memequal")
		needsize = true
	case 1, 2, 4, 8, 16:
		buf := fmt.Sprintf("memequal%d", int(size)*8)
		fn = syslook(buf)
	}

	fn = substArgTypes(fn, t, t)
	return fn, needsize
}

// memrun finds runs of struct fields for which memory-only algs are appropriate.
// t is the parent struct type, and start is the field index at which to start the run.
// size is the length in bytes of the memory included in the run.
// next is the index just after the end of the memory run.
func memrun(t *Type, start int) (size int64, next int) {
	next = start
	for {
		next++
		if next == t.NumFields() {
			break
		}
		// Stop run after a padded field.
		if ispaddedfield(t, next-1) {
			break
		}
		// Also, stop before a blank or non-memory field.
		if f := t.Field(next); isblanksym(f.Sym) || !f.Type.IsRegularMemory() {
			break
		}
	}
	return t.Field(next-1).End() - t.Field(start).Offset, next
}

// ispaddedfield reports whether the i'th field of struct type t is followed
// by padding.
func ispaddedfield(t *Type, i int) bool {
	if !t.IsStruct() {
		Fatalf("ispaddedfield called non-struct %v", t)
	}
	end := t.Width
	if i+1 < t.NumFields() {
		end = t.Field(i + 1).Offset
	}
	return t.Field(i).End() != end
}
