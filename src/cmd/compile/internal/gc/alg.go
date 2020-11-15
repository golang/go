// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"fmt"
	"sort"
)

// AlgKind describes the kind of algorithms used for comparing and
// hashing a Type.
type AlgKind int

//go:generate stringer -type AlgKind -trimprefix A

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
func IsComparable(t *types.Type) bool {
	a, _ := algtype1(t)
	return a != ANOEQ
}

// IsRegularMemory reports whether t can be compared/hashed as regular memory.
func IsRegularMemory(t *types.Type) bool {
	a, _ := algtype1(t)
	return a == AMEM
}

// IncomparableField returns an incomparable Field of struct Type t, if any.
func IncomparableField(t *types.Type) *types.Field {
	for _, f := range t.FieldSlice() {
		if !IsComparable(f.Type) {
			return f
		}
	}
	return nil
}

// EqCanPanic reports whether == on type t could panic (has an interface somewhere).
// t must be comparable.
func EqCanPanic(t *types.Type) bool {
	switch t.Etype {
	default:
		return false
	case TINTER:
		return true
	case TARRAY:
		return EqCanPanic(t.Elem())
	case TSTRUCT:
		for _, f := range t.FieldSlice() {
			if !f.Sym.IsBlank() && EqCanPanic(f.Type) {
				return true
			}
		}
		return false
	}
}

// algtype is like algtype1, except it returns the fixed-width AMEMxx variants
// instead of the general AMEM kind when possible.
func algtype(t *types.Type) AlgKind {
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
func algtype1(t *types.Type) (AlgKind, *types.Type) {
	if t.Broke() {
		return AMEM, nil
	}
	if t.Noalg() {
		return ANOEQ, t
	}

	switch t.Etype {
	case TANY, TFORW:
		// will be defined later.
		return ANOEQ, t

	case TINT8, TUINT8, TINT16, TUINT16,
		TINT32, TUINT32, TINT64, TUINT64,
		TINT, TUINT, TUINTPTR,
		TBOOL, TPTR,
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
		if len(fields) == 1 && !fields[0].Sym.IsBlank() {
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
			if a != AMEM || f.Sym.IsBlank() || ispaddedfield(t, i) {
				ret = ASPECIAL
			}
		}

		return ret, nil
	}

	Fatalf("algtype1: unexpected type %v", t)
	return 0, nil
}

// genhash returns a symbol which is the closure used to compute
// the hash of a value of type t.
// Note: the generated function must match runtime.typehash exactly.
func genhash(t *types.Type) *obj.LSym {
	switch algtype(t) {
	default:
		// genhash is only called for types that have equality
		Fatalf("genhash %v", t)
	case AMEM0:
		return sysClosure("memhash0")
	case AMEM8:
		return sysClosure("memhash8")
	case AMEM16:
		return sysClosure("memhash16")
	case AMEM32:
		return sysClosure("memhash32")
	case AMEM64:
		return sysClosure("memhash64")
	case AMEM128:
		return sysClosure("memhash128")
	case ASTRING:
		return sysClosure("strhash")
	case AINTER:
		return sysClosure("interhash")
	case ANILINTER:
		return sysClosure("nilinterhash")
	case AFLOAT32:
		return sysClosure("f32hash")
	case AFLOAT64:
		return sysClosure("f64hash")
	case ACPLX64:
		return sysClosure("c64hash")
	case ACPLX128:
		return sysClosure("c128hash")
	case AMEM:
		// For other sizes of plain memory, we build a closure
		// that calls memhash_varlen. The size of the memory is
		// encoded in the first slot of the closure.
		closure := typeLookup(fmt.Sprintf(".hashfunc%d", t.Width)).Linksym()
		if len(closure.P) > 0 { // already generated
			return closure
		}
		if memhashvarlen == nil {
			memhashvarlen = sysfunc("memhash_varlen")
		}
		ot := 0
		ot = dsymptr(closure, ot, memhashvarlen, 0)
		ot = duintptr(closure, ot, uint64(t.Width)) // size encoded in closure
		ggloblsym(closure, int32(ot), obj.DUPOK|obj.RODATA)
		return closure
	case ASPECIAL:
		break
	}

	closure := typesymprefix(".hashfunc", t).Linksym()
	if len(closure.P) > 0 { // already generated
		return closure
	}

	// Generate hash functions for subtypes.
	// There are cases where we might not use these hashes,
	// but in that case they will get dead-code eliminated.
	// (And the closure generated by genhash will also get
	// dead-code eliminated, as we call the subtype hashers
	// directly.)
	switch t.Etype {
	case types.TARRAY:
		genhash(t.Elem())
	case types.TSTRUCT:
		for _, f := range t.FieldSlice() {
			genhash(f.Type)
		}
	}

	sym := typesymprefix(".hash", t)
	if Debug.r != 0 {
		fmt.Printf("genhash %v %v %v\n", closure, sym, t)
	}

	lineno = autogeneratedPos // less confusing than end of input
	dclcontext = PEXTERN

	// func sym(p *T, h uintptr) uintptr
	tfn := nod(OTFUNC, nil, nil)
	tfn.List.Set2(
		namedfield("p", types.NewPtr(t)),
		namedfield("h", types.Types[TUINTPTR]),
	)
	tfn.Rlist.Set1(anonfield(types.Types[TUINTPTR]))

	fn := dclfunc(sym, tfn)
	np := asNode(tfn.Type.Params().Field(0).Nname)
	nh := asNode(tfn.Type.Params().Field(1).Nname)

	switch t.Etype {
	case types.TARRAY:
		// An array of pure memory would be handled by the
		// standard algorithm, so the element type must not be
		// pure memory.
		hashel := hashfor(t.Elem())

		n := nod(ORANGE, nil, nod(ODEREF, np, nil))
		ni := newname(lookup("i"))
		ni.Type = types.Types[TINT]
		n.List.Set1(ni)
		n.SetColas(true)
		colasdefn(n.List.Slice(), n)
		ni = n.List.First()

		// h = hashel(&p[i], h)
		call := nod(OCALL, hashel, nil)

		nx := nod(OINDEX, np, ni)
		nx.SetBounded(true)
		na := nod(OADDR, nx, nil)
		call.List.Append(na)
		call.List.Append(nh)
		n.Nbody.Append(nod(OAS, nh, call))

		fn.Nbody.Append(n)

	case types.TSTRUCT:
		// Walk the struct using memhash for runs of AMEM
		// and calling specific hash functions for the others.
		for i, fields := 0, t.FieldSlice(); i < len(fields); {
			f := fields[i]

			// Skip blank fields.
			if f.Sym.IsBlank() {
				i++
				continue
			}

			// Hash non-memory fields with appropriate hash function.
			if !IsRegularMemory(f.Type) {
				hashel := hashfor(f.Type)
				call := nod(OCALL, hashel, nil)
				nx := nodSym(OXDOT, np, f.Sym) // TODO: fields from other packages?
				na := nod(OADDR, nx, nil)
				call.List.Append(na)
				call.List.Append(nh)
				fn.Nbody.Append(nod(OAS, nh, call))
				i++
				continue
			}

			// Otherwise, hash a maximal length run of raw memory.
			size, next := memrun(t, i)

			// h = hashel(&p.first, size, h)
			hashel := hashmem(f.Type)
			call := nod(OCALL, hashel, nil)
			nx := nodSym(OXDOT, np, f.Sym) // TODO: fields from other packages?
			na := nod(OADDR, nx, nil)
			call.List.Append(na)
			call.List.Append(nh)
			call.List.Append(nodintconst(size))
			fn.Nbody.Append(nod(OAS, nh, call))

			i = next
		}
	}

	r := nod(ORETURN, nil, nil)
	r.List.Append(nh)
	fn.Nbody.Append(r)

	if Debug.r != 0 {
		dumplist("genhash body", fn.Nbody)
	}

	funcbody()

	fn.Func.SetDupok(true)
	fn = typecheck(fn, ctxStmt)

	Curfn = fn
	typecheckslice(fn.Nbody.Slice(), ctxStmt)
	Curfn = nil

	if debug_dclstack != 0 {
		testdclstack()
	}

	fn.Func.SetNilCheckDisabled(true)
	xtop = append(xtop, fn)

	// Build closure. It doesn't close over any variables, so
	// it contains just the function pointer.
	dsymptr(closure, 0, sym.Linksym(), 0)
	ggloblsym(closure, int32(Widthptr), obj.DUPOK|obj.RODATA)

	return closure
}

func hashfor(t *types.Type) *Node {
	var sym *types.Sym

	switch a, _ := algtype1(t); a {
	case AMEM:
		Fatalf("hashfor with AMEM type")
	case AINTER:
		sym = Runtimepkg.Lookup("interhash")
	case ANILINTER:
		sym = Runtimepkg.Lookup("nilinterhash")
	case ASTRING:
		sym = Runtimepkg.Lookup("strhash")
	case AFLOAT32:
		sym = Runtimepkg.Lookup("f32hash")
	case AFLOAT64:
		sym = Runtimepkg.Lookup("f64hash")
	case ACPLX64:
		sym = Runtimepkg.Lookup("c64hash")
	case ACPLX128:
		sym = Runtimepkg.Lookup("c128hash")
	default:
		// Note: the caller of hashfor ensured that this symbol
		// exists and has a body by calling genhash for t.
		sym = typesymprefix(".hash", t)
	}

	n := newname(sym)
	setNodeNameFunc(n)
	n.Type = functype(nil, []*Node{
		anonfield(types.NewPtr(t)),
		anonfield(types.Types[TUINTPTR]),
	}, []*Node{
		anonfield(types.Types[TUINTPTR]),
	})
	return n
}

// sysClosure returns a closure which will call the
// given runtime function (with no closed-over variables).
func sysClosure(name string) *obj.LSym {
	s := sysvar(name + "·f")
	if len(s.P) == 0 {
		f := sysfunc(name)
		dsymptr(s, 0, f, 0)
		ggloblsym(s, int32(Widthptr), obj.DUPOK|obj.RODATA)
	}
	return s
}

// geneq returns a symbol which is the closure used to compute
// equality for two objects of type t.
func geneq(t *types.Type) *obj.LSym {
	switch algtype(t) {
	case ANOEQ:
		// The runtime will panic if it tries to compare
		// a type with a nil equality function.
		return nil
	case AMEM0:
		return sysClosure("memequal0")
	case AMEM8:
		return sysClosure("memequal8")
	case AMEM16:
		return sysClosure("memequal16")
	case AMEM32:
		return sysClosure("memequal32")
	case AMEM64:
		return sysClosure("memequal64")
	case AMEM128:
		return sysClosure("memequal128")
	case ASTRING:
		return sysClosure("strequal")
	case AINTER:
		return sysClosure("interequal")
	case ANILINTER:
		return sysClosure("nilinterequal")
	case AFLOAT32:
		return sysClosure("f32equal")
	case AFLOAT64:
		return sysClosure("f64equal")
	case ACPLX64:
		return sysClosure("c64equal")
	case ACPLX128:
		return sysClosure("c128equal")
	case AMEM:
		// make equality closure. The size of the type
		// is encoded in the closure.
		closure := typeLookup(fmt.Sprintf(".eqfunc%d", t.Width)).Linksym()
		if len(closure.P) != 0 {
			return closure
		}
		if memequalvarlen == nil {
			memequalvarlen = sysvar("memequal_varlen") // asm func
		}
		ot := 0
		ot = dsymptr(closure, ot, memequalvarlen, 0)
		ot = duintptr(closure, ot, uint64(t.Width))
		ggloblsym(closure, int32(ot), obj.DUPOK|obj.RODATA)
		return closure
	case ASPECIAL:
		break
	}

	closure := typesymprefix(".eqfunc", t).Linksym()
	if len(closure.P) > 0 { // already generated
		return closure
	}
	sym := typesymprefix(".eq", t)
	if Debug.r != 0 {
		fmt.Printf("geneq %v\n", t)
	}

	// Autogenerate code for equality of structs and arrays.

	lineno = autogeneratedPos // less confusing than end of input
	dclcontext = PEXTERN

	// func sym(p, q *T) bool
	tfn := nod(OTFUNC, nil, nil)
	tfn.List.Set2(
		namedfield("p", types.NewPtr(t)),
		namedfield("q", types.NewPtr(t)),
	)
	tfn.Rlist.Set1(namedfield("r", types.Types[TBOOL]))

	fn := dclfunc(sym, tfn)
	np := asNode(tfn.Type.Params().Field(0).Nname)
	nq := asNode(tfn.Type.Params().Field(1).Nname)
	nr := asNode(tfn.Type.Results().Field(0).Nname)

	// Label to jump to if an equality test fails.
	neq := autolabel(".neq")

	// We reach here only for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch t.Etype {
	default:
		Fatalf("geneq %v", t)

	case TARRAY:
		nelem := t.NumElem()

		// checkAll generates code to check the equality of all array elements.
		// If unroll is greater than nelem, checkAll generates:
		//
		// if eq(p[0], q[0]) && eq(p[1], q[1]) && ... {
		// } else {
		//   return
		// }
		//
		// And so on.
		//
		// Otherwise it generates:
		//
		// for i := 0; i < nelem; i++ {
		//   if eq(p[i], q[i]) {
		//   } else {
		//     goto neq
		//   }
		// }
		//
		// TODO(josharian): consider doing some loop unrolling
		// for larger nelem as well, processing a few elements at a time in a loop.
		checkAll := func(unroll int64, last bool, eq func(pi, qi *Node) *Node) {
			// checkIdx generates a node to check for equality at index i.
			checkIdx := func(i *Node) *Node {
				// pi := p[i]
				pi := nod(OINDEX, np, i)
				pi.SetBounded(true)
				pi.Type = t.Elem()
				// qi := q[i]
				qi := nod(OINDEX, nq, i)
				qi.SetBounded(true)
				qi.Type = t.Elem()
				return eq(pi, qi)
			}

			if nelem <= unroll {
				if last {
					// Do last comparison in a different manner.
					nelem--
				}
				// Generate a series of checks.
				for i := int64(0); i < nelem; i++ {
					// if check {} else { goto neq }
					nif := nod(OIF, checkIdx(nodintconst(i)), nil)
					nif.Rlist.Append(nodSym(OGOTO, nil, neq))
					fn.Nbody.Append(nif)
				}
				if last {
					fn.Nbody.Append(nod(OAS, nr, checkIdx(nodintconst(nelem))))
				}
			} else {
				// Generate a for loop.
				// for i := 0; i < nelem; i++
				i := temp(types.Types[TINT])
				init := nod(OAS, i, nodintconst(0))
				cond := nod(OLT, i, nodintconst(nelem))
				post := nod(OAS, i, nod(OADD, i, nodintconst(1)))
				loop := nod(OFOR, cond, post)
				loop.Ninit.Append(init)
				// if eq(pi, qi) {} else { goto neq }
				nif := nod(OIF, checkIdx(i), nil)
				nif.Rlist.Append(nodSym(OGOTO, nil, neq))
				loop.Nbody.Append(nif)
				fn.Nbody.Append(loop)
				if last {
					fn.Nbody.Append(nod(OAS, nr, nodbool(true)))
				}
			}
		}

		switch t.Elem().Etype {
		case TSTRING:
			// Do two loops. First, check that all the lengths match (cheap).
			// Second, check that all the contents match (expensive).
			// TODO: when the array size is small, unroll the length match checks.
			checkAll(3, false, func(pi, qi *Node) *Node {
				// Compare lengths.
				eqlen, _ := eqstring(pi, qi)
				return eqlen
			})
			checkAll(1, true, func(pi, qi *Node) *Node {
				// Compare contents.
				_, eqmem := eqstring(pi, qi)
				return eqmem
			})
		case TFLOAT32, TFLOAT64:
			checkAll(2, true, func(pi, qi *Node) *Node {
				// p[i] == q[i]
				return nod(OEQ, pi, qi)
			})
		// TODO: pick apart structs, do them piecemeal too
		default:
			checkAll(1, true, func(pi, qi *Node) *Node {
				// p[i] == q[i]
				return nod(OEQ, pi, qi)
			})
		}

	case TSTRUCT:
		// Build a list of conditions to satisfy.
		// The conditions are a list-of-lists. Conditions are reorderable
		// within each inner list. The outer lists must be evaluated in order.
		var conds [][]*Node
		conds = append(conds, []*Node{})
		and := func(n *Node) {
			i := len(conds) - 1
			conds[i] = append(conds[i], n)
		}

		// Walk the struct using memequal for runs of AMEM
		// and calling specific equality tests for the others.
		for i, fields := 0, t.FieldSlice(); i < len(fields); {
			f := fields[i]

			// Skip blank-named fields.
			if f.Sym.IsBlank() {
				i++
				continue
			}

			// Compare non-memory fields with field equality.
			if !IsRegularMemory(f.Type) {
				if EqCanPanic(f.Type) {
					// Enforce ordering by starting a new set of reorderable conditions.
					conds = append(conds, []*Node{})
				}
				p := nodSym(OXDOT, np, f.Sym)
				q := nodSym(OXDOT, nq, f.Sym)
				switch {
				case f.Type.IsString():
					eqlen, eqmem := eqstring(p, q)
					and(eqlen)
					and(eqmem)
				default:
					and(nod(OEQ, p, q))
				}
				if EqCanPanic(f.Type) {
					// Also enforce ordering after something that can panic.
					conds = append(conds, []*Node{})
				}
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

		// Sort conditions to put runtime calls last.
		// Preserve the rest of the ordering.
		var flatConds []*Node
		for _, c := range conds {
			isCall := func(n *Node) bool {
				return n.Op == OCALL || n.Op == OCALLFUNC
			}
			sort.SliceStable(c, func(i, j int) bool {
				return !isCall(c[i]) && isCall(c[j])
			})
			flatConds = append(flatConds, c...)
		}

		if len(flatConds) == 0 {
			fn.Nbody.Append(nod(OAS, nr, nodbool(true)))
		} else {
			for _, c := range flatConds[:len(flatConds)-1] {
				// if cond {} else { goto neq }
				n := nod(OIF, c, nil)
				n.Rlist.Append(nodSym(OGOTO, nil, neq))
				fn.Nbody.Append(n)
			}
			fn.Nbody.Append(nod(OAS, nr, flatConds[len(flatConds)-1]))
		}
	}

	// ret:
	//   return
	ret := autolabel(".ret")
	fn.Nbody.Append(nodSym(OLABEL, nil, ret))
	fn.Nbody.Append(nod(ORETURN, nil, nil))

	// neq:
	//   r = false
	//   return (or goto ret)
	fn.Nbody.Append(nodSym(OLABEL, nil, neq))
	fn.Nbody.Append(nod(OAS, nr, nodbool(false)))
	if EqCanPanic(t) || hasCall(fn) {
		// Epilogue is large, so share it with the equal case.
		fn.Nbody.Append(nodSym(OGOTO, nil, ret))
	} else {
		// Epilogue is small, so don't bother sharing.
		fn.Nbody.Append(nod(ORETURN, nil, nil))
	}
	// TODO(khr): the epilogue size detection condition above isn't perfect.
	// We should really do a generic CL that shares epilogues across
	// the board. See #24936.

	if Debug.r != 0 {
		dumplist("geneq body", fn.Nbody)
	}

	funcbody()

	fn.Func.SetDupok(true)
	fn = typecheck(fn, ctxStmt)

	Curfn = fn
	typecheckslice(fn.Nbody.Slice(), ctxStmt)
	Curfn = nil

	if debug_dclstack != 0 {
		testdclstack()
	}

	// Disable checknils while compiling this code.
	// We are comparing a struct or an array,
	// neither of which can be nil, and our comparisons
	// are shallow.
	fn.Func.SetNilCheckDisabled(true)
	xtop = append(xtop, fn)

	// Generate a closure which points at the function we just generated.
	dsymptr(closure, 0, sym.Linksym(), 0)
	ggloblsym(closure, int32(Widthptr), obj.DUPOK|obj.RODATA)
	return closure
}

func hasCall(n *Node) bool {
	if n.Op == OCALL || n.Op == OCALLFUNC {
		return true
	}
	if n.Left != nil && hasCall(n.Left) {
		return true
	}
	if n.Right != nil && hasCall(n.Right) {
		return true
	}
	for _, x := range n.Ninit.Slice() {
		if hasCall(x) {
			return true
		}
	}
	for _, x := range n.Nbody.Slice() {
		if hasCall(x) {
			return true
		}
	}
	for _, x := range n.List.Slice() {
		if hasCall(x) {
			return true
		}
	}
	for _, x := range n.Rlist.Slice() {
		if hasCall(x) {
			return true
		}
	}
	return false
}

// eqfield returns the node
// 	p.field == q.field
func eqfield(p *Node, q *Node, field *types.Sym) *Node {
	nx := nodSym(OXDOT, p, field)
	ny := nodSym(OXDOT, q, field)
	ne := nod(OEQ, nx, ny)
	return ne
}

// eqstring returns the nodes
//   len(s) == len(t)
// and
//   memequal(s.ptr, t.ptr, len(s))
// which can be used to construct string equality comparison.
// eqlen must be evaluated before eqmem, and shortcircuiting is required.
func eqstring(s, t *Node) (eqlen, eqmem *Node) {
	s = conv(s, types.Types[TSTRING])
	t = conv(t, types.Types[TSTRING])
	sptr := nod(OSPTR, s, nil)
	tptr := nod(OSPTR, t, nil)
	slen := conv(nod(OLEN, s, nil), types.Types[TUINTPTR])
	tlen := conv(nod(OLEN, t, nil), types.Types[TUINTPTR])

	fn := syslook("memequal")
	fn = substArgTypes(fn, types.Types[TUINT8], types.Types[TUINT8])
	call := nod(OCALL, fn, nil)
	call.List.Append(sptr, tptr, slen.copy())
	call = typecheck(call, ctxExpr|ctxMultiOK)

	cmp := nod(OEQ, slen, tlen)
	cmp = typecheck(cmp, ctxExpr)
	cmp.Type = types.Types[TBOOL]
	return cmp, call
}

// eqinterface returns the nodes
//   s.tab == t.tab (or s.typ == t.typ, as appropriate)
// and
//   ifaceeq(s.tab, s.data, t.data) (or efaceeq(s.typ, s.data, t.data), as appropriate)
// which can be used to construct interface equality comparison.
// eqtab must be evaluated before eqdata, and shortcircuiting is required.
func eqinterface(s, t *Node) (eqtab, eqdata *Node) {
	if !types.Identical(s.Type, t.Type) {
		Fatalf("eqinterface %v %v", s.Type, t.Type)
	}
	// func ifaceeq(tab *uintptr, x, y unsafe.Pointer) (ret bool)
	// func efaceeq(typ *uintptr, x, y unsafe.Pointer) (ret bool)
	var fn *Node
	if s.Type.IsEmptyInterface() {
		fn = syslook("efaceeq")
	} else {
		fn = syslook("ifaceeq")
	}

	stab := nod(OITAB, s, nil)
	ttab := nod(OITAB, t, nil)
	sdata := nod(OIDATA, s, nil)
	tdata := nod(OIDATA, t, nil)
	sdata.Type = types.Types[TUNSAFEPTR]
	tdata.Type = types.Types[TUNSAFEPTR]
	sdata.SetTypecheck(1)
	tdata.SetTypecheck(1)

	call := nod(OCALL, fn, nil)
	call.List.Append(stab, sdata, tdata)
	call = typecheck(call, ctxExpr|ctxMultiOK)

	cmp := nod(OEQ, stab, ttab)
	cmp = typecheck(cmp, ctxExpr)
	cmp.Type = types.Types[TBOOL]
	return cmp, call
}

// eqmem returns the node
// 	memequal(&p.field, &q.field [, size])
func eqmem(p *Node, q *Node, field *types.Sym, size int64) *Node {
	nx := nod(OADDR, nodSym(OXDOT, p, field), nil)
	ny := nod(OADDR, nodSym(OXDOT, q, field), nil)
	nx = typecheck(nx, ctxExpr)
	ny = typecheck(ny, ctxExpr)

	fn, needsize := eqmemfunc(size, nx.Type.Elem())
	call := nod(OCALL, fn, nil)
	call.List.Append(nx)
	call.List.Append(ny)
	if needsize {
		call.List.Append(nodintconst(size))
	}

	return call
}

func eqmemfunc(size int64, t *types.Type) (fn *Node, needsize bool) {
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
func memrun(t *types.Type, start int) (size int64, next int) {
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
		if f := t.Field(next); f.Sym.IsBlank() || !IsRegularMemory(f.Type) {
			break
		}
	}
	return t.Field(next-1).End() - t.Field(start).Offset, next
}

// ispaddedfield reports whether the i'th field of struct type t is followed
// by padding.
func ispaddedfield(t *types.Type, i int) bool {
	if !t.IsStruct() {
		Fatalf("ispaddedfield called non-struct %v", t)
	}
	end := t.Width
	if i+1 < t.NumFields() {
		end = t.Field(i + 1).Offset
	}
	return t.Field(i).End() != end
}
