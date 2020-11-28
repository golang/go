// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
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
	case types.TINTER:
		return true
	case types.TARRAY:
		return EqCanPanic(t.Elem())
	case types.TSTRUCT:
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
	case types.TANY, types.TFORW:
		// will be defined later.
		return ANOEQ, t

	case types.TINT8, types.TUINT8, types.TINT16, types.TUINT16,
		types.TINT32, types.TUINT32, types.TINT64, types.TUINT64,
		types.TINT, types.TUINT, types.TUINTPTR,
		types.TBOOL, types.TPTR,
		types.TCHAN, types.TUNSAFEPTR:
		return AMEM, nil

	case types.TFUNC, types.TMAP:
		return ANOEQ, t

	case types.TFLOAT32:
		return AFLOAT32, nil

	case types.TFLOAT64:
		return AFLOAT64, nil

	case types.TCOMPLEX64:
		return ACPLX64, nil

	case types.TCOMPLEX128:
		return ACPLX128, nil

	case types.TSTRING:
		return ASTRING, nil

	case types.TINTER:
		if t.IsEmptyInterface() {
			return ANILINTER, nil
		}
		return AINTER, nil

	case types.TSLICE:
		return ANOEQ, t

	case types.TARRAY:
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

	case types.TSTRUCT:
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

	base.Fatalf("algtype1: unexpected type %v", t)
	return 0, nil
}

// genhash returns a symbol which is the closure used to compute
// the hash of a value of type t.
// Note: the generated function must match runtime.typehash exactly.
func genhash(t *types.Type) *obj.LSym {
	switch algtype(t) {
	default:
		// genhash is only called for types that have equality
		base.Fatalf("genhash %v", t)
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
	if base.Flag.LowerR != 0 {
		fmt.Printf("genhash %v %v %v\n", closure, sym, t)
	}

	base.Pos = autogeneratedPos // less confusing than end of input
	dclcontext = ir.PEXTERN

	// func sym(p *T, h uintptr) uintptr
	tfn := ir.Nod(ir.OTFUNC, nil, nil)
	tfn.PtrList().Set2(
		namedfield("p", types.NewPtr(t)),
		namedfield("h", types.Types[types.TUINTPTR]),
	)
	tfn.PtrRlist().Set1(anonfield(types.Types[types.TUINTPTR]))

	fn := dclfunc(sym, tfn)
	np := ir.AsNode(tfn.Type().Params().Field(0).Nname)
	nh := ir.AsNode(tfn.Type().Params().Field(1).Nname)

	switch t.Etype {
	case types.TARRAY:
		// An array of pure memory would be handled by the
		// standard algorithm, so the element type must not be
		// pure memory.
		hashel := hashfor(t.Elem())

		n := ir.Nod(ir.ORANGE, nil, ir.Nod(ir.ODEREF, np, nil))
		ni := ir.Node(NewName(lookup("i")))
		ni.SetType(types.Types[types.TINT])
		n.PtrList().Set1(ni)
		n.SetColas(true)
		colasdefn(n.List().Slice(), n)
		ni = n.List().First()

		// h = hashel(&p[i], h)
		call := ir.Nod(ir.OCALL, hashel, nil)

		nx := ir.Nod(ir.OINDEX, np, ni)
		nx.SetBounded(true)
		na := ir.Nod(ir.OADDR, nx, nil)
		call.PtrList().Append(na)
		call.PtrList().Append(nh)
		n.PtrBody().Append(ir.Nod(ir.OAS, nh, call))

		fn.PtrBody().Append(n)

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
				call := ir.Nod(ir.OCALL, hashel, nil)
				nx := nodSym(ir.OXDOT, np, f.Sym) // TODO: fields from other packages?
				na := ir.Nod(ir.OADDR, nx, nil)
				call.PtrList().Append(na)
				call.PtrList().Append(nh)
				fn.PtrBody().Append(ir.Nod(ir.OAS, nh, call))
				i++
				continue
			}

			// Otherwise, hash a maximal length run of raw memory.
			size, next := memrun(t, i)

			// h = hashel(&p.first, size, h)
			hashel := hashmem(f.Type)
			call := ir.Nod(ir.OCALL, hashel, nil)
			nx := nodSym(ir.OXDOT, np, f.Sym) // TODO: fields from other packages?
			na := ir.Nod(ir.OADDR, nx, nil)
			call.PtrList().Append(na)
			call.PtrList().Append(nh)
			call.PtrList().Append(nodintconst(size))
			fn.PtrBody().Append(ir.Nod(ir.OAS, nh, call))

			i = next
		}
	}

	r := ir.Nod(ir.ORETURN, nil, nil)
	r.PtrList().Append(nh)
	fn.PtrBody().Append(r)

	if base.Flag.LowerR != 0 {
		ir.DumpList("genhash body", fn.Body())
	}

	funcbody()

	fn.SetDupok(true)
	typecheckFunc(fn)

	Curfn = fn
	typecheckslice(fn.Body().Slice(), ctxStmt)
	Curfn = nil

	if base.Debug.DclStack != 0 {
		testdclstack()
	}

	fn.SetNilCheckDisabled(true)
	xtop = append(xtop, fn)

	// Build closure. It doesn't close over any variables, so
	// it contains just the function pointer.
	dsymptr(closure, 0, sym.Linksym(), 0)
	ggloblsym(closure, int32(Widthptr), obj.DUPOK|obj.RODATA)

	return closure
}

func hashfor(t *types.Type) ir.Node {
	var sym *types.Sym

	switch a, _ := algtype1(t); a {
	case AMEM:
		base.Fatalf("hashfor with AMEM type")
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

	n := NewName(sym)
	setNodeNameFunc(n)
	n.SetType(functype(nil, []ir.Node{
		anonfield(types.NewPtr(t)),
		anonfield(types.Types[types.TUINTPTR]),
	}, []ir.Node{
		anonfield(types.Types[types.TUINTPTR]),
	}))
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
	if base.Flag.LowerR != 0 {
		fmt.Printf("geneq %v\n", t)
	}

	// Autogenerate code for equality of structs and arrays.

	base.Pos = autogeneratedPos // less confusing than end of input
	dclcontext = ir.PEXTERN

	// func sym(p, q *T) bool
	tfn := ir.Nod(ir.OTFUNC, nil, nil)
	tfn.PtrList().Set2(
		namedfield("p", types.NewPtr(t)),
		namedfield("q", types.NewPtr(t)),
	)
	tfn.PtrRlist().Set1(namedfield("r", types.Types[types.TBOOL]))

	fn := dclfunc(sym, tfn)
	np := ir.AsNode(tfn.Type().Params().Field(0).Nname)
	nq := ir.AsNode(tfn.Type().Params().Field(1).Nname)
	nr := ir.AsNode(tfn.Type().Results().Field(0).Nname)

	// Label to jump to if an equality test fails.
	neq := autolabel(".neq")

	// We reach here only for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch t.Etype {
	default:
		base.Fatalf("geneq %v", t)

	case types.TARRAY:
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
		checkAll := func(unroll int64, last bool, eq func(pi, qi ir.Node) ir.Node) {
			// checkIdx generates a node to check for equality at index i.
			checkIdx := func(i ir.Node) ir.Node {
				// pi := p[i]
				pi := ir.Nod(ir.OINDEX, np, i)
				pi.SetBounded(true)
				pi.SetType(t.Elem())
				// qi := q[i]
				qi := ir.Nod(ir.OINDEX, nq, i)
				qi.SetBounded(true)
				qi.SetType(t.Elem())
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
					nif := ir.Nod(ir.OIF, checkIdx(nodintconst(i)), nil)
					nif.PtrRlist().Append(nodSym(ir.OGOTO, nil, neq))
					fn.PtrBody().Append(nif)
				}
				if last {
					fn.PtrBody().Append(ir.Nod(ir.OAS, nr, checkIdx(nodintconst(nelem))))
				}
			} else {
				// Generate a for loop.
				// for i := 0; i < nelem; i++
				i := temp(types.Types[types.TINT])
				init := ir.Nod(ir.OAS, i, nodintconst(0))
				cond := ir.Nod(ir.OLT, i, nodintconst(nelem))
				post := ir.Nod(ir.OAS, i, ir.Nod(ir.OADD, i, nodintconst(1)))
				loop := ir.Nod(ir.OFOR, cond, post)
				loop.PtrInit().Append(init)
				// if eq(pi, qi) {} else { goto neq }
				nif := ir.Nod(ir.OIF, checkIdx(i), nil)
				nif.PtrRlist().Append(nodSym(ir.OGOTO, nil, neq))
				loop.PtrBody().Append(nif)
				fn.PtrBody().Append(loop)
				if last {
					fn.PtrBody().Append(ir.Nod(ir.OAS, nr, nodbool(true)))
				}
			}
		}

		switch t.Elem().Etype {
		case types.TSTRING:
			// Do two loops. First, check that all the lengths match (cheap).
			// Second, check that all the contents match (expensive).
			// TODO: when the array size is small, unroll the length match checks.
			checkAll(3, false, func(pi, qi ir.Node) ir.Node {
				// Compare lengths.
				eqlen, _ := eqstring(pi, qi)
				return eqlen
			})
			checkAll(1, true, func(pi, qi ir.Node) ir.Node {
				// Compare contents.
				_, eqmem := eqstring(pi, qi)
				return eqmem
			})
		case types.TFLOAT32, types.TFLOAT64:
			checkAll(2, true, func(pi, qi ir.Node) ir.Node {
				// p[i] == q[i]
				return ir.Nod(ir.OEQ, pi, qi)
			})
		// TODO: pick apart structs, do them piecemeal too
		default:
			checkAll(1, true, func(pi, qi ir.Node) ir.Node {
				// p[i] == q[i]
				return ir.Nod(ir.OEQ, pi, qi)
			})
		}

	case types.TSTRUCT:
		// Build a list of conditions to satisfy.
		// The conditions are a list-of-lists. Conditions are reorderable
		// within each inner list. The outer lists must be evaluated in order.
		var conds [][]ir.Node
		conds = append(conds, []ir.Node{})
		and := func(n ir.Node) {
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
					conds = append(conds, []ir.Node{})
				}
				p := nodSym(ir.OXDOT, np, f.Sym)
				q := nodSym(ir.OXDOT, nq, f.Sym)
				switch {
				case f.Type.IsString():
					eqlen, eqmem := eqstring(p, q)
					and(eqlen)
					and(eqmem)
				default:
					and(ir.Nod(ir.OEQ, p, q))
				}
				if EqCanPanic(f.Type) {
					// Also enforce ordering after something that can panic.
					conds = append(conds, []ir.Node{})
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
		var flatConds []ir.Node
		for _, c := range conds {
			isCall := func(n ir.Node) bool {
				return n.Op() == ir.OCALL || n.Op() == ir.OCALLFUNC
			}
			sort.SliceStable(c, func(i, j int) bool {
				return !isCall(c[i]) && isCall(c[j])
			})
			flatConds = append(flatConds, c...)
		}

		if len(flatConds) == 0 {
			fn.PtrBody().Append(ir.Nod(ir.OAS, nr, nodbool(true)))
		} else {
			for _, c := range flatConds[:len(flatConds)-1] {
				// if cond {} else { goto neq }
				n := ir.Nod(ir.OIF, c, nil)
				n.PtrRlist().Append(nodSym(ir.OGOTO, nil, neq))
				fn.PtrBody().Append(n)
			}
			fn.PtrBody().Append(ir.Nod(ir.OAS, nr, flatConds[len(flatConds)-1]))
		}
	}

	// ret:
	//   return
	ret := autolabel(".ret")
	fn.PtrBody().Append(nodSym(ir.OLABEL, nil, ret))
	fn.PtrBody().Append(ir.Nod(ir.ORETURN, nil, nil))

	// neq:
	//   r = false
	//   return (or goto ret)
	fn.PtrBody().Append(nodSym(ir.OLABEL, nil, neq))
	fn.PtrBody().Append(ir.Nod(ir.OAS, nr, nodbool(false)))
	if EqCanPanic(t) || hasCall(fn) {
		// Epilogue is large, so share it with the equal case.
		fn.PtrBody().Append(nodSym(ir.OGOTO, nil, ret))
	} else {
		// Epilogue is small, so don't bother sharing.
		fn.PtrBody().Append(ir.Nod(ir.ORETURN, nil, nil))
	}
	// TODO(khr): the epilogue size detection condition above isn't perfect.
	// We should really do a generic CL that shares epilogues across
	// the board. See #24936.

	if base.Flag.LowerR != 0 {
		ir.DumpList("geneq body", fn.Body())
	}

	funcbody()

	fn.SetDupok(true)
	typecheckFunc(fn)

	Curfn = fn
	typecheckslice(fn.Body().Slice(), ctxStmt)
	Curfn = nil

	if base.Debug.DclStack != 0 {
		testdclstack()
	}

	// Disable checknils while compiling this code.
	// We are comparing a struct or an array,
	// neither of which can be nil, and our comparisons
	// are shallow.
	fn.SetNilCheckDisabled(true)
	xtop = append(xtop, fn)

	// Generate a closure which points at the function we just generated.
	dsymptr(closure, 0, sym.Linksym(), 0)
	ggloblsym(closure, int32(Widthptr), obj.DUPOK|obj.RODATA)
	return closure
}

func hasCall(n ir.Node) bool {
	if n.Op() == ir.OCALL || n.Op() == ir.OCALLFUNC {
		return true
	}
	if n.Left() != nil && hasCall(n.Left()) {
		return true
	}
	if n.Right() != nil && hasCall(n.Right()) {
		return true
	}
	for _, x := range n.Init().Slice() {
		if hasCall(x) {
			return true
		}
	}
	for _, x := range n.Body().Slice() {
		if hasCall(x) {
			return true
		}
	}
	for _, x := range n.List().Slice() {
		if hasCall(x) {
			return true
		}
	}
	for _, x := range n.Rlist().Slice() {
		if hasCall(x) {
			return true
		}
	}
	return false
}

// eqfield returns the node
// 	p.field == q.field
func eqfield(p ir.Node, q ir.Node, field *types.Sym) ir.Node {
	nx := nodSym(ir.OXDOT, p, field)
	ny := nodSym(ir.OXDOT, q, field)
	ne := ir.Nod(ir.OEQ, nx, ny)
	return ne
}

// eqstring returns the nodes
//   len(s) == len(t)
// and
//   memequal(s.ptr, t.ptr, len(s))
// which can be used to construct string equality comparison.
// eqlen must be evaluated before eqmem, and shortcircuiting is required.
func eqstring(s, t ir.Node) (eqlen, eqmem ir.Node) {
	s = conv(s, types.Types[types.TSTRING])
	t = conv(t, types.Types[types.TSTRING])
	sptr := ir.Nod(ir.OSPTR, s, nil)
	tptr := ir.Nod(ir.OSPTR, t, nil)
	slen := conv(ir.Nod(ir.OLEN, s, nil), types.Types[types.TUINTPTR])
	tlen := conv(ir.Nod(ir.OLEN, t, nil), types.Types[types.TUINTPTR])

	fn := syslook("memequal")
	fn = substArgTypes(fn, types.Types[types.TUINT8], types.Types[types.TUINT8])
	call := ir.Nod(ir.OCALL, fn, nil)
	call.PtrList().Append(sptr, tptr, ir.Copy(slen))
	call = typecheck(call, ctxExpr|ctxMultiOK)

	cmp := ir.Nod(ir.OEQ, slen, tlen)
	cmp = typecheck(cmp, ctxExpr)
	cmp.SetType(types.Types[types.TBOOL])
	return cmp, call
}

// eqinterface returns the nodes
//   s.tab == t.tab (or s.typ == t.typ, as appropriate)
// and
//   ifaceeq(s.tab, s.data, t.data) (or efaceeq(s.typ, s.data, t.data), as appropriate)
// which can be used to construct interface equality comparison.
// eqtab must be evaluated before eqdata, and shortcircuiting is required.
func eqinterface(s, t ir.Node) (eqtab, eqdata ir.Node) {
	if !types.Identical(s.Type(), t.Type()) {
		base.Fatalf("eqinterface %v %v", s.Type(), t.Type())
	}
	// func ifaceeq(tab *uintptr, x, y unsafe.Pointer) (ret bool)
	// func efaceeq(typ *uintptr, x, y unsafe.Pointer) (ret bool)
	var fn ir.Node
	if s.Type().IsEmptyInterface() {
		fn = syslook("efaceeq")
	} else {
		fn = syslook("ifaceeq")
	}

	stab := ir.Nod(ir.OITAB, s, nil)
	ttab := ir.Nod(ir.OITAB, t, nil)
	sdata := ir.Nod(ir.OIDATA, s, nil)
	tdata := ir.Nod(ir.OIDATA, t, nil)
	sdata.SetType(types.Types[types.TUNSAFEPTR])
	tdata.SetType(types.Types[types.TUNSAFEPTR])
	sdata.SetTypecheck(1)
	tdata.SetTypecheck(1)

	call := ir.Nod(ir.OCALL, fn, nil)
	call.PtrList().Append(stab, sdata, tdata)
	call = typecheck(call, ctxExpr|ctxMultiOK)

	cmp := ir.Nod(ir.OEQ, stab, ttab)
	cmp = typecheck(cmp, ctxExpr)
	cmp.SetType(types.Types[types.TBOOL])
	return cmp, call
}

// eqmem returns the node
// 	memequal(&p.field, &q.field [, size])
func eqmem(p ir.Node, q ir.Node, field *types.Sym, size int64) ir.Node {
	nx := ir.Nod(ir.OADDR, nodSym(ir.OXDOT, p, field), nil)
	ny := ir.Nod(ir.OADDR, nodSym(ir.OXDOT, q, field), nil)
	nx = typecheck(nx, ctxExpr)
	ny = typecheck(ny, ctxExpr)

	fn, needsize := eqmemfunc(size, nx.Type().Elem())
	call := ir.Nod(ir.OCALL, fn, nil)
	call.PtrList().Append(nx)
	call.PtrList().Append(ny)
	if needsize {
		call.PtrList().Append(nodintconst(size))
	}

	return call
}

func eqmemfunc(size int64, t *types.Type) (fn ir.Node, needsize bool) {
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
		base.Fatalf("ispaddedfield called non-struct %v", t)
	}
	end := t.Width
	if i+1 < t.NumFields() {
		end = t.Field(i + 1).Offset
	}
	return t.Field(i).End() != end
}
