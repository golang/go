// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// This file defines a hash function for Types.

import (
	"fmt"
	"hash/maphash"
)

type (
	// Hasher and HasherIgnoreTags define hash functions and
	// equivalence relations for [Types] that are consistent with
	// [Identical] and [IdenticalIgnoreTags], respectively.
	// They use the same hash function, which ignores tags;
	// only the Equal methods vary.
	//
	// Hashers are stateless.
	Hasher           struct{}
	HasherIgnoreTags struct{}
)

var (
	_ maphash.Hasher[Type] = Hasher{}
	_ maphash.Hasher[Type] = HasherIgnoreTags{}
)

func (Hasher) Hash(h *maphash.Hash, t Type)           { hasher{inGenericSig: false}.hash(h, t) }
func (HasherIgnoreTags) Hash(h *maphash.Hash, t Type) { hasher{inGenericSig: false}.hash(h, t) }
func (Hasher) Equal(x, y Type) bool                   { return Identical(x, y) }
func (HasherIgnoreTags) Equal(x, y Type) bool         { return IdenticalIgnoreTags(x, y) }

// hasher holds the state of a single hash traversal, namely,
// whether we are inside the signature of a generic function.
// This is used to optimize [hasher.hashTypeParam].
type hasher struct{ inGenericSig bool }

func (hr hasher) hash(h *maphash.Hash, t Type) {
	// See [Identical] for rationale.
	switch t := t.(type) {
	case *Alias:
		hr.hash(h, Unalias(t))

	case *Array:
		h.WriteByte('A')
		maphash.WriteComparable(h, t.Len())
		hr.hash(h, t.Elem())

	case *Basic:
		h.WriteByte('B')
		h.WriteByte(byte(t.Kind()))

	case *Chan:
		h.WriteByte('C')
		h.WriteByte(byte(t.Dir()))
		hr.hash(h, t.Elem())

	case *Interface:
		h.WriteByte('I')
		h.WriteByte(byte(t.NumMethods()))

		// Interfaces are identical if they have the same set of methods, with
		// identical names and types, and they have the same set of type
		// restrictions. See [Identical] for more details.

		// Hash the methods.
		//
		// Because [Identical] treats Methods as an unordered set,
		// we must either:
		// (a) sort the methods into some canonical order; or
		// (b) hash them each in parallel, combine them with a
		//     commutative operation such as + or ^, and then
		//     write this value into the primary hasher.
		// Since (a) requires allocation, we choose (b).
		var hash uint64
		for m := range t.Methods() {
			var subh maphash.Hash
			subh.SetSeed(h.Seed())
			// Ignore m.Pkg().
			// Use shallow hash on method signature to
			// avoid anonymous interface cycles.
			subh.WriteString(m.Name())
			hr.shallowHash(&subh, m.Type())
			hash ^= subh.Sum64()
		}
		maphash.WriteComparable(h, hash)

		// Hash type restrictions.
		// TODO(adonovan): call (fork of) InterfaceTermSet from
		// golang.org/x/tools/internal/typeparams/normalize.go.
		// hr.hashTermSet(h, terms)

	case *Map:
		h.WriteByte('M')
		hr.hash(h, t.Key())
		hr.hash(h, t.Elem())

	case *Named:
		h.WriteByte('N')
		hr.hashTypeName(h, t.Obj())
		for targ := range t.TypeArgs().Types() {
			hr.hash(h, targ)
		}

	case *Pointer:
		h.WriteByte('P')
		hr.hash(h, t.Elem())

	case *Signature:
		h.WriteByte('F')
		maphash.WriteComparable(h, t.Variadic())
		tparams := t.TypeParams()
		if n := tparams.Len(); n > 0 {
			hr.inGenericSig = true // affects constraints, params, and results

			maphash.WriteComparable(h, n)
			for tparam := range tparams.TypeParams() {
				hr.hash(h, tparam.Constraint())
			}
		}
		hr.hashTuple(h, t.Params())
		hr.hashTuple(h, t.Results())

	case *Slice:
		h.WriteByte('S')
		hr.hash(h, t.Elem())

	case *Struct:
		h.WriteByte('R') // mnemonic: a struct is a record type
		n := t.NumFields()
		h.WriteByte(byte(n))
		for i := range n {
			f := t.Field(i)
			maphash.WriteComparable(h, f.Anonymous())
			// Ignore t.Tag(i), so that a single hash function
			// can be used with both [Identical] and [IdenticalIgnoreTags].
			h.WriteString(f.Name()) // (ignore f.Pkg)
			hr.hash(h, f.Type())
		}

	case *Tuple:
		hr.hashTuple(h, t)

	case *TypeParam:
		hr.hashTypeParam(h, t)

	case *Union:
		h.WriteByte('U')
		// TODO(adonovan): opt: call (fork of) UnionTermSet from
		// golang.org/x/tools/internal/typeparams/normalize.go.
		// hr.hashTermSet(h, terms)

	default:
		panic(fmt.Sprintf("%T: %v", t, t))
	}
}

func (hr hasher) hashTuple(h *maphash.Hash, t *Tuple) {
	h.WriteByte('T')
	h.WriteByte(byte(t.Len()))
	for v := range t.Variables() {
		hr.hash(h, v.Type())
	}
}

// func (hr hasher) hashTermSet(h *maphash.Hash, terms []*Term) {
// 	h.WriteByte(byte(len(terms)))
// 	for _, term := range terms {
// 		// term order is not significant.
// 		h.WriteByte(byte(btoi(term.Tilde())))
// 		hr.hash(h, term.Type())
// 	}
// }

// hashTypeParam encodes a type parameter into hasher h.
func (hr hasher) hashTypeParam(h *maphash.Hash, t *TypeParam) {
	h.WriteByte('P')
	// Within the signature of a generic function, TypeParams are
	// identical if they have the same index and constraint, so we
	// hash them based on index.
	//
	// When we are outside a generic function, free TypeParams are
	// identical iff they are the same object, so we can use a
	// more discriminating hash consistent with object identity.
	// This optimization saves [Map] about 4% when hashing all the
	// Info.Types in the forward closure of net/http.
	if !hr.inGenericSig {
		// Optimization: outside a generic function signature,
		// use a more discrimating hash consistent with object identity.
		hr.hashTypeName(h, t.Obj())
	} else {
		h.WriteByte(byte(t.Index()))
	}
}

// hashTypeName hashes the pointer of tname.
func (hasher) hashTypeName(h *maphash.Hash, tname *TypeName) {
	h.WriteByte('N')
	// Since Identical uses == to compare TypeNames,
	// the hash function uses maphash.Comparable.
	maphash.WriteComparable(h, tname)
}

// shallowHash computes a hash of t without looking at any of its
// element Types, to avoid potential anonymous cycles in the types of
// interface methods.
//
// When an unnamed non-empty interface type appears anywhere among the
// arguments or results of an interface method, there is a potential
// for endless recursion. Consider:
//
//	type X interface { m() []*interface { X } }
//
// The problem is that the Methods of the interface in m's result type
// include m itself; there is no mention of the named type X that
// might help us break the cycle.
// (See comment in [Identical], case *Interface, for more.)
func (hr hasher) shallowHash(h *maphash.Hash, t Type) {
	// t is the type of an interface method (Signature),
	// its params or results (Tuples), or their immediate
	// elements (mostly Slice, Pointer, Basic, Named),
	// so there's no need to optimize anything else.
	switch t := t.(type) {
	case *Alias:
		hr.shallowHash(h, Unalias(t))

	case *Array:
		h.WriteByte('A')
		maphash.WriteComparable(h, t.Len())
		// ignore t.Elem()

	case *Basic:
		h.WriteByte('B')
		h.WriteByte(byte(t.Kind()))

	case *Chan:
		h.WriteByte('C')
		// ignore Dir(), Elem()

	case *Interface:
		h.WriteByte('I')
		// no recursion here

	case *Map:
		h.WriteByte('M')
		// ignore Key(), Elem()

	case *Named:
		hr.hashTypeName(h, t.Obj())

	case *Pointer:
		h.WriteByte('P')
		// ignore t.Elem()

	case *Signature:
		h.WriteByte(byte(btoi(t.Variadic())))
		// The Signature/Tuple recursion is always
		// finite and invariably shallow.
		hr.shallowHash(h, t.Params())
		hr.shallowHash(h, t.Results())

	case *Slice:
		h.WriteByte('S')
		// ignore t.Elem()

	case *Struct:
		h.WriteByte('R') // mnemonic: a struct is a record type
		h.WriteByte(byte(t.NumFields()))
		// ignore t.Fields()

	case *Tuple:
		h.WriteByte('T')
		h.WriteByte(byte(t.Len()))
		for v := range t.Variables() {
			hr.shallowHash(h, v.Type())
		}

	case *TypeParam:
		hr.hashTypeParam(h, t)

	case *Union:
		h.WriteByte('U')
		// ignore term set

	default:
		panic(fmt.Sprintf("shallowHash: %T: %v", t, t))
	}
}

func btoi(b bool) int {
	if b {
		return 1
	} else {
		return 0
	}
}
