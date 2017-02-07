// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysis

// This file computes the "implements" relation over all pairs of
// named types in the program.  (The mark-up is done by typeinfo.go.)

// TODO(adonovan): do we want to report implements(C, I) where C and I
// belong to different packages and at least one is not exported?

import (
	"go/types"
	"sort"

	"golang.org/x/tools/go/types/typeutil"
)

// computeImplements computes the "implements" relation over all pairs
// of named types in allNamed.
func computeImplements(cache *typeutil.MethodSetCache, allNamed []*types.Named) map[*types.Named]implementsFacts {
	// Information about a single type's method set.
	type msetInfo struct {
		typ          types.Type
		mset         *types.MethodSet
		mask1, mask2 uint64
	}

	initMsetInfo := func(info *msetInfo, typ types.Type) {
		info.typ = typ
		info.mset = cache.MethodSet(typ)
		for i := 0; i < info.mset.Len(); i++ {
			name := info.mset.At(i).Obj().Name()
			info.mask1 |= 1 << methodBit(name[0])
			info.mask2 |= 1 << methodBit(name[len(name)-1])
		}
	}

	// satisfies(T, U) reports whether type T satisfies type U.
	// U must be an interface.
	//
	// Since there are thousands of types (and thus millions of
	// pairs of types) and types.Assignable(T, U) is relatively
	// expensive, we compute assignability directly from the
	// method sets.  (At least one of T and U must be an
	// interface.)
	//
	// We use a trick (thanks gri!) related to a Bloom filter to
	// quickly reject most tests, which are false.  For each
	// method set, we precompute a mask, a set of bits, one per
	// distinct initial byte of each method name.  Thus the mask
	// for io.ReadWriter would be {'R','W'}.  AssignableTo(T, U)
	// cannot be true unless mask(T)&mask(U)==mask(U).
	//
	// As with a Bloom filter, we can improve precision by testing
	// additional hashes, e.g. using the last letter of each
	// method name, so long as the subset mask property holds.
	//
	// When analyzing the standard library, there are about 1e6
	// calls to satisfies(), of which 0.6% return true.  With a
	// 1-hash filter, 95% of calls avoid the expensive check; with
	// a 2-hash filter, this grows to 98.2%.
	satisfies := func(T, U *msetInfo) bool {
		return T.mask1&U.mask1 == U.mask1 &&
			T.mask2&U.mask2 == U.mask2 &&
			containsAllIdsOf(T.mset, U.mset)
	}

	// Information about a named type N, and perhaps also *N.
	type namedInfo struct {
		isInterface bool
		base        msetInfo // N
		ptr         msetInfo // *N, iff N !isInterface
	}

	var infos []namedInfo

	// Precompute the method sets and their masks.
	for _, N := range allNamed {
		var info namedInfo
		initMsetInfo(&info.base, N)
		_, info.isInterface = N.Underlying().(*types.Interface)
		if !info.isInterface {
			initMsetInfo(&info.ptr, types.NewPointer(N))
		}

		if info.base.mask1|info.ptr.mask1 == 0 {
			continue // neither N nor *N has methods
		}

		infos = append(infos, info)
	}

	facts := make(map[*types.Named]implementsFacts)

	// Test all pairs of distinct named types (T, U).
	// TODO(adonovan): opt: compute (U, T) at the same time.
	for t := range infos {
		T := &infos[t]
		var to, from, fromPtr []types.Type
		for u := range infos {
			if t == u {
				continue
			}
			U := &infos[u]
			switch {
			case T.isInterface && U.isInterface:
				if satisfies(&U.base, &T.base) {
					to = append(to, U.base.typ)
				}
				if satisfies(&T.base, &U.base) {
					from = append(from, U.base.typ)
				}
			case T.isInterface: // U concrete
				if satisfies(&U.base, &T.base) {
					to = append(to, U.base.typ)
				} else if satisfies(&U.ptr, &T.base) {
					to = append(to, U.ptr.typ)
				}
			case U.isInterface: // T concrete
				if satisfies(&T.base, &U.base) {
					from = append(from, U.base.typ)
				} else if satisfies(&T.ptr, &U.base) {
					fromPtr = append(fromPtr, U.base.typ)
				}
			}
		}

		// Sort types (arbitrarily) to avoid nondeterminism.
		sort.Sort(typesByString(to))
		sort.Sort(typesByString(from))
		sort.Sort(typesByString(fromPtr))

		facts[T.base.typ.(*types.Named)] = implementsFacts{to, from, fromPtr}
	}

	return facts
}

type implementsFacts struct {
	to      []types.Type // named or ptr-to-named types assignable to interface T
	from    []types.Type // named interfaces assignable from T
	fromPtr []types.Type // named interfaces assignable only from *T
}

type typesByString []types.Type

func (p typesByString) Len() int           { return len(p) }
func (p typesByString) Less(i, j int) bool { return p[i].String() < p[j].String() }
func (p typesByString) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// methodBit returns the index of x in [a-zA-Z], or 52 if not found.
func methodBit(x byte) uint64 {
	switch {
	case 'a' <= x && x <= 'z':
		return uint64(x - 'a')
	case 'A' <= x && x <= 'Z':
		return uint64(26 + x - 'A')
	}
	return 52 // all other bytes
}

// containsAllIdsOf reports whether the method identifiers of T are a
// superset of those in U.  If U belongs to an interface type, the
// result is equal to types.Assignable(T, U), but is cheaper to compute.
//
// TODO(gri): make this a method of *types.MethodSet.
//
func containsAllIdsOf(T, U *types.MethodSet) bool {
	t, tlen := 0, T.Len()
	u, ulen := 0, U.Len()
	for t < tlen && u < ulen {
		tMeth := T.At(t).Obj()
		uMeth := U.At(u).Obj()
		tId := tMeth.Id()
		uId := uMeth.Id()
		if tId > uId {
			// U has a method T lacks: fail.
			return false
		}
		if tId < uId {
			// T has a method U lacks: ignore it.
			t++
			continue
		}
		// U and T both have a method of this Id.  Check types.
		if !types.Identical(tMeth.Type(), uMeth.Type()) {
			return false // type mismatch
		}
		u++
		t++
	}
	return u == ulen
}
