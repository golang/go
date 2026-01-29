// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeutil defines various utilities for types, such as [Map],
// a hash table that maps [types.Type] to any value.
package typeutil

import (
	"bytes"
	"fmt"
	"go/types"
	"hash/maphash"

	"golang.org/x/tools/internal/typeparams"
)

// Map is a hash-table-based mapping from types (types.Type) to
// arbitrary values.  The concrete types that implement
// the Type interface are pointers.  Since they are not canonicalized,
// == cannot be used to check for equivalence, and thus we cannot
// simply use a Go map.
//
// Just as with map[K]V, a nil *Map is a valid empty map.
//
// Read-only map operations ([Map.At], [Map.Len], and so on) may
// safely be called concurrently.
//
// TODO(adonovan): deprecate in favor of https://go.dev/issues/69420
// and 69559, if the latter proposals for a generic hash-map type and
// a types.Hash function are accepted.
type Map struct {
	table  map[uint32][]entry // maps hash to bucket; entry.key==nil means unused
	length int                // number of map entries
}

// entry is an entry (key/value association) in a hash bucket.
type entry struct {
	key   types.Type
	value any
}

// SetHasher has no effect.
//
// It is a relic of an optimization that is no longer profitable. Do
// not use [Hasher], [MakeHasher], or [SetHasher] in new code.
func (m *Map) SetHasher(Hasher) {}

// Delete removes the entry with the given key, if any.
// It returns true if the entry was found.
func (m *Map) Delete(key types.Type) bool {
	if m != nil && m.table != nil {
		hash := hash(key)
		bucket := m.table[hash]
		for i, e := range bucket {
			if e.key != nil && types.Identical(key, e.key) {
				// We can't compact the bucket as it
				// would disturb iterators.
				bucket[i] = entry{}
				m.length--
				return true
			}
		}
	}
	return false
}

// At returns the map entry for the given key.
// The result is nil if the entry is not present.
func (m *Map) At(key types.Type) any {
	if m != nil && m.table != nil {
		for _, e := range m.table[hash(key)] {
			if e.key != nil && types.Identical(key, e.key) {
				return e.value
			}
		}
	}
	return nil
}

// Set sets the map entry for key to val,
// and returns the previous entry, if any.
func (m *Map) Set(key types.Type, value any) (prev any) {
	if m.table != nil {
		hash := hash(key)
		bucket := m.table[hash]
		var hole *entry
		for i, e := range bucket {
			if e.key == nil {
				hole = &bucket[i]
			} else if types.Identical(key, e.key) {
				prev = e.value
				bucket[i].value = value
				return
			}
		}

		if hole != nil {
			*hole = entry{key, value} // overwrite deleted entry
		} else {
			m.table[hash] = append(bucket, entry{key, value})
		}
	} else {
		hash := hash(key)
		m.table = map[uint32][]entry{hash: {entry{key, value}}}
	}

	m.length++
	return
}

// Len returns the number of map entries.
func (m *Map) Len() int {
	if m != nil {
		return m.length
	}
	return 0
}

// Iterate calls function f on each entry in the map in unspecified order.
//
// If f should mutate the map, Iterate provides the same guarantees as
// Go maps: if f deletes a map entry that Iterate has not yet reached,
// f will not be invoked for it, but if f inserts a map entry that
// Iterate has not yet reached, whether or not f will be invoked for
// it is unspecified.
func (m *Map) Iterate(f func(key types.Type, value any)) {
	if m != nil {
		for _, bucket := range m.table {
			for _, e := range bucket {
				if e.key != nil {
					f(e.key, e.value)
				}
			}
		}
	}
}

// Keys returns a new slice containing the set of map keys.
// The order is unspecified.
func (m *Map) Keys() []types.Type {
	keys := make([]types.Type, 0, m.Len())
	m.Iterate(func(key types.Type, _ any) {
		keys = append(keys, key)
	})
	return keys
}

func (m *Map) toString(values bool) string {
	if m == nil {
		return "{}"
	}
	var buf bytes.Buffer
	fmt.Fprint(&buf, "{")
	sep := ""
	m.Iterate(func(key types.Type, value any) {
		fmt.Fprint(&buf, sep)
		sep = ", "
		fmt.Fprint(&buf, key)
		if values {
			fmt.Fprintf(&buf, ": %q", value)
		}
	})
	fmt.Fprint(&buf, "}")
	return buf.String()
}

// String returns a string representation of the map's entries.
// Values are printed using fmt.Sprintf("%v", v).
// Order is unspecified.
func (m *Map) String() string {
	return m.toString(true)
}

// KeysString returns a string representation of the map's key set.
// Order is unspecified.
func (m *Map) KeysString() string {
	return m.toString(false)
}

// -- Hasher --

// hash returns the hash of type t.
// TODO(adonovan): replace by types.Hash when Go proposal #69420 is accepted.
func hash(t types.Type) uint32 {
	return theHasher.Hash(t)
}

// A Hasher provides a [Hasher.Hash] method to map a type to its hash value.
// Hashers are stateless, and all are equivalent.
type Hasher struct{}

var theHasher Hasher

// MakeHasher returns Hasher{}.
// Hashers are stateless; all are equivalent.
func MakeHasher() Hasher { return theHasher }

// Hash computes a hash value for the given type t such that
// Identical(t, t') => Hash(t) == Hash(t').
func (h Hasher) Hash(t types.Type) uint32 {
	return hasher{inGenericSig: false}.hash(t)
}

// hasher holds the state of a single Hash traversal: whether we are
// inside the signature of a generic function; this is used to
// optimize [hasher.hashTypeParam].
type hasher struct{ inGenericSig bool }

// hashString computes the Fowler–Noll–Vo hash of s.
func hashString(s string) uint32 {
	var h uint32
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= 16777619
	}
	return h
}

// hash computes the hash of t.
func (h hasher) hash(t types.Type) uint32 {
	// See Identical for rationale.
	switch t := t.(type) {
	case *types.Basic:
		return uint32(t.Kind())

	case *types.Alias:
		return h.hash(types.Unalias(t))

	case *types.Array:
		return 9043 + 2*uint32(t.Len()) + 3*h.hash(t.Elem())

	case *types.Slice:
		return 9049 + 2*h.hash(t.Elem())

	case *types.Struct:
		var hash uint32 = 9059
		for i, n := 0, t.NumFields(); i < n; i++ {
			f := t.Field(i)
			if f.Anonymous() {
				hash += 8861
			}
			hash += hashString(t.Tag(i))
			hash += hashString(f.Name()) // (ignore f.Pkg)
			hash += h.hash(f.Type())
		}
		return hash

	case *types.Pointer:
		return 9067 + 2*h.hash(t.Elem())

	case *types.Signature:
		var hash uint32 = 9091
		if t.Variadic() {
			hash *= 8863
		}

		tparams := t.TypeParams()
		if n := tparams.Len(); n > 0 {
			h.inGenericSig = true // affects constraints, params, and results

			for i := range n {
				tparam := tparams.At(i)
				hash += 7 * h.hash(tparam.Constraint())
			}
		}

		return hash + 3*h.hashTuple(t.Params()) + 5*h.hashTuple(t.Results())

	case *types.Union:
		return h.hashUnion(t)

	case *types.Interface:
		// Interfaces are identical if they have the same set of methods, with
		// identical names and types, and they have the same set of type
		// restrictions. See go/types.identical for more details.
		var hash uint32 = 9103

		// Hash methods.
		for i, n := 0, t.NumMethods(); i < n; i++ {
			// Method order is not significant.
			// Ignore m.Pkg().
			m := t.Method(i)
			// Use shallow hash on method signature to
			// avoid anonymous interface cycles.
			hash += 3*hashString(m.Name()) + 5*h.shallowHash(m.Type())
		}

		// Hash type restrictions.
		terms, err := typeparams.InterfaceTermSet(t)
		// if err != nil t has invalid type restrictions.
		if err == nil {
			hash += h.hashTermSet(terms)
		}

		return hash

	case *types.Map:
		return 9109 + 2*h.hash(t.Key()) + 3*h.hash(t.Elem())

	case *types.Chan:
		return 9127 + 2*uint32(t.Dir()) + 3*h.hash(t.Elem())

	case *types.Named:
		hash := h.hashTypeName(t.Obj())
		targs := t.TypeArgs()
		for targ := range targs.Types() {
			hash += 2 * h.hash(targ)
		}
		return hash

	case *types.TypeParam:
		return h.hashTypeParam(t)

	case *types.Tuple:
		return h.hashTuple(t)
	}

	panic(fmt.Sprintf("%T: %v", t, t))
}

func (h hasher) hashTuple(tuple *types.Tuple) uint32 {
	// See go/types.identicalTypes for rationale.
	n := tuple.Len()
	hash := 9137 + 2*uint32(n)
	for i := range n {
		hash += 3 * h.hash(tuple.At(i).Type())
	}
	return hash
}

func (h hasher) hashUnion(t *types.Union) uint32 {
	// Hash type restrictions.
	terms, err := typeparams.UnionTermSet(t)
	// if err != nil t has invalid type restrictions. Fall back on a non-zero
	// hash.
	if err != nil {
		return 9151
	}
	return h.hashTermSet(terms)
}

func (h hasher) hashTermSet(terms []*types.Term) uint32 {
	hash := 9157 + 2*uint32(len(terms))
	for _, term := range terms {
		// term order is not significant.
		termHash := h.hash(term.Type())
		if term.Tilde() {
			termHash *= 9161
		}
		hash += 3 * termHash
	}
	return hash
}

// hashTypeParam returns the hash of a type parameter.
func (h hasher) hashTypeParam(t *types.TypeParam) uint32 {
	// Within the signature of a generic function, TypeParams are
	// identical if they have the same index and constraint, so we
	// hash them based on index.
	//
	// When we are outside a generic function, free TypeParams are
	// identical iff they are the same object, so we can use a
	// more discriminating hash consistent with object identity.
	// This optimization saves [Map] about 4% when hashing all the
	// types.Info.Types in the forward closure of net/http.
	if !h.inGenericSig {
		// Optimization: outside a generic function signature,
		// use a more discrimating hash consistent with object identity.
		return h.hashTypeName(t.Obj())
	}
	return 9173 + 3*uint32(t.Index())
}

var theSeed = maphash.MakeSeed()

// hashTypeName hashes the pointer of tname.
func (hasher) hashTypeName(tname *types.TypeName) uint32 {
	// Since types.Identical uses == to compare TypeNames,
	// the Hash function uses maphash.Comparable.
	hash := maphash.Comparable(theSeed, tname)
	return uint32(hash ^ (hash >> 32))
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
// (See comment in go/types.identical, case *Interface, for more.)
func (h hasher) shallowHash(t types.Type) uint32 {
	// t is the type of an interface method (Signature),
	// its params or results (Tuples), or their immediate
	// elements (mostly Slice, Pointer, Basic, Named),
	// so there's no need to optimize anything else.
	switch t := t.(type) {
	case *types.Alias:
		return h.shallowHash(types.Unalias(t))

	case *types.Signature:
		var hash uint32 = 604171
		if t.Variadic() {
			hash *= 971767
		}
		// The Signature/Tuple recursion is always finite
		// and invariably shallow.
		return hash + 1062599*h.shallowHash(t.Params()) + 1282529*h.shallowHash(t.Results())

	case *types.Tuple:
		n := t.Len()
		hash := 9137 + 2*uint32(n)
		for i := range n {
			hash += 53471161 * h.shallowHash(t.At(i).Type())
		}
		return hash

	case *types.Basic:
		return 45212177 * uint32(t.Kind())

	case *types.Array:
		return 1524181 + 2*uint32(t.Len())

	case *types.Slice:
		return 2690201

	case *types.Struct:
		return 3326489

	case *types.Pointer:
		return 4393139

	case *types.Union:
		return 562448657

	case *types.Interface:
		return 2124679 // no recursion here

	case *types.Map:
		return 9109

	case *types.Chan:
		return 9127

	case *types.Named:
		return h.hashTypeName(t.Obj())

	case *types.TypeParam:
		return h.hashTypeParam(t)
	}
	panic(fmt.Sprintf("shallowHash: %T: %v", t, t))
}
