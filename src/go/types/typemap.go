// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements TypeMap, a modified and simplified
// version of x/tools/go/types/typeutil/map.go.

package types

import (
	"fmt"
	"reflect"
)

// A TypeMap is a map from a Type implementation (a pointer) to an
// arbitrary interface{} value using the Identical relation for type
// equality.
// The zero value of a TypeMap is ready to use and corresponds to an
// empty map.
type TypeMap struct {
	hcache  map[Type]uint64    // maps types to hash values
	buckets map[uint64][]entry // maps hash values to buckets
	length  int                // number of map entries
}

type entry struct {
	key Type
	val interface{}
}

func (m *TypeMap) Insert(key Type, val interface{}) interface{} {
	if m.buckets == nil {
		m.hcache = map[Type]uint64{}
		m.buckets = map[uint64][]entry{m.hash(key): {{key, val}}}
		m.length = 1
		return nil
	}

	h := m.hash(key)
	b := m.buckets[h]
	for i := range b {
		if e := &b[i]; Identical(key, e.key) {
			old := e.val
			e.val = val
			return old
		}
	}
	m.buckets[h] = append(b, entry{key, val})
	m.length++
	return nil
}

// At returns the type map entry for the given key.
// The result is nil if the entry is not present.
func (m *TypeMap) At(key Type) interface{} {
	if m != nil && m.buckets != nil {
		h := m.hash(key)
		b := m.buckets[h]
		for i := range b {
			if e := &b[i]; Identical(key, e.key) {
				return e.val
			}
		}
	}
	return nil
}

// Len returns the number of map entries.
func (m *TypeMap) Len() int {
	if m != nil {
		return m.length
	}
	return 0
}

// hash computes a hash value for the given type t such that
// Identical(t, t') => hash(t) == hash(t').
func (m *TypeMap) hash(t Type) uint64 {
	h, ok := m.hcache[t]
	if !ok {
		h = m.hashFor(t)
		m.hcache[t] = h
	}
	return h
}

// hashFor computes the hash of t.
func (m *TypeMap) hashFor(t Type) (h uint64) {
	// See Identical for rationale.
	switch t := t.(type) {
	case *Basic:
		h = uint64(t.Kind())

	case *Array:
		h = 9043 + 2*uint64(t.Len()) + 3*m.hash(t.Elem())

	case *Slice:
		h = 9049 + 2*m.hash(t.Elem())

	case *Struct:
		h = 9059
		for i, n := 0, t.NumFields(); i < n; i++ {
			f := t.Field(i)
			if f.Anonymous() {
				h += 8861
			}
			h += hashString(t.Tag(i))
			h += hashString(f.Name()) // (ignore f.Pkg)
			h += m.hash(f.Type())
		}

	case *Pointer:
		h = 9067 + 2*m.hash(t.Elem())

	case *Signature:
		h = 9091
		if t.Variadic() {
			h *= 8863
		}
		h += 3*m.hashTuple(t.Params()) + 5*m.hashTuple(t.Results())

	case *Interface:
		h = 9103
		for i, n := 0, t.NumMethods(); i < n; i++ {
			// See identicalMethods for rationale.
			// Method order is not significant.
			// Ignore m.Pkg().
			tm := t.Method(i)
			h += 3*hashString(tm.Name()) + 5*m.hash(tm.Type())
		}

	case *Map:
		h = 9109 + 2*m.hash(t.Key()) + 3*m.hash(t.Elem())

	case *Chan:
		h = 9127 + 2*uint64(t.Dir()) + 3*m.hash(t.Elem())

	case *Named:
		// Not safe with a copying GC; objects may move.
		h = uint64(reflect.ValueOf(t.Obj()).Pointer())

	case *Tuple:
		h = m.hashTuple(t)

	default:
		panic(fmt.Sprintf("unknown type %T", t))
	}

	return h
}

// hashString implements the Fowler-Noll-Vo FNV-1a 64bit hash function.
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
func hashString(s string) (h uint64) {
	h = 14695981039346656037
	for i := 0; i < len(s); i++ {
		h = (h ^ uint64(s[i])) * 1099511628211
	}
	return
}

func (m *TypeMap) hashTuple(tuple *Tuple) uint64 {
	// See go/types.identicalTypes for rationale.
	n := tuple.Len()
	h := 9137 + 2*uint64(n)
	for i := 0; i < n; i++ {
		h += 3 * m.hash(tuple.At(i).Type())
	}
	return h
}
