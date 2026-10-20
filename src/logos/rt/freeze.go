// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"reflect"
	"unsafe"
)

// Freeze returns a deep copy of the value pointed to by p (SPEC T-201,
// M5-S6): the copy shares no mutable heap object with the source, so a
// later write to the source, or to any child object it points to, does
// not change the frozen view. .lgo code sees the result as const *T;
// Go importers see the plain *T declared here.
//
// The copy walks the value with reflect and rebuilds pointer, map,
// slice, array, struct, and interface structure. Cycles and shared
// subobjects keep their shape through a visited map: a self-referential
// value terminates, a back edge resolves to the copy under construction,
// and two views of one child still view one copy. Nils stay nil.
// Identical slice headers share their copy; slices that share a backing
// array under different lengths copy separately. Functions, channels,
// and unsafe pointers copy by reference: they name code or rendezvous,
// not data, and have no copy semantics. A nil p returns nil.
//
// Freeze performs one fresh allocation per reachable heap object plus
// its bookkeeping; it performs no synchronization, so mutating the
// source during a Freeze is a data race. Each call owns its visited
// map, so concurrent Freeze calls over one source are safe.
func Freeze[T any](p *T) *T {
	if p == nil {
		return nil
	}
	fr := &freezer{visited: make(map[freezeVisit]reflect.Value)}
	src := reflect.ValueOf(p).Elem()
	dst := reflect.New(src.Type())
	fr.visited[freezeVisit{typ: reflect.TypeOf(p), ptr: unsafe.Pointer(p)}] = dst
	fr.copy(dst.Elem(), src)
	return dst.Interface().(*T)
}

// freezer holds one Freeze call's visited map: every heap object the
// copy already rebuilt, keyed by the source object's identity. The key
// carries the type so two objects that share an address under
// different types never collide, and the length for slices so headers
// over one backing array with different lengths never collide.
type freezer struct {
	visited map[freezeVisit]reflect.Value
}

// freezeVisit identifies one source heap object: a pointer, map, or
// slice header. n holds the slice length and is zero otherwise.
type freezeVisit struct {
	typ reflect.Type
	ptr unsafe.Pointer
	n   int
}

// copy rebuilds src into dst. dst is always addressable and settable:
// the root comes from reflect.New and every nested dst is a fresh
// allocation, an element of one, or a field cleared through unsafe.
// src is either directly readable or addressable, so unexported fields
// resolve through readable below; map keys and values arrive staged in
// addressable temporaries because map results are copies.
func (fr *freezer) copy(dst, src reflect.Value) {
	switch src.Kind() {
	case reflect.Pointer:
		fr.copyPointer(dst, src)
	case reflect.Interface:
		fr.copyInterface(dst, src)
	case reflect.Struct:
		st := src.Type()
		for i := range st.NumField() {
			fr.copy(settable(dst.Field(i)), readable(src.Field(i)))
		}
	case reflect.Array:
		for i := range src.Len() {
			fr.copy(settable(dst.Index(i)), readable(src.Index(i)))
		}
	case reflect.Slice:
		fr.copySlice(dst, src)
	case reflect.Map:
		fr.copyMap(dst, src)
	default:
		dst.Set(readable(src))
	}
}

// copyPointer rebuilds one pointer: nil stays nil, a revisited source
// resolves to its copy (this is what keeps cycles terminating and
// shared children shared), and a fresh source allocates before the
// element copy so a back edge to the pointer under construction lands.
func (fr *freezer) copyPointer(dst, src reflect.Value) {
	src = readable(src)
	if src.IsNil() {
		dst.Set(reflect.Zero(src.Type()))
		return
	}
	key := freezeVisit{typ: src.Type(), ptr: src.UnsafePointer()}
	if prev, ok := fr.visited[key]; ok {
		dst.Set(prev)
		return
	}
	np := reflect.New(src.Type().Elem())
	fr.visited[key] = np
	fr.copy(np.Elem(), src.Elem())
	dst.Set(np)
}

// copyInterface rebuilds one interface value: nil stays nil and a live
// payload copies into a fresh value of its dynamic type. The payload
// stages through an addressable temporary so unexported fields of a
// struct payload stay reachable.
func (fr *freezer) copyInterface(dst, src reflect.Value) {
	src = readable(src)
	if src.IsNil() {
		dst.Set(reflect.Zero(src.Type()))
		return
	}
	dyn := src.Elem()
	staged := reflect.New(dyn.Type()).Elem()
	staged.Set(dyn)
	nv := reflect.New(dyn.Type()).Elem()
	fr.copy(nv, staged)
	dst.Set(nv)
}

// copySlice rebuilds one slice: nil stays nil, an empty slice keeps its
// shape with no elements to visit, a revisited header resolves to its
// copy, and a fresh header allocates before the element copies so a
// cycle through the elements lands.
func (fr *freezer) copySlice(dst, src reflect.Value) {
	src = readable(src)
	if src.IsNil() {
		dst.Set(reflect.Zero(src.Type()))
		return
	}
	if src.Len() == 0 {
		dst.Set(reflect.MakeSlice(src.Type(), 0, src.Cap()))
		return
	}
	key := freezeVisit{typ: src.Type(), ptr: src.UnsafePointer(), n: src.Len()}
	if prev, ok := fr.visited[key]; ok {
		dst.Set(prev)
		return
	}
	ns := reflect.MakeSlice(src.Type(), src.Len(), src.Cap())
	fr.visited[key] = ns
	for i := range src.Len() {
		fr.copy(settable(ns.Index(i)), readable(src.Index(i)))
	}
	dst.Set(ns)
}

// copyMap rebuilds one map: nil stays nil, a revisited map resolves to
// its copy, and a fresh map allocates before the entries copy so a
// cycle through keys or values lands. Keys and values stage through
// addressable temporaries because map results are unaddressable
// copies.
func (fr *freezer) copyMap(dst, src reflect.Value) {
	src = readable(src)
	if src.IsNil() {
		dst.Set(reflect.Zero(src.Type()))
		return
	}
	key := freezeVisit{typ: src.Type(), ptr: src.UnsafePointer()}
	if prev, ok := fr.visited[key]; ok {
		dst.Set(prev)
		return
	}
	nm := reflect.MakeMapWithSize(src.Type(), src.Len())
	fr.visited[key] = nm
	kt, et := src.Type().Key(), src.Type().Elem()
	iter := src.MapRange()
	for iter.Next() {
		sk := reflect.New(kt).Elem()
		sk.Set(iter.Key())
		nk := reflect.New(kt).Elem()
		fr.copy(nk, sk)
		sv := reflect.New(et).Elem()
		sv.Set(iter.Value())
		nv := reflect.New(et).Elem()
		fr.copy(nv, sv)
		nm.SetMapIndex(nk, nv)
	}
	dst.Set(nm)
}

// readable returns src in a form Set accepts: directly readable values
// pass through, while values obtained through unexported fields are
// re-expressed through unsafe without their read-only flag. The
// unexported case needs src addressable, which holds for every struct,
// element, and staged value Freeze visits.
func readable(src reflect.Value) reflect.Value {
	if src.CanInterface() {
		return src
	}
	return reflect.NewAt(src.Type(), unsafe.Pointer(src.UnsafeAddr())).Elem()
}

// settable returns dst in a form Set targets: settable destinations
// pass through, while unexported fields are re-expressed through
// unsafe without their read-only flag. dst is always addressable by
// the copy invariant above.
func settable(dst reflect.Value) reflect.Value {
	if dst.CanSet() {
		return dst
	}
	return reflect.NewAt(dst.Type(), unsafe.Pointer(dst.UnsafeAddr())).Elem()
}
