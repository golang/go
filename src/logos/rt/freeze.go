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
// Identical slice headers share their copy. Views with different lengths
// or capacities copy separately. The copy includes elements up to capacity.
// Freeze panics on a reachable non-nil function, channel, or unsafe pointer
// because it cannot isolate that value's referenced state. Nil leaves remain
// nil. A nil p returns nil.
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

type freezer struct {
	visited map[freezeVisit]reflect.Value
}

type freezeVisit struct {
	typ reflect.Type
	ptr unsafe.Pointer
	n   int
	c   int
}

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
	case reflect.Func, reflect.Chan, reflect.UnsafePointer:
		if !src.IsNil() {
			panic("logos/rt: Freeze cannot isolate " + src.Kind().String())
		}
		dst.SetZero()
	default:
		dst.Set(readable(src))
	}
}

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

func (fr *freezer) copySlice(dst, src reflect.Value) {
	src = readable(src)
	if src.IsNil() {
		dst.Set(reflect.Zero(src.Type()))
		return
	}
	key := freezeVisit{typ: src.Type(), ptr: src.UnsafePointer(), n: src.Len(), c: src.Cap()}
	if prev, ok := fr.visited[key]; ok {
		dst.Set(prev)
		return
	}
	ns := reflect.MakeSlice(src.Type(), src.Cap(), src.Cap())
	view := ns.Slice3(0, src.Len(), src.Cap())
	fr.visited[key] = view
	if src.Type().Elem().Size() != 0 {
		full := src.Slice(0, src.Cap())
		for i := range full.Len() {
			fr.copy(settable(ns.Index(i)), readable(full.Index(i)))
		}
	}
	dst.Set(view)
}

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

func readable(src reflect.Value) reflect.Value {
	if src.CanInterface() {
		return src
	}
	return reflect.NewAt(src.Type(), unsafe.Pointer(src.UnsafeAddr())).Elem()
}

func settable(dst reflect.Value) reflect.Value {
	if dst.CanSet() {
		return dst
	}
	return reflect.NewAt(dst.Type(), unsafe.Pointer(dst.UnsafeAddr())).Elem()
}
