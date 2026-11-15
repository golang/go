// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"maps"
	"reflect"
	"unsafe"
)

// Freeze returns a deep copy of the value pointed to by p (SPEC T-201,
// M5-S6): the copy shares no mutable heap object with the source, so a
// later write to the source, or to any child object it points to, does
// not change the frozen view. .lgo code sees the result as const *T;
// Go importers see the plain *T declared here.
//
// The copy rebuilds pointer, map, slice, array, struct, and interface
// structure. Cycles and shared subobjects keep their shape through the
// visited state of one Freezer: a self-referential value terminates, a
// back edge resolves to the copy under construction, and two views of
// one child still view one copy. Nils stay nil. Identical slice headers
// share their copy. Views with different lengths or capacities copy
// separately. The copy includes elements up to capacity. Freeze panics
// on a reachable non-nil function, channel, or unsafe pointer because it
// cannot isolate that value's referenced state. Nil leaves remain nil. A
// nil p returns nil.
//
// Freeze performs one fresh allocation per reachable heap object plus
// its bookkeeping; it performs no synchronization, so mutating the
// source during a Freeze is a data race. Each call owns its visited
// state, so concurrent Freeze calls over one source are safe.
//
// The loGo/s lowering replaces every .lgo call whose T it can model with
// FreezeRoot and a copier generated for T. A Go caller, which no
// lowering sees, copies through FreezeDynamic.
func Freeze[T any](p *T) *T {
	return FreezeRoot(p, FreezeDynamic[T])
}

// Freezer holds the visited state of one Freeze call: each source
// pointer, slice header, and map already reached maps to its copy, so
// cycles terminate and shared children stay shared. Generated copiers
// and FreezeDynamic share one Freezer per call. The zero Freezer is
// ready to use.
//
// A copier func(fr, dst, src) receives in *dst the zero value or a
// shallow copy of *src and writes every reference it replaces. A nil
// copier copies by assignment, which suits a type that holds no
// pointer, slice, map, interface, func, chan, or unsafe.Pointer.
type Freezer struct {
	visited map[freezeKey]any
}

type freezeKey struct {
	typ  any
	ptr  unsafe.Pointer
	n, c int
}

func (fr *Freezer) remember(key freezeKey, v any) {
	if fr.visited == nil {
		fr.visited = make(map[freezeKey]any)
	}
	fr.visited[key] = v
}

// FreezeRoot returns a deep copy of *p made by copyElem, the copier of
// T, with the contract of Freeze. A nil p returns nil.
func FreezeRoot[T any](p *T, copyElem func(*Freezer, *T, *T)) *T {
	if p == nil {
		return nil
	}
	if copyElem == nil {
		np := new(T)
		*np = *p
		return np
	}
	return FreezePointer(&Freezer{}, p, copyElem)
}

// FreezePointer returns the copy of the object p points to: the copy fr
// already made for p, or a fresh object that copyElem fills after fr
// records it. A nil p stays nil.
func FreezePointer[P ~*E, E any](fr *Freezer, p P, copyElem func(*Freezer, *E, *E)) P {
	if p == nil {
		return nil
	}
	key := freezeKey{typ: (*P)(nil), ptr: unsafe.Pointer(p)}
	if prev, ok := fr.visited[key]; ok {
		return prev.(P)
	}
	np := P(new(E))
	fr.remember(key, np)
	if copyElem == nil {
		*np = *p
	} else {
		copyElem(fr, np, p)
	}
	return np
}

// FreezeSlice returns the copy of the slice header s: the copy fr
// already made for a header with the same type, data pointer, length,
// and capacity, or a fresh backing array of cap(s) elements that
// copyElem fills after fr records it. A nil s stays nil.
func FreezeSlice[S ~[]E, E any](fr *Freezer, s S, copyElem func(*Freezer, *E, *E)) S {
	if s == nil {
		return nil
	}
	key := freezeKey{typ: (*S)(nil), ptr: unsafe.Pointer(unsafe.SliceData(s)), n: len(s), c: cap(s)}
	if prev, ok := fr.visited[key]; ok {
		return prev.(S)
	}
	full := s[:cap(s)]
	ns := make(S, len(full))
	view := ns[:len(s)]
	fr.remember(key, view)
	if copyElem == nil {
		copy(ns, full)
		return view
	}
	for i := range full {
		copyElem(fr, &ns[i], &full[i])
	}
	return view
}

// FreezeMap returns the copy of the map m: the copy fr already made for
// m, or a fresh map whose entries copyKey and copyValue fill after fr
// records it. A nil m stays nil.
func FreezeMap[M ~map[K]V, K comparable, V any](fr *Freezer, m M, copyKey func(*Freezer, *K, *K), copyValue func(*Freezer, *V, *V)) M {
	if m == nil {
		return nil
	}
	key := freezeKey{typ: (*M)(nil), ptr: *(*unsafe.Pointer)(unsafe.Pointer(&m))}
	if prev, ok := fr.visited[key]; ok {
		return prev.(M)
	}
	if copyKey == nil && copyValue == nil {
		nm := maps.Clone(m)
		fr.remember(key, nm)
		return nm
	}
	nm := make(M, len(m))
	fr.remember(key, nm)
	for k, v := range m {
		nk, nv := k, v
		if copyKey != nil {
			copyKey(fr, &nk, &k)
		}
		if copyValue != nil {
			copyValue(fr, &nv, &v)
		}
		nm[nk] = nv
	}
	return nm
}

// FreezeReject panics with the Freeze rejection of a reachable non-nil
// leaf of kind "func", "chan", or "unsafe.Pointer".
func FreezeReject(kind string) {
	panic("logos/rt: Freeze cannot isolate " + kind)
}

// FreezeDynamic deep-copies *src into *dst through reflect and shares
// fr's visited state with generated copiers. Generated copiers call it
// only for a value whose structure is known at run time alone: the
// payload of an open interface, a closed-union value whose dynamic type
// is not a member, a type parameter, and a value whose type or field
// generated code cannot name (an unexported type of another package, or
// a struct of another package with an unexported or blank reference
// field, which it reads and writes through unsafe). Freeze uses it for
// every Go caller.
func FreezeDynamic[T any](fr *Freezer, dst, src *T) {
	fr.copy(reflect.ValueOf(dst).Elem(), reflect.ValueOf(src).Elem())
}

func (fr *Freezer) copy(dst, src reflect.Value) {
	switch src.Kind() {
	case reflect.Pointer:
		fr.copyPointer(dst, src)
	case reflect.Interface:
		fr.copyInterface(dst, src)
	case reflect.Struct:
		fr.copyFields(dst, src)
	case reflect.Array:
		fr.copyElems(dst, src)
	case reflect.Slice:
		fr.copySlice(dst, src)
	case reflect.Map:
		fr.copyMap(dst, src)
	case reflect.Func, reflect.Chan, reflect.UnsafePointer:
		copyLeaf(dst, src)
	default:
		dst.Set(readable(src))
	}
}

func (fr *Freezer) copyFields(dst, src reflect.Value) {
	for i := range src.NumField() {
		fr.copy(settable(dst.Field(i)), readable(src.Field(i)))
	}
}

func (fr *Freezer) copyElems(dst, src reflect.Value) {
	for i := range src.Len() {
		fr.copy(settable(dst.Index(i)), readable(src.Index(i)))
	}
}

func copyLeaf(dst, src reflect.Value) {
	if !src.IsNil() {
		FreezeReject(src.Kind().String())
	}
	dst.SetZero()
}

func dynamicKey(src reflect.Value, n, c int) freezeKey {
	return freezeKey{typ: reflect.Zero(reflect.PointerTo(src.Type())).Interface(), ptr: src.UnsafePointer(), n: n, c: c}
}

func (fr *Freezer) copyPointer(dst, src reflect.Value) {
	src = readable(src)
	if src.IsNil() {
		dst.SetZero()
		return
	}
	key := dynamicKey(src, 0, 0)
	if prev, ok := fr.visited[key]; ok {
		dst.Set(reflect.ValueOf(prev))
		return
	}
	np := reflect.New(src.Type().Elem())
	fr.remember(key, np.Convert(src.Type()).Interface())
	fr.copy(np.Elem(), src.Elem())
	dst.Set(np)
}

func (fr *Freezer) copyInterface(dst, src reflect.Value) {
	src = readable(src)
	if src.IsNil() {
		dst.SetZero()
		return
	}
	dyn := src.Elem()
	dst.Set(fr.copyStaged(dyn.Type(), dyn))
}

func (fr *Freezer) copyStaged(t reflect.Type, v reflect.Value) reflect.Value {
	staged := reflect.New(t).Elem()
	staged.Set(v)
	nv := reflect.New(t).Elem()
	fr.copy(nv, staged)
	return nv
}

func (fr *Freezer) copySlice(dst, src reflect.Value) {
	src = readable(src)
	if src.IsNil() {
		dst.SetZero()
		return
	}
	key := dynamicKey(src, src.Len(), src.Cap())
	if prev, ok := fr.visited[key]; ok {
		dst.Set(reflect.ValueOf(prev))
		return
	}
	ns := reflect.MakeSlice(src.Type(), src.Cap(), src.Cap())
	view := ns.Slice3(0, src.Len(), src.Cap())
	fr.remember(key, view.Interface())
	if src.Type().Elem().Size() != 0 {
		full := src.Slice(0, src.Cap())
		for i := range full.Len() {
			fr.copy(settable(ns.Index(i)), readable(full.Index(i)))
		}
	}
	dst.Set(view)
}

func (fr *Freezer) copyMap(dst, src reflect.Value) {
	src = readable(src)
	if src.IsNil() {
		dst.SetZero()
		return
	}
	key := dynamicKey(src, 0, 0)
	if prev, ok := fr.visited[key]; ok {
		dst.Set(reflect.ValueOf(prev))
		return
	}
	nm := reflect.MakeMapWithSize(src.Type(), src.Len())
	fr.remember(key, nm.Interface())
	kt, et := src.Type().Key(), src.Type().Elem()
	iter := src.MapRange()
	for iter.Next() {
		nk := fr.copyStaged(kt, iter.Key())
		nm.SetMapIndex(nk, fr.copyStaged(et, iter.Value()))
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
