// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.swissmap

package reflect

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

// mapType represents a map type.
type mapType struct {
	abi.OldMapType
}

// Pushed from runtime.

//go:noescape
func mapiterinit(t *abi.Type, m unsafe.Pointer, it *hiter)

//go:noescape
func mapiternext(it *hiter)

func (t *rtype) Key() Type {
	if t.Kind() != Map {
		panic("reflect: Key of non-map type " + t.String())
	}
	tt := (*mapType)(unsafe.Pointer(t))
	return toType(tt.Key)
}

// MapOf returns the map type with the given key and element types.
// For example, if k represents int and e represents string,
// MapOf(k, e) represents map[int]string.
//
// If the key type is not a valid map key type (that is, if it does
// not implement Go's == operator), MapOf panics.
func MapOf(key, elem Type) Type {
	ktyp := key.common()
	etyp := elem.common()

	if ktyp.Equal == nil {
		panic("reflect.MapOf: invalid key type " + stringFor(ktyp))
	}

	// Look in cache.
	ckey := cacheKey{Map, ktyp, etyp, 0}
	if mt, ok := lookupCache.Load(ckey); ok {
		return mt.(Type)
	}

	// Look in known types.
	s := "map[" + stringFor(ktyp) + "]" + stringFor(etyp)
	for _, tt := range typesByString(s) {
		mt := (*mapType)(unsafe.Pointer(tt))
		if mt.Key == ktyp && mt.Elem == etyp {
			ti, _ := lookupCache.LoadOrStore(ckey, toRType(tt))
			return ti.(Type)
		}
	}

	// Make a map type.
	// Note: flag values must match those used in the TMAP case
	// in ../cmd/compile/internal/reflectdata/reflect.go:writeType.
	var imap any = (map[unsafe.Pointer]unsafe.Pointer)(nil)
	mt := **(**mapType)(unsafe.Pointer(&imap))
	mt.Str = resolveReflectName(newName(s, "", false, false))
	mt.TFlag = abi.TFlagDirectIface
	mt.Hash = fnv1(etyp.Hash, 'm', byte(ktyp.Hash>>24), byte(ktyp.Hash>>16), byte(ktyp.Hash>>8), byte(ktyp.Hash))
	mt.Key = ktyp
	mt.Elem = etyp
	mt.Bucket = bucketOf(ktyp, etyp)
	mt.Hasher = func(p unsafe.Pointer, seed uintptr) uintptr {
		return typehash(ktyp, p, seed)
	}
	mt.Flags = 0
	if ktyp.Size_ > abi.OldMapMaxKeyBytes {
		mt.KeySize = uint8(goarch.PtrSize)
		mt.Flags |= 1 // indirect key
	} else {
		mt.KeySize = uint8(ktyp.Size_)
	}
	if etyp.Size_ > abi.OldMapMaxElemBytes {
		mt.ValueSize = uint8(goarch.PtrSize)
		mt.Flags |= 2 // indirect value
	} else {
		mt.ValueSize = uint8(etyp.Size_)
	}
	mt.BucketSize = uint16(mt.Bucket.Size_)
	if isReflexive(ktyp) {
		mt.Flags |= 4
	}
	if needKeyUpdate(ktyp) {
		mt.Flags |= 8
	}
	if hashMightPanic(ktyp) {
		mt.Flags |= 16
	}
	mt.PtrToThis = 0

	ti, _ := lookupCache.LoadOrStore(ckey, toRType(&mt.Type))
	return ti.(Type)
}

func bucketOf(ktyp, etyp *abi.Type) *abi.Type {
	if ktyp.Size_ > abi.OldMapMaxKeyBytes {
		ktyp = ptrTo(ktyp)
	}
	if etyp.Size_ > abi.OldMapMaxElemBytes {
		etyp = ptrTo(etyp)
	}

	// Prepare GC data if any.
	// A bucket is at most bucketSize*(1+maxKeySize+maxValSize)+ptrSize bytes,
	// or 2064 bytes, or 258 pointer-size words, or 33 bytes of pointer bitmap.
	// Note that since the key and value are known to be <= 128 bytes,
	// they're guaranteed to have bitmaps instead of GC programs.
	var gcdata *byte
	var ptrdata uintptr

	size := abi.OldMapBucketCount*(1+ktyp.Size_+etyp.Size_) + goarch.PtrSize
	if size&uintptr(ktyp.Align_-1) != 0 || size&uintptr(etyp.Align_-1) != 0 {
		panic("reflect: bad size computation in MapOf")
	}

	if ktyp.Pointers() || etyp.Pointers() {
		nptr := (abi.OldMapBucketCount*(1+ktyp.Size_+etyp.Size_) + goarch.PtrSize) / goarch.PtrSize
		n := (nptr + 7) / 8

		// Runtime needs pointer masks to be a multiple of uintptr in size.
		n = (n + goarch.PtrSize - 1) &^ (goarch.PtrSize - 1)
		mask := make([]byte, n)
		base := uintptr(abi.OldMapBucketCount / goarch.PtrSize)

		if ktyp.Pointers() {
			emitGCMask(mask, base, ktyp, abi.OldMapBucketCount)
		}
		base += abi.OldMapBucketCount * ktyp.Size_ / goarch.PtrSize

		if etyp.Pointers() {
			emitGCMask(mask, base, etyp, abi.OldMapBucketCount)
		}
		base += abi.OldMapBucketCount * etyp.Size_ / goarch.PtrSize

		word := base
		mask[word/8] |= 1 << (word % 8)
		gcdata = &mask[0]
		ptrdata = (word + 1) * goarch.PtrSize

		// overflow word must be last
		if ptrdata != size {
			panic("reflect: bad layout computation in MapOf")
		}
	}

	b := &abi.Type{
		Align_:   goarch.PtrSize,
		Size_:    size,
		Kind_:    abi.Struct,
		PtrBytes: ptrdata,
		GCData:   gcdata,
	}
	s := "bucket(" + stringFor(ktyp) + "," + stringFor(etyp) + ")"
	b.Str = resolveReflectName(newName(s, "", false, false))
	return b
}

var stringType = rtypeOf("")

// MapIndex returns the value associated with key in the map v.
// It panics if v's Kind is not [Map].
// It returns the zero Value if key is not found in the map or if v represents a nil map.
// As in Go, the key's value must be assignable to the map's key type.
func (v Value) MapIndex(key Value) Value {
	v.mustBe(Map)
	tt := (*mapType)(unsafe.Pointer(v.typ()))

	// Do not require key to be exported, so that DeepEqual
	// and other programs can use all the keys returned by
	// MapKeys as arguments to MapIndex. If either the map
	// or the key is unexported, though, the result will be
	// considered unexported. This is consistent with the
	// behavior for structs, which allow read but not write
	// of unexported fields.

	var e unsafe.Pointer
	if (tt.Key == stringType || key.kind() == String) && tt.Key == key.typ() && tt.Elem.Size() <= abi.OldMapMaxElemBytes {
		k := *(*string)(key.ptr)
		e = mapaccess_faststr(v.typ(), v.pointer(), k)
	} else {
		key = key.assignTo("reflect.Value.MapIndex", tt.Key, nil)
		var k unsafe.Pointer
		if key.flag&flagIndir != 0 {
			k = key.ptr
		} else {
			k = unsafe.Pointer(&key.ptr)
		}
		e = mapaccess(v.typ(), v.pointer(), k)
	}
	if e == nil {
		return Value{}
	}
	typ := tt.Elem
	fl := (v.flag | key.flag).ro()
	fl |= flag(typ.Kind())
	return copyVal(typ, fl, e)
}

// MapKeys returns a slice containing all the keys present in the map,
// in unspecified order.
// It panics if v's Kind is not [Map].
// It returns an empty slice if v represents a nil map.
func (v Value) MapKeys() []Value {
	v.mustBe(Map)
	tt := (*mapType)(unsafe.Pointer(v.typ()))
	keyType := tt.Key

	fl := v.flag.ro() | flag(keyType.Kind())

	m := v.pointer()
	mlen := int(0)
	if m != nil {
		mlen = maplen(m)
	}
	var it hiter
	mapiterinit(v.typ(), m, &it)
	a := make([]Value, mlen)
	var i int
	for i = 0; i < len(a); i++ {
		key := it.key
		if key == nil {
			// Someone deleted an entry from the map since we
			// called maplen above. It's a data race, but nothing
			// we can do about it.
			break
		}
		a[i] = copyVal(keyType, fl, key)
		mapiternext(&it)
	}
	return a[:i]
}

// hiter's structure matches runtime.hiter's structure.
// Having a clone here allows us to embed a map iterator
// inside type MapIter so that MapIters can be re-used
// without doing any allocations.
type hiter struct {
	key         unsafe.Pointer
	elem        unsafe.Pointer
	t           unsafe.Pointer
	h           unsafe.Pointer
	buckets     unsafe.Pointer
	bptr        unsafe.Pointer
	overflow    *[]unsafe.Pointer
	oldoverflow *[]unsafe.Pointer
	startBucket uintptr
	offset      uint8
	wrapped     bool
	B           uint8
	i           uint8
	bucket      uintptr
	checkBucket uintptr
	clearSeq    uint64
}

func (h *hiter) initialized() bool {
	return h.t != nil
}

// A MapIter is an iterator for ranging over a map.
// See [Value.MapRange].
type MapIter struct {
	m     Value
	hiter hiter
}

// Key returns the key of iter's current map entry.
func (iter *MapIter) Key() Value {
	if !iter.hiter.initialized() {
		panic("MapIter.Key called before Next")
	}
	iterkey := iter.hiter.key
	if iterkey == nil {
		panic("MapIter.Key called on exhausted iterator")
	}

	t := (*mapType)(unsafe.Pointer(iter.m.typ()))
	ktype := t.Key
	return copyVal(ktype, iter.m.flag.ro()|flag(ktype.Kind()), iterkey)
}

// SetIterKey assigns to v the key of iter's current map entry.
// It is equivalent to v.Set(iter.Key()), but it avoids allocating a new Value.
// As in Go, the key must be assignable to v's type and
// must not be derived from an unexported field.
// It panics if [Value.CanSet] returns false.
func (v Value) SetIterKey(iter *MapIter) {
	if !iter.hiter.initialized() {
		panic("reflect: Value.SetIterKey called before Next")
	}
	iterkey := iter.hiter.key
	if iterkey == nil {
		panic("reflect: Value.SetIterKey called on exhausted iterator")
	}

	v.mustBeAssignable()
	var target unsafe.Pointer
	if v.kind() == Interface {
		target = v.ptr
	}

	t := (*mapType)(unsafe.Pointer(iter.m.typ()))
	ktype := t.Key

	iter.m.mustBeExported() // do not let unexported m leak
	key := Value{ktype, iterkey, iter.m.flag | flag(ktype.Kind()) | flagIndir}
	key = key.assignTo("reflect.MapIter.SetKey", v.typ(), target)
	typedmemmove(v.typ(), v.ptr, key.ptr)
}

// Value returns the value of iter's current map entry.
func (iter *MapIter) Value() Value {
	if !iter.hiter.initialized() {
		panic("MapIter.Value called before Next")
	}
	iterelem := iter.hiter.elem
	if iterelem == nil {
		panic("MapIter.Value called on exhausted iterator")
	}

	t := (*mapType)(unsafe.Pointer(iter.m.typ()))
	vtype := t.Elem
	return copyVal(vtype, iter.m.flag.ro()|flag(vtype.Kind()), iterelem)
}

// SetIterValue assigns to v the value of iter's current map entry.
// It is equivalent to v.Set(iter.Value()), but it avoids allocating a new Value.
// As in Go, the value must be assignable to v's type and
// must not be derived from an unexported field.
// It panics if [Value.CanSet] returns false.
func (v Value) SetIterValue(iter *MapIter) {
	if !iter.hiter.initialized() {
		panic("reflect: Value.SetIterValue called before Next")
	}
	iterelem := iter.hiter.elem
	if iterelem == nil {
		panic("reflect: Value.SetIterValue called on exhausted iterator")
	}

	v.mustBeAssignable()
	var target unsafe.Pointer
	if v.kind() == Interface {
		target = v.ptr
	}

	t := (*mapType)(unsafe.Pointer(iter.m.typ()))
	vtype := t.Elem

	iter.m.mustBeExported() // do not let unexported m leak
	elem := Value{vtype, iterelem, iter.m.flag | flag(vtype.Kind()) | flagIndir}
	elem = elem.assignTo("reflect.MapIter.SetValue", v.typ(), target)
	typedmemmove(v.typ(), v.ptr, elem.ptr)
}

// Next advances the map iterator and reports whether there is another
// entry. It returns false when iter is exhausted; subsequent
// calls to [MapIter.Key], [MapIter.Value], or [MapIter.Next] will panic.
func (iter *MapIter) Next() bool {
	if !iter.m.IsValid() {
		panic("MapIter.Next called on an iterator that does not have an associated map Value")
	}
	if !iter.hiter.initialized() {
		mapiterinit(iter.m.typ(), iter.m.pointer(), &iter.hiter)
	} else {
		if iter.hiter.key == nil {
			panic("MapIter.Next called on exhausted iterator")
		}
		mapiternext(&iter.hiter)
	}
	return iter.hiter.key != nil
}

// Reset modifies iter to iterate over v.
// It panics if v's Kind is not [Map] and v is not the zero Value.
// Reset(Value{}) causes iter to not to refer to any map,
// which may allow the previously iterated-over map to be garbage collected.
func (iter *MapIter) Reset(v Value) {
	if v.IsValid() {
		v.mustBe(Map)
	}
	iter.m = v
	iter.hiter = hiter{}
}

// MapRange returns a range iterator for a map.
// It panics if v's Kind is not [Map].
//
// Call [MapIter.Next] to advance the iterator, and [MapIter.Key]/[MapIter.Value] to access each entry.
// [MapIter.Next] returns false when the iterator is exhausted.
// MapRange follows the same iteration semantics as a range statement.
//
// Example:
//
//	iter := reflect.ValueOf(m).MapRange()
//	for iter.Next() {
//		k := iter.Key()
//		v := iter.Value()
//		...
//	}
func (v Value) MapRange() *MapIter {
	// This is inlinable to take advantage of "function outlining".
	// The allocation of MapIter can be stack allocated if the caller
	// does not allow it to escape.
	// See https://blog.filippo.io/efficient-go-apis-with-the-inliner/
	if v.kind() != Map {
		v.panicNotMap()
	}
	return &MapIter{m: v}
}

// SetMapIndex sets the element associated with key in the map v to elem.
// It panics if v's Kind is not [Map].
// If elem is the zero Value, SetMapIndex deletes the key from the map.
// Otherwise if v holds a nil map, SetMapIndex will panic.
// As in Go, key's elem must be assignable to the map's key type,
// and elem's value must be assignable to the map's elem type.
func (v Value) SetMapIndex(key, elem Value) {
	v.mustBe(Map)
	v.mustBeExported()
	key.mustBeExported()
	tt := (*mapType)(unsafe.Pointer(v.typ()))

	if (tt.Key == stringType || key.kind() == String) && tt.Key == key.typ() && tt.Elem.Size() <= abi.OldMapMaxElemBytes {
		k := *(*string)(key.ptr)
		if elem.typ() == nil {
			mapdelete_faststr(v.typ(), v.pointer(), k)
			return
		}
		elem.mustBeExported()
		elem = elem.assignTo("reflect.Value.SetMapIndex", tt.Elem, nil)
		var e unsafe.Pointer
		if elem.flag&flagIndir != 0 {
			e = elem.ptr
		} else {
			e = unsafe.Pointer(&elem.ptr)
		}
		mapassign_faststr(v.typ(), v.pointer(), k, e)
		return
	}

	key = key.assignTo("reflect.Value.SetMapIndex", tt.Key, nil)
	var k unsafe.Pointer
	if key.flag&flagIndir != 0 {
		k = key.ptr
	} else {
		k = unsafe.Pointer(&key.ptr)
	}
	if elem.typ() == nil {
		mapdelete(v.typ(), v.pointer(), k)
		return
	}
	elem.mustBeExported()
	elem = elem.assignTo("reflect.Value.SetMapIndex", tt.Elem, nil)
	var e unsafe.Pointer
	if elem.flag&flagIndir != 0 {
		e = elem.ptr
	} else {
		e = unsafe.Pointer(&elem.ptr)
	}
	mapassign(v.typ(), v.pointer(), k, e)
}

// Force slow panicking path not inlined, so it won't add to the
// inlining budget of the caller.
// TODO: undo when the inliner is no longer bottom-up only.
//
//go:noinline
func (f flag) panicNotMap() {
	f.mustBe(Map)
}
