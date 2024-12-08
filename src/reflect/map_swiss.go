// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package reflect

import (
	"internal/abi"
	"internal/runtime/maps"
	"unsafe"
)

// mapType represents a map type.
type mapType struct {
	abi.SwissMapType
}

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

	group, slot := groupAndSlotOf(key, elem)

	// Make a map type.
	// Note: flag values must match those used in the TMAP case
	// in ../cmd/compile/internal/reflectdata/reflect.go:writeType.
	var imap any = (map[unsafe.Pointer]unsafe.Pointer)(nil)
	mt := **(**mapType)(unsafe.Pointer(&imap))
	mt.Str = resolveReflectName(newName(s, "", false, false))
	mt.TFlag = 0
	mt.Hash = fnv1(etyp.Hash, 'm', byte(ktyp.Hash>>24), byte(ktyp.Hash>>16), byte(ktyp.Hash>>8), byte(ktyp.Hash))
	mt.Key = ktyp
	mt.Elem = etyp
	mt.Group = group.common()
	mt.Hasher = func(p unsafe.Pointer, seed uintptr) uintptr {
		return typehash(ktyp, p, seed)
	}
	mt.GroupSize = mt.Group.Size()
	mt.SlotSize = slot.Size()
	mt.ElemOff = slot.Field(1).Offset
	mt.Flags = 0
	if needKeyUpdate(ktyp) {
		mt.Flags |= abi.SwissMapNeedKeyUpdate
	}
	if hashMightPanic(ktyp) {
		mt.Flags |= abi.SwissMapHashMightPanic
	}
	if ktyp.Size_ > abi.SwissMapMaxKeyBytes {
		mt.Flags |= abi.SwissMapIndirectKey
	}
	if etyp.Size_ > abi.SwissMapMaxKeyBytes {
		mt.Flags |= abi.SwissMapIndirectElem
	}
	mt.PtrToThis = 0

	ti, _ := lookupCache.LoadOrStore(ckey, toRType(&mt.Type))
	return ti.(Type)
}

func groupAndSlotOf(ktyp, etyp Type) (Type, Type) {
	// type group struct {
	//     ctrl uint64
	//     slots [abi.SwissMapGroupSlots]struct {
	//         key  keyType
	//         elem elemType
	//     }
	// }

	if ktyp.Size() > abi.SwissMapMaxKeyBytes {
		ktyp = PointerTo(ktyp)
	}
	if etyp.Size() > abi.SwissMapMaxElemBytes {
		etyp = PointerTo(etyp)
	}

	fields := []StructField{
		{
			Name: "Key",
			Type: ktyp,
		},
		{
			Name: "Elem",
			Type: etyp,
		},
	}
	slot := StructOf(fields)

	fields = []StructField{
		{
			Name: "Ctrl",
			Type: TypeFor[uint64](),
		},
		{
			Name: "Slots",
			Type: ArrayOf(abi.SwissMapGroupSlots, slot),
		},
	}
	group := StructOf(fields)
	return group, slot
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
	if (tt.Key == stringType || key.kind() == String) && tt.Key == key.typ() && tt.Elem.Size() <= abi.SwissMapMaxElemBytes {
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
	var it maps.Iter
	mapiterinit(v.typ(), m, &it)
	a := make([]Value, mlen)
	var i int
	for i = 0; i < len(a); i++ {
		key := it.Key()
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

// A MapIter is an iterator for ranging over a map.
// See [Value.MapRange].
type MapIter struct {
	m     Value
	hiter maps.Iter
}

// TODO(prattmic): only for sharing the linkname declarations with old maps.
// Remove with old maps.
type hiter = maps.Iter

// Key returns the key of iter's current map entry.
func (iter *MapIter) Key() Value {
	if !iter.hiter.Initialized() {
		panic("MapIter.Key called before Next")
	}
	iterkey := iter.hiter.Key()
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
func (v Value) SetIterKey(iter *MapIter) {
	if !iter.hiter.Initialized() {
		panic("reflect: Value.SetIterKey called before Next")
	}
	iterkey := iter.hiter.Key()
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
	if !iter.hiter.Initialized() {
		panic("MapIter.Value called before Next")
	}
	iterelem := iter.hiter.Elem()
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
func (v Value) SetIterValue(iter *MapIter) {
	if !iter.hiter.Initialized() {
		panic("reflect: Value.SetIterValue called before Next")
	}
	iterelem := iter.hiter.Elem()
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
	if !iter.hiter.Initialized() {
		mapiterinit(iter.m.typ(), iter.m.pointer(), &iter.hiter)
	} else {
		if iter.hiter.Key() == nil {
			panic("MapIter.Next called on exhausted iterator")
		}
		mapiternext(&iter.hiter)
	}
	return iter.hiter.Key() != nil
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
	iter.hiter = maps.Iter{}
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

	if (tt.Key == stringType || key.kind() == String) && tt.Key == key.typ() && tt.Elem.Size() <= abi.SwissMapMaxElemBytes {
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
