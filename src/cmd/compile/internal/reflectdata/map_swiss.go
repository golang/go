// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflectdata

import (
	"internal/abi"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/rttype"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

// SwissMapGroupType makes the map slot group type given the type of the map.
func SwissMapGroupType(t *types.Type) *types.Type {
	if t.MapType().SwissGroup != nil {
		return t.MapType().SwissGroup
	}

	// Builds a type representing a group structure for the given map type.
	// This type is not visible to users, we include it so we can generate
	// a correct GC program for it.
	//
	// Make sure this stays in sync with internal/runtime/maps/group.go.
	//
	// type group struct {
	//     ctrl uint64
	//     slots [abi.SwissMapGroupSlots]struct {
	//         key  keyType
	//         elem elemType
	//     }
	// }
	slotFields := []*types.Field{
		makefield("key", t.Key()),
		makefield("typ", t.Elem()),
	}
	slot := types.NewStruct(slotFields)
	slot.SetNoalg(true)

	slotArr := types.NewArray(slot, abi.SwissMapGroupSlots)
	slotArr.SetNoalg(true)

	fields := []*types.Field{
		makefield("ctrl", types.Types[types.TUINT64]),
		makefield("slots", slotArr),
	}

	group := types.NewStruct(fields)
	group.SetNoalg(true)
	types.CalcSize(group)

	// Check invariants that map code depends on.
	if !types.IsComparable(t.Key()) {
		base.Fatalf("unsupported map key type for %v", t)
	}
	if group.Size() <= 8 {
		// internal/runtime/maps creates pointers to slots, even if
		// both key and elem are size zero. In this case, each slot is
		// size 0, but group should still reserve a word of padding at
		// the end to ensure pointers are valid.
		base.Fatalf("bad group size for %v", t)
	}

	t.MapType().SwissGroup = group
	group.StructType().Map = t
	return group
}

var swissHmapType *types.Type

// SwissMapType returns a type interchangeable with internal/runtime/maps.Map.
// Make sure this stays in sync with internal/runtime/maps/map.go.
func SwissMapType() *types.Type {
	if swissHmapType != nil {
		return swissHmapType
	}

	// build a struct:
	// type table struct {
	//     used uint64
	//     typ  unsafe.Pointer // *abi.SwissMapType
	//     seed uintptr
	//
	//     // From groups.
	//     groups_typ        unsafe.Pointer // *abi.SwissMapType
	//     groups_data       unsafe.Pointer
	//     groups_lengthMask uint64
	//
	//     capacity   uint64
	//     growthLeft uint64
	//
	//     clearSeq uint64
	// }
	// must match internal/runtime/maps/map.go:Map.
	fields := []*types.Field{
		makefield("used", types.Types[types.TUINT64]),
		makefield("typ", types.Types[types.TUNSAFEPTR]),
		makefield("seed", types.Types[types.TUINTPTR]),
		makefield("groups_typ", types.Types[types.TUNSAFEPTR]),
		makefield("groups_data", types.Types[types.TUNSAFEPTR]),
		makefield("groups_lengthMask", types.Types[types.TUINT64]),
		makefield("capacity", types.Types[types.TUINT64]),
		makefield("growthLeft", types.Types[types.TUINT64]),
		makefield("clearSeq", types.Types[types.TUINT64]),
	}

	n := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, ir.Pkgs.InternalMaps.Lookup("table"))
	hmap := types.NewNamed(n)
	n.SetType(hmap)
	n.SetTypecheck(1)

	hmap.SetUnderlying(types.NewStruct(fields))
	types.CalcSize(hmap)

	// The size of Map should be 64 bytes on 64 bit
	// and 48 bytes on 32 bit platforms.
	if size := int64(5*8 + 4*types.PtrSize); hmap.Size() != size {
		base.Fatalf("internal/runtime/maps.Map size not correct: got %d, want %d", hmap.Size(), size)
	}

	swissHmapType = hmap
	return hmap
}

var swissHiterType *types.Type

// SwissMapIterType returns a type interchangeable with runtime.hiter.
// Make sure this stays in sync with runtime/map.go.
func SwissMapIterType() *types.Type {
	if swissHiterType != nil {
		return swissHiterType
	}

	hmap := SwissMapType()

	// build a struct:
	// type Iter struct {
	//    key      unsafe.Pointer // *Key
	//    elem     unsafe.Pointer // *Elem
	//    typ      unsafe.Pointer // *SwissMapType
	//    m        *Map
	//
	//    // From groups.
	//    groups_typ        unsafe.Pointer // *abi.SwissMapType
	//    groups_data       unsafe.Pointer
	//    groups_lengthMask uint64
	//
	//    clearSeq uint64
	//
	//    offset   uint64
	//    groupIdx uint64
	//    slotIdx  uint32
	//
	//    // 4 bytes of padding on 64-bit arches.
	// }
	// must match internal/runtime/maps/table.go:Iter.
	fields := []*types.Field{
		makefield("key", types.Types[types.TUNSAFEPTR]),  // Used in range.go for TMAP.
		makefield("elem", types.Types[types.TUNSAFEPTR]), // Used in range.go for TMAP.
		makefield("typ", types.Types[types.TUNSAFEPTR]),
		makefield("m", types.NewPtr(hmap)),
		makefield("groups_typ", types.Types[types.TUNSAFEPTR]),
		makefield("groups_data", types.Types[types.TUNSAFEPTR]),
		makefield("groups_lengthMask", types.Types[types.TUINT64]),
		makefield("clearSeq", types.Types[types.TUINT64]),
		makefield("offset", types.Types[types.TUINT64]),
		makefield("groupIdx", types.Types[types.TUINT64]),
		makefield("slotIdx", types.Types[types.TUINT32]),
	}

	// build iterator struct hswissing the above fields
	n := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, ir.Pkgs.InternalMaps.Lookup("Iter"))
	hiter := types.NewNamed(n)
	n.SetType(hiter)
	n.SetTypecheck(1)

	hiter.SetUnderlying(types.NewStruct(fields))
	types.CalcSize(hiter)
	want := 6*types.PtrSize + 4*8 + 1*4
	if types.PtrSize == 8 {
		want += 4 // tailing padding
	}
	if hiter.Size() != int64(want) {
		base.Fatalf("hash_iter size not correct %d %d", hiter.Size(), want)
	}

	swissHiterType = hiter
	return hiter
}

func writeSwissMapType(t *types.Type, lsym *obj.LSym, c rttype.Cursor) {
	// internal/abi.SwissMapType
	gtyp := SwissMapGroupType(t)
	s1 := writeType(t.Key())
	s2 := writeType(t.Elem())
	s3 := writeType(gtyp)
	hasher := genhash(t.Key())

	slotTyp := gtyp.Field(1).Type.Elem()
	elemOff := slotTyp.Field(1).Offset

	c.Field("Key").WritePtr(s1)
	c.Field("Elem").WritePtr(s2)
	c.Field("Group").WritePtr(s3)
	c.Field("Hasher").WritePtr(hasher)
	c.Field("SlotSize").WriteUintptr(uint64(slotTyp.Size()))
	c.Field("ElemOff").WriteUintptr(uint64(elemOff))
	var flags uint32
	if needkeyupdate(t.Key()) {
		flags |= abi.SwissMapNeedKeyUpdate
	}
	if hashMightPanic(t.Key()) {
		flags |= abi.SwissMapHashMightPanic
	}
	c.Field("Flags").WriteUint32(flags)

	if u := t.Underlying(); u != t {
		// If t is a named map type, also keep the underlying map
		// type live in the binary. This is important to make sure that
		// a named map and that same map cast to its underlying type via
		// reflection, use the same hash function. See issue 37716.
		r := obj.Addrel(lsym)
		r.Sym = writeType(u)
		r.Type = objabi.R_KEEP
	}
}
