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

// OldMapBucketType makes the map bucket type given the type of the map.
func OldMapBucketType(t *types.Type) *types.Type {
	// Builds a type representing a Bucket structure for
	// the given map type. This type is not visible to users -
	// we include only enough information to generate a correct GC
	// program for it.
	// Make sure this stays in sync with runtime/map.go.
	//
	//	A "bucket" is a "struct" {
	//	      tophash [abi.OldMapBucketCount]uint8
	//	      keys [abi.OldMapBucketCount]keyType
	//	      elems [abi.OldMapBucketCount]elemType
	//	      overflow *bucket
	//	    }
	if t.MapType().OldBucket != nil {
		return t.MapType().OldBucket
	}

	keytype := t.Key()
	elemtype := t.Elem()
	types.CalcSize(keytype)
	types.CalcSize(elemtype)
	if keytype.Size() > abi.OldMapMaxKeyBytes {
		keytype = types.NewPtr(keytype)
	}
	if elemtype.Size() > abi.OldMapMaxElemBytes {
		elemtype = types.NewPtr(elemtype)
	}

	field := make([]*types.Field, 0, 5)

	// The first field is: uint8 topbits[BUCKETSIZE].
	arr := types.NewArray(types.Types[types.TUINT8], abi.OldMapBucketCount)
	field = append(field, makefield("topbits", arr))

	arr = types.NewArray(keytype, abi.OldMapBucketCount)
	arr.SetNoalg(true)
	keys := makefield("keys", arr)
	field = append(field, keys)

	arr = types.NewArray(elemtype, abi.OldMapBucketCount)
	arr.SetNoalg(true)
	elems := makefield("elems", arr)
	field = append(field, elems)

	// If keys and elems have no pointers, the map implementation
	// can keep a list of overflow pointers on the side so that
	// buckets can be marked as having no pointers.
	// Arrange for the bucket to have no pointers by changing
	// the type of the overflow field to uintptr in this case.
	// See comment on hmap.overflow in runtime/map.go.
	otyp := types.Types[types.TUNSAFEPTR]
	if !elemtype.HasPointers() && !keytype.HasPointers() {
		otyp = types.Types[types.TUINTPTR]
	}
	overflow := makefield("overflow", otyp)
	field = append(field, overflow)

	// link up fields
	bucket := types.NewStruct(field[:])
	bucket.SetNoalg(true)
	types.CalcSize(bucket)

	// Check invariants that map code depends on.
	if !types.IsComparable(t.Key()) {
		base.Fatalf("unsupported map key type for %v", t)
	}
	if abi.OldMapBucketCount < 8 {
		base.Fatalf("bucket size %d too small for proper alignment %d", abi.OldMapBucketCount, 8)
	}
	if uint8(keytype.Alignment()) > abi.OldMapBucketCount {
		base.Fatalf("key align too big for %v", t)
	}
	if uint8(elemtype.Alignment()) > abi.OldMapBucketCount {
		base.Fatalf("elem align %d too big for %v, BUCKETSIZE=%d", elemtype.Alignment(), t, abi.OldMapBucketCount)
	}
	if keytype.Size() > abi.OldMapMaxKeyBytes {
		base.Fatalf("key size too large for %v", t)
	}
	if elemtype.Size() > abi.OldMapMaxElemBytes {
		base.Fatalf("elem size too large for %v", t)
	}
	if t.Key().Size() > abi.OldMapMaxKeyBytes && !keytype.IsPtr() {
		base.Fatalf("key indirect incorrect for %v", t)
	}
	if t.Elem().Size() > abi.OldMapMaxElemBytes && !elemtype.IsPtr() {
		base.Fatalf("elem indirect incorrect for %v", t)
	}
	if keytype.Size()%keytype.Alignment() != 0 {
		base.Fatalf("key size not a multiple of key align for %v", t)
	}
	if elemtype.Size()%elemtype.Alignment() != 0 {
		base.Fatalf("elem size not a multiple of elem align for %v", t)
	}
	if uint8(bucket.Alignment())%uint8(keytype.Alignment()) != 0 {
		base.Fatalf("bucket align not multiple of key align %v", t)
	}
	if uint8(bucket.Alignment())%uint8(elemtype.Alignment()) != 0 {
		base.Fatalf("bucket align not multiple of elem align %v", t)
	}
	if keys.Offset%keytype.Alignment() != 0 {
		base.Fatalf("bad alignment of keys in bmap for %v", t)
	}
	if elems.Offset%elemtype.Alignment() != 0 {
		base.Fatalf("bad alignment of elems in bmap for %v", t)
	}

	// Double-check that overflow field is final memory in struct,
	// with no padding at end.
	if overflow.Offset != bucket.Size()-int64(types.PtrSize) {
		base.Fatalf("bad offset of overflow in bmap for %v, overflow.Offset=%d, bucket.Size()-int64(types.PtrSize)=%d",
			t, overflow.Offset, bucket.Size()-int64(types.PtrSize))
	}

	t.MapType().OldBucket = bucket

	bucket.StructType().Map = t
	return bucket
}

var oldHmapType *types.Type

// OldMapType returns a type interchangeable with runtime.hmap.
// Make sure this stays in sync with runtime/map.go.
func OldMapType() *types.Type {
	if oldHmapType != nil {
		return oldHmapType
	}

	// build a struct:
	// type hmap struct {
	//    count      int
	//    flags      uint8
	//    B          uint8
	//    noverflow  uint16
	//    hash0      uint32
	//    buckets    unsafe.Pointer
	//    oldbuckets unsafe.Pointer
	//    nevacuate  uintptr
	//    clearSeq   uint64
	//    extra      unsafe.Pointer // *mapextra
	// }
	// must match runtime/map.go:hmap.
	fields := []*types.Field{
		makefield("count", types.Types[types.TINT]),
		makefield("flags", types.Types[types.TUINT8]),
		makefield("B", types.Types[types.TUINT8]),
		makefield("noverflow", types.Types[types.TUINT16]),
		makefield("hash0", types.Types[types.TUINT32]),      // Used in walk.go for OMAKEMAP.
		makefield("buckets", types.Types[types.TUNSAFEPTR]), // Used in walk.go for OMAKEMAP.
		makefield("oldbuckets", types.Types[types.TUNSAFEPTR]),
		makefield("nevacuate", types.Types[types.TUINTPTR]),
		makefield("clearSeq", types.Types[types.TUINT64]),
		makefield("extra", types.Types[types.TUNSAFEPTR]),
	}

	n := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, ir.Pkgs.Runtime.Lookup("hmap"))
	hmap := types.NewNamed(n)
	n.SetType(hmap)
	n.SetTypecheck(1)

	hmap.SetUnderlying(types.NewStruct(fields))
	types.CalcSize(hmap)

	// The size of hmap should be 56 bytes on 64 bit
	// and 36 bytes on 32 bit platforms.
	if size := int64(2*8 + 5*types.PtrSize); hmap.Size() != size {
		base.Fatalf("hmap size not correct: got %d, want %d", hmap.Size(), size)
	}

	oldHmapType = hmap
	return hmap
}

var oldHiterType *types.Type

// OldMapIterType returns a type interchangeable with runtime.hiter.
// Make sure this stays in sync with runtime/map.go.
func OldMapIterType() *types.Type {
	if oldHiterType != nil {
		return oldHiterType
	}

	hmap := OldMapType()

	// build a struct:
	// type hiter struct {
	//    key         unsafe.Pointer // *Key
	//    elem        unsafe.Pointer // *Elem
	//    t           unsafe.Pointer // *OldMapType
	//    h           *hmap
	//    buckets     unsafe.Pointer
	//    bptr        unsafe.Pointer // *bmap
	//    overflow    unsafe.Pointer // *[]*bmap
	//    oldoverflow unsafe.Pointer // *[]*bmap
	//    startBucket uintptr
	//    offset      uint8
	//    wrapped     bool
	//    B           uint8
	//    i           uint8
	//    bucket      uintptr
	//    checkBucket uintptr
	//    clearSeq    uint64
	// }
	// must match runtime/map.go:hiter.
	fields := []*types.Field{
		makefield("elem", types.Types[types.TUNSAFEPTR]), // Used in range.go for TMAP.
		makefield("key", types.Types[types.TUNSAFEPTR]),  // Used in range.go for TMAP.
		makefield("t", types.Types[types.TUNSAFEPTR]),
		makefield("h", types.NewPtr(hmap)),
		makefield("buckets", types.Types[types.TUNSAFEPTR]),
		makefield("bptr", types.Types[types.TUNSAFEPTR]),
		makefield("overflow", types.Types[types.TUNSAFEPTR]),
		makefield("oldoverflow", types.Types[types.TUNSAFEPTR]),
		makefield("startBucket", types.Types[types.TUINTPTR]),
		makefield("offset", types.Types[types.TUINT8]),
		makefield("wrapped", types.Types[types.TBOOL]),
		makefield("B", types.Types[types.TUINT8]),
		makefield("i", types.Types[types.TUINT8]),
		makefield("bucket", types.Types[types.TUINTPTR]),
		makefield("checkBucket", types.Types[types.TUINTPTR]),
		makefield("clearSeq", types.Types[types.TUINT64]),
	}

	// build iterator struct holding the above fields
	n := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, ir.Pkgs.Runtime.Lookup("hiter"))
	hiter := types.NewNamed(n)
	n.SetType(hiter)
	n.SetTypecheck(1)

	hiter.SetUnderlying(types.NewStruct(fields))
	types.CalcSize(hiter)
	if hiter.Size() != int64(8+12*types.PtrSize) {
		base.Fatalf("hash_iter size not correct %d %d", hiter.Size(), 8+12*types.PtrSize)
	}

	oldHiterType = hiter
	return hiter
}

func writeOldMapType(t *types.Type, lsym *obj.LSym, c rttype.Cursor) {
	// internal/abi.OldMapType
	s1 := writeType(t.Key())
	s2 := writeType(t.Elem())
	s3 := writeType(OldMapBucketType(t))
	hasher := genhash(t.Key())

	c.Field("Elem").WritePtr(s2)
	c.Field("Key").WritePtr(s1)

	c.Field("Bucket").WritePtr(s3)
	c.Field("Hasher").WritePtr(hasher)
	var flags uint32
	// Note: flags must match maptype accessors in ../../../../runtime/type.go
	// and maptype builder in ../../../../reflect/type.go:MapOf.
	if t.Key().Size() > abi.OldMapMaxKeyBytes {
		c.Field("KeySize").WriteUint8(uint8(types.PtrSize))
		flags |= 1 // indirect key
	} else {
		c.Field("KeySize").WriteUint8(uint8(t.Key().Size()))
	}

	if t.Elem().Size() > abi.OldMapMaxElemBytes {
		c.Field("ValueSize").WriteUint8(uint8(types.PtrSize))
		flags |= 2 // indirect value
	} else {
		c.Field("ValueSize").WriteUint8(uint8(t.Elem().Size()))
	}
	c.Field("BucketSize").WriteUint16(uint16(OldMapBucketType(t).Size()))
	if types.IsReflexive(t.Key()) {
		flags |= 4 // reflexive key
	}
	if needkeyupdate(t.Key()) {
		flags |= 8 // need key update
	}
	if hashMightPanic(t.Key()) {
		flags |= 16 // hash might panic
	}
	c.Field("Flags").WriteUint32(flags)

	if u := t.Underlying(); u != t {
		// If t is a named map type, also keep the underlying map
		// type live in the binary. This is important to make sure that
		// a named map and that same map cast to its underlying type via
		// reflection, use the same hash function. See issue 37716.
		lsym.AddRel(base.Ctxt, obj.Reloc{Type: objabi.R_KEEP, Sym: writeType(u)})
	}
}
