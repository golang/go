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

// SwissMapBucketType makes the map bucket type given the type of the map.
func SwissMapBucketType(t *types.Type) *types.Type {
	// Builds a type representing a Bucket structure for
	// the given map type. This type is not visible to users -
	// we include only enough information to generate a correct GC
	// program for it.
	// Make sure this stays in sync with runtime/map.go.
	//
	//	A "bucket" is a "struct" {
	//	      tophash [abi.SwissMapBucketCount]uint8
	//	      keys [abi.SwissMapBucketCount]keyType
	//	      elems [abi.SwissMapBucketCount]elemType
	//	      overflow *bucket
	//	    }
	if t.MapType().SwissBucket != nil {
		return t.MapType().SwissBucket
	}

	keytype := t.Key()
	elemtype := t.Elem()
	types.CalcSize(keytype)
	types.CalcSize(elemtype)
	if keytype.Size() > abi.SwissMapMaxKeyBytes {
		keytype = types.NewPtr(keytype)
	}
	if elemtype.Size() > abi.SwissMapMaxElemBytes {
		elemtype = types.NewPtr(elemtype)
	}

	field := make([]*types.Field, 0, 5)

	// The first field is: uint8 topbits[BUCKETSIZE].
	arr := types.NewArray(types.Types[types.TUINT8], abi.SwissMapBucketCount)
	field = append(field, makefield("topbits", arr))

	arr = types.NewArray(keytype, abi.SwissMapBucketCount)
	arr.SetNoalg(true)
	keys := makefield("keys", arr)
	field = append(field, keys)

	arr = types.NewArray(elemtype, abi.SwissMapBucketCount)
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
	if abi.SwissMapBucketCount < 8 {
		base.Fatalf("bucket size %d too small for proper alignment %d", abi.SwissMapBucketCount, 8)
	}
	if uint8(keytype.Alignment()) > abi.SwissMapBucketCount {
		base.Fatalf("key align too big for %v", t)
	}
	if uint8(elemtype.Alignment()) > abi.SwissMapBucketCount {
		base.Fatalf("elem align %d too big for %v, BUCKETSIZE=%d", elemtype.Alignment(), t, abi.SwissMapBucketCount)
	}
	if keytype.Size() > abi.SwissMapMaxKeyBytes {
		base.Fatalf("key size too large for %v", t)
	}
	if elemtype.Size() > abi.SwissMapMaxElemBytes {
		base.Fatalf("elem size too large for %v", t)
	}
	if t.Key().Size() > abi.SwissMapMaxKeyBytes && !keytype.IsPtr() {
		base.Fatalf("key indirect incorrect for %v", t)
	}
	if t.Elem().Size() > abi.SwissMapMaxElemBytes && !elemtype.IsPtr() {
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

	t.MapType().SwissBucket = bucket

	bucket.StructType().Map = t
	return bucket
}

var swissHmapType *types.Type

// SwissMapType returns a type interchangeable with runtime.hmap.
// Make sure this stays in sync with runtime/map.go.
func SwissMapType() *types.Type {
	if swissHmapType != nil {
		return swissHmapType
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
		makefield("extra", types.Types[types.TUNSAFEPTR]),
	}

	n := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, ir.Pkgs.Runtime.Lookup("hmap"))
	hmap := types.NewNamed(n)
	n.SetType(hmap)
	n.SetTypecheck(1)

	hmap.SetUnderlying(types.NewStruct(fields))
	types.CalcSize(hmap)

	// The size of hmap should be 48 bytes on 64 bit
	// and 28 bytes on 32 bit platforms.
	if size := int64(8 + 5*types.PtrSize); hmap.Size() != size {
		base.Fatalf("hmap size not correct: got %d, want %d", hmap.Size(), size)
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
	// type hiter struct {
	//    key         unsafe.Pointer // *Key
	//    elem        unsafe.Pointer // *Elem
	//    t           unsafe.Pointer // *SwissMapType
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
	// }
	// must match runtime/map.go:hiter.
	fields := []*types.Field{
		makefield("key", types.Types[types.TUNSAFEPTR]),  // Used in range.go for TMAP.
		makefield("elem", types.Types[types.TUNSAFEPTR]), // Used in range.go for TMAP.
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
	}

	// build iterator struct hswissing the above fields
	n := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, ir.Pkgs.Runtime.Lookup("hiter"))
	hiter := types.NewNamed(n)
	n.SetType(hiter)
	n.SetTypecheck(1)

	hiter.SetUnderlying(types.NewStruct(fields))
	types.CalcSize(hiter)
	if hiter.Size() != int64(12*types.PtrSize) {
		base.Fatalf("hash_iter size not correct %d %d", hiter.Size(), 12*types.PtrSize)
	}

	swissHiterType = hiter
	return hiter
}

func writeSwissMapType(t *types.Type, lsym *obj.LSym, c rttype.Cursor) {
	// internal/abi.SwissMapType
	s1 := writeType(t.Key())
	s2 := writeType(t.Elem())
	s3 := writeType(SwissMapBucketType(t))
	hasher := genhash(t.Key())

	c.Field("Key").WritePtr(s1)
	c.Field("Elem").WritePtr(s2)
	c.Field("Bucket").WritePtr(s3)
	c.Field("Hasher").WritePtr(hasher)
	var flags uint32
	// Note: flags must match maptype accessors in ../../../../runtime/type.go
	// and maptype builder in ../../../../reflect/type.go:MapOf.
	if t.Key().Size() > abi.SwissMapMaxKeyBytes {
		c.Field("KeySize").WriteUint8(uint8(types.PtrSize))
		flags |= 1 // indirect key
	} else {
		c.Field("KeySize").WriteUint8(uint8(t.Key().Size()))
	}

	if t.Elem().Size() > abi.SwissMapMaxElemBytes {
		c.Field("ValueSize").WriteUint8(uint8(types.PtrSize))
		flags |= 2 // indirect value
	} else {
		c.Field("ValueSize").WriteUint8(uint8(t.Elem().Size()))
	}
	c.Field("BucketSize").WriteUint16(uint16(SwissMapBucketType(t).Size()))
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
		r := obj.Addrel(lsym)
		r.Sym = writeType(u)
		r.Type = objabi.R_KEEP
	}
}
