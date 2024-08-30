// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

import (
	"unsafe"
)

// Map constants common to several packages
// runtime/runtime-gdb.py:MapTypePrinter contains its own copy
const (
	// Maximum number of key/elem pairs a bucket can hold.
	SwissMapBucketCountBits = 3 // log2 of number of elements in a bucket.
	SwissMapBucketCount     = 1 << SwissMapBucketCountBits

	// Maximum key or elem size to keep inline (instead of mallocing per element).
	// Must fit in a uint8.
	// Note: fast map functions cannot handle big elems (bigger than MapMaxElemBytes).
	SwissMapMaxKeyBytes  = 128
	SwissMapMaxElemBytes = 128 // Must fit in a uint8.
)

type SwissMapType struct {
	Type
	Key    *Type
	Elem   *Type
	Bucket *Type // internal type representing a hash bucket
	// function for hashing keys (ptr to key, seed) -> hash
	Hasher     func(unsafe.Pointer, uintptr) uintptr
	KeySize    uint8  // size of key slot
	ValueSize  uint8  // size of elem slot
	BucketSize uint16 // size of bucket
	Flags      uint32
}

// Note: flag values must match those used in the TMAP case
// in ../cmd/compile/internal/reflectdata/reflect.go:writeType.
func (mt *SwissMapType) IndirectKey() bool { // store ptr to key instead of key itself
	return mt.Flags&1 != 0
}
func (mt *SwissMapType) IndirectElem() bool { // store ptr to elem instead of elem itself
	return mt.Flags&2 != 0
}
func (mt *SwissMapType) ReflexiveKey() bool { // true if k==k for all keys
	return mt.Flags&4 != 0
}
func (mt *SwissMapType) NeedKeyUpdate() bool { // true if we need to update key on an overwrite
	return mt.Flags&8 != 0
}
func (mt *SwissMapType) HashMightPanic() bool { // true if hash function might panic
	return mt.Flags&16 != 0
}

