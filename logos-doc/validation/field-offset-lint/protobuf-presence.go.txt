// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"sync/atomic"
	"unsafe"
)

// presenceSize represents the size of a presence set, which should be the largest index of the set+1
type presenceSize uint32

// presence is the internal representation of the bitmap array in a generated protobuf
type presence struct {
	// This is a pointer to the beginning of an array of uint32
	P unsafe.Pointer
}

func (p presence) toElem(num uint32) (ret *uint32) {
	const (
		bitsPerByte = 8
		siz         = unsafe.Sizeof(*ret)
	)
	// p.P points to an array of uint32, num is the bit in this array that the
	// caller wants to check/manipulate. Calculate the index in the array that
	// contains this specific bit. E.g.: 76 / 32 = 2 (integer division).
	offset := uintptr(num) / (siz * bitsPerByte) * siz
	return (*uint32)(unsafe.Pointer(uintptr(p.P) + offset))
}

// Present checks for the presence of a specific field number in a presence set.
func (p presence) Present(num uint32) bool {
	return Export{}.Present(p.toElem(num), num)
}

// SetPresent adds presence for a specific field number in a presence set.
func (p presence) SetPresent(num uint32, size presenceSize) {
	Export{}.SetPresent(p.toElem(num), num, uint32(size))
}

// SetPresentUnatomic adds presence for a specific field number in a presence set without using
// atomic operations. Only to be called during unmarshaling.
func (p presence) SetPresentUnatomic(num uint32, size presenceSize) {
	Export{}.SetPresentNonAtomic(p.toElem(num), num, uint32(size))
}

// ClearPresent removes presence for a specific field number in a presence set.
func (p presence) ClearPresent(num uint32) {
	Export{}.ClearPresent(p.toElem(num), num)
}

// LoadPresenceCache (together with PresentInCache) allows for a
// cached version of checking for presence without re-reading the word
// for every field. It is optimized for efficiency and assumes no
// simltaneous mutation of the presence set (or at least does not have
// a problem with simultaneous mutation giving inconsistent results).
func (p presence) LoadPresenceCache() (current uint32) {
	if p.P == nil {
		return 0
	}
	return atomic.LoadUint32((*uint32)(p.P))
}

// PresentInCache reads presence from a cached word in the presence
// bitmap. It caches up a new word if the bit is outside the
// word. This is for really fast iteration through bitmaps in cases
// where we either know that the bitmap will not be altered, or we
// don't care about inconsistencies caused by simultaneous writes.
func (p presence) PresentInCache(num uint32, cachedElement *uint32, current *uint32) bool {
	if num/32 != *cachedElement {
		o := uintptr(num/32) * unsafe.Sizeof(uint32(0))
		q := (*uint32)(unsafe.Pointer(uintptr(p.P) + o))
		*current = atomic.LoadUint32(q)
		*cachedElement = num / 32
	}
	return (*current & (1 << (num % 32))) > 0
}

// AnyPresent checks if any field is marked as present in the bitmap.
func (p presence) AnyPresent(size presenceSize) bool {
	n := uintptr((size + 31) / 32)
	for j := uintptr(0); j < n; j++ {
		o := j * unsafe.Sizeof(uint32(0))
		q := (*uint32)(unsafe.Pointer(uintptr(p.P) + o))
		b := atomic.LoadUint32(q)
		if b > 0 {
			return true
		}
	}
	return false
}

// toRaceDetectData finds the preceding RaceDetectHookData in a
// message by using pointer arithmetic. As the type of the presence
// set (bitmap) varies with the number of fields in the protobuf, we
// can not have a struct type containing the array and the
// RaceDetectHookData.  instead the RaceDetectHookData is placed
// immediately before the bitmap array, and we find it by walking
// backwards in the struct.
//
// This method is only called from the race-detect version of the code,
// so RaceDetectHookData is never an empty struct.
func (p presence) toRaceDetectData() *RaceDetectHookData {
	var template struct {
		d RaceDetectHookData
		a [1]uint32
	}
	o := (uintptr(unsafe.Pointer(&template.a)) - uintptr(unsafe.Pointer(&template.d)))
	return (*RaceDetectHookData)(unsafe.Pointer(uintptr(p.P) - o))
}

func atomicLoadShadowPresence(p **[]byte) *[]byte {
	return (*[]byte)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(p))))
}
func atomicStoreShadowPresence(p **[]byte, v *[]byte) {
	atomic.CompareAndSwapPointer((*unsafe.Pointer)(unsafe.Pointer(p)), nil, unsafe.Pointer(v))
}

// findPointerToRaceDetectData finds the preceding RaceDetectHookData
// in a message by using pointer arithmetic. For the methods called
// directy from generated code, we don't have a pointer to the
// beginning of the presence set, but a pointer inside the array. As
// we know the index of the bit we're manipulating (num), we can
// calculate which element of the array ptr is pointing to. With that
// information we find the preceding RaceDetectHookData and can
// manipulate the shadow bitmap.
//
// This method is only called from the race-detect version of the
// code, so RaceDetectHookData is never an empty struct.
func findPointerToRaceDetectData(ptr *uint32, num uint32) *RaceDetectHookData {
	var template struct {
		d RaceDetectHookData
		a [1]uint32
	}
	o := (uintptr(unsafe.Pointer(&template.a)) - uintptr(unsafe.Pointer(&template.d))) + uintptr(num/32)*unsafe.Sizeof(uint32(0))
	return (*RaceDetectHookData)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) - o))
}
