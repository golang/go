// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

func mapaccess1_fast32(t *maptype, h *hmap, key uint32) unsafe.Pointer {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapaccess1_fast32))
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0])
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	var b *bmap
	if h.B == 0 {
		// One-bucket table. No need to hash.
		b = (*bmap)(h.buckets)
	} else {
		hash := t.key.alg.hash(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))
		m := uintptr(1)<<h.B - 1
		b = (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
		if c := h.oldbuckets; c != nil {
			if !h.sameSizeGrow() {
				// There used to be half as many buckets; mask down one more power of two.
				m >>= 1
			}
			oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
			if !evacuated(oldb) {
				b = oldb
			}
		}
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			k := *((*uint32)(add(unsafe.Pointer(b), dataOffset+i*4)))
			if k != key {
				continue
			}
			x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
			if x == empty {
				continue
			}
			return add(unsafe.Pointer(b), dataOffset+bucketCnt*4+i*uintptr(t.valuesize))
		}
		b = b.overflow(t)
		if b == nil {
			return unsafe.Pointer(&zeroVal[0])
		}
	}
}

func mapaccess2_fast32(t *maptype, h *hmap, key uint32) (unsafe.Pointer, bool) {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapaccess2_fast32))
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0]), false
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	var b *bmap
	if h.B == 0 {
		// One-bucket table. No need to hash.
		b = (*bmap)(h.buckets)
	} else {
		hash := t.key.alg.hash(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))
		m := uintptr(1)<<h.B - 1
		b = (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
		if c := h.oldbuckets; c != nil {
			if !h.sameSizeGrow() {
				// There used to be half as many buckets; mask down one more power of two.
				m >>= 1
			}
			oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
			if !evacuated(oldb) {
				b = oldb
			}
		}
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			k := *((*uint32)(add(unsafe.Pointer(b), dataOffset+i*4)))
			if k != key {
				continue
			}
			x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
			if x == empty {
				continue
			}
			return add(unsafe.Pointer(b), dataOffset+bucketCnt*4+i*uintptr(t.valuesize)), true
		}
		b = b.overflow(t)
		if b == nil {
			return unsafe.Pointer(&zeroVal[0]), false
		}
	}
}

func mapaccess1_fast64(t *maptype, h *hmap, key uint64) unsafe.Pointer {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapaccess1_fast64))
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0])
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	var b *bmap
	if h.B == 0 {
		// One-bucket table. No need to hash.
		b = (*bmap)(h.buckets)
	} else {
		hash := t.key.alg.hash(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))
		m := uintptr(1)<<h.B - 1
		b = (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
		if c := h.oldbuckets; c != nil {
			if !h.sameSizeGrow() {
				// There used to be half as many buckets; mask down one more power of two.
				m >>= 1
			}
			oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
			if !evacuated(oldb) {
				b = oldb
			}
		}
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			k := *((*uint64)(add(unsafe.Pointer(b), dataOffset+i*8)))
			if k != key {
				continue
			}
			x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
			if x == empty {
				continue
			}
			return add(unsafe.Pointer(b), dataOffset+bucketCnt*8+i*uintptr(t.valuesize))
		}
		b = b.overflow(t)
		if b == nil {
			return unsafe.Pointer(&zeroVal[0])
		}
	}
}

func mapaccess2_fast64(t *maptype, h *hmap, key uint64) (unsafe.Pointer, bool) {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapaccess2_fast64))
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0]), false
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	var b *bmap
	if h.B == 0 {
		// One-bucket table. No need to hash.
		b = (*bmap)(h.buckets)
	} else {
		hash := t.key.alg.hash(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))
		m := uintptr(1)<<h.B - 1
		b = (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
		if c := h.oldbuckets; c != nil {
			if !h.sameSizeGrow() {
				// There used to be half as many buckets; mask down one more power of two.
				m >>= 1
			}
			oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
			if !evacuated(oldb) {
				b = oldb
			}
		}
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			k := *((*uint64)(add(unsafe.Pointer(b), dataOffset+i*8)))
			if k != key {
				continue
			}
			x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
			if x == empty {
				continue
			}
			return add(unsafe.Pointer(b), dataOffset+bucketCnt*8+i*uintptr(t.valuesize)), true
		}
		b = b.overflow(t)
		if b == nil {
			return unsafe.Pointer(&zeroVal[0]), false
		}
	}
}

func mapaccess1_faststr(t *maptype, h *hmap, ky string) unsafe.Pointer {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapaccess1_faststr))
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0])
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	key := stringStructOf(&ky)
	if h.B == 0 {
		// One-bucket table.
		b := (*bmap)(h.buckets)
		if key.len < 32 {
			// short key, doing lots of comparisons is ok
			for i := uintptr(0); i < bucketCnt; i++ {
				x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
				if x == empty {
					continue
				}
				k := (*stringStruct)(add(unsafe.Pointer(b), dataOffset+i*2*sys.PtrSize))
				if k.len != key.len {
					continue
				}
				if k.str == key.str || memequal(k.str, key.str, uintptr(key.len)) {
					return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize))
				}
			}
			return unsafe.Pointer(&zeroVal[0])
		}
		// long key, try not to do more comparisons than necessary
		keymaybe := uintptr(bucketCnt)
		for i := uintptr(0); i < bucketCnt; i++ {
			x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
			if x == empty {
				continue
			}
			k := (*stringStruct)(add(unsafe.Pointer(b), dataOffset+i*2*sys.PtrSize))
			if k.len != key.len {
				continue
			}
			if k.str == key.str {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize))
			}
			// check first 4 bytes
			// TODO: on amd64/386 at least, make this compile to one 4-byte comparison instead of
			// four 1-byte comparisons.
			if *((*[4]byte)(key.str)) != *((*[4]byte)(k.str)) {
				continue
			}
			// check last 4 bytes
			if *((*[4]byte)(add(key.str, uintptr(key.len)-4))) != *((*[4]byte)(add(k.str, uintptr(key.len)-4))) {
				continue
			}
			if keymaybe != bucketCnt {
				// Two keys are potential matches. Use hash to distinguish them.
				goto dohash
			}
			keymaybe = i
		}
		if keymaybe != bucketCnt {
			k := (*stringStruct)(add(unsafe.Pointer(b), dataOffset+keymaybe*2*sys.PtrSize))
			if memequal(k.str, key.str, uintptr(key.len)) {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+keymaybe*uintptr(t.valuesize))
			}
		}
		return unsafe.Pointer(&zeroVal[0])
	}
dohash:
	hash := t.key.alg.hash(noescape(unsafe.Pointer(&ky)), uintptr(h.hash0))
	m := uintptr(1)<<h.B - 1
	b := (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
	if c := h.oldbuckets; c != nil {
		if !h.sameSizeGrow() {
			// There used to be half as many buckets; mask down one more power of two.
			m >>= 1
		}
		oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := uint8(hash >> (sys.PtrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
			if x != top {
				continue
			}
			k := (*stringStruct)(add(unsafe.Pointer(b), dataOffset+i*2*sys.PtrSize))
			if k.len != key.len {
				continue
			}
			if k.str == key.str || memequal(k.str, key.str, uintptr(key.len)) {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize))
			}
		}
		b = b.overflow(t)
		if b == nil {
			return unsafe.Pointer(&zeroVal[0])
		}
	}
}

func mapaccess2_faststr(t *maptype, h *hmap, ky string) (unsafe.Pointer, bool) {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapaccess2_faststr))
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0]), false
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	key := stringStructOf(&ky)
	if h.B == 0 {
		// One-bucket table.
		b := (*bmap)(h.buckets)
		if key.len < 32 {
			// short key, doing lots of comparisons is ok
			for i := uintptr(0); i < bucketCnt; i++ {
				x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
				if x == empty {
					continue
				}
				k := (*stringStruct)(add(unsafe.Pointer(b), dataOffset+i*2*sys.PtrSize))
				if k.len != key.len {
					continue
				}
				if k.str == key.str || memequal(k.str, key.str, uintptr(key.len)) {
					return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize)), true
				}
			}
			return unsafe.Pointer(&zeroVal[0]), false
		}
		// long key, try not to do more comparisons than necessary
		keymaybe := uintptr(bucketCnt)
		for i := uintptr(0); i < bucketCnt; i++ {
			x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
			if x == empty {
				continue
			}
			k := (*stringStruct)(add(unsafe.Pointer(b), dataOffset+i*2*sys.PtrSize))
			if k.len != key.len {
				continue
			}
			if k.str == key.str {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize)), true
			}
			// check first 4 bytes
			if *((*[4]byte)(key.str)) != *((*[4]byte)(k.str)) {
				continue
			}
			// check last 4 bytes
			if *((*[4]byte)(add(key.str, uintptr(key.len)-4))) != *((*[4]byte)(add(k.str, uintptr(key.len)-4))) {
				continue
			}
			if keymaybe != bucketCnt {
				// Two keys are potential matches. Use hash to distinguish them.
				goto dohash
			}
			keymaybe = i
		}
		if keymaybe != bucketCnt {
			k := (*stringStruct)(add(unsafe.Pointer(b), dataOffset+keymaybe*2*sys.PtrSize))
			if memequal(k.str, key.str, uintptr(key.len)) {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+keymaybe*uintptr(t.valuesize)), true
			}
		}
		return unsafe.Pointer(&zeroVal[0]), false
	}
dohash:
	hash := t.key.alg.hash(noescape(unsafe.Pointer(&ky)), uintptr(h.hash0))
	m := uintptr(1)<<h.B - 1
	b := (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
	if c := h.oldbuckets; c != nil {
		if !h.sameSizeGrow() {
			// There used to be half as many buckets; mask down one more power of two.
			m >>= 1
		}
		oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := uint8(hash >> (sys.PtrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			x := *((*uint8)(add(unsafe.Pointer(b), i))) // b.topbits[i] without the bounds check
			if x != top {
				continue
			}
			k := (*stringStruct)(add(unsafe.Pointer(b), dataOffset+i*2*sys.PtrSize))
			if k.len != key.len {
				continue
			}
			if k.str == key.str || memequal(k.str, key.str, uintptr(key.len)) {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize)), true
			}
		}
		b = b.overflow(t)
		if b == nil {
			return unsafe.Pointer(&zeroVal[0]), false
		}
	}
}
