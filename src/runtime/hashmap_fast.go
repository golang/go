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
		m := bucketMask(h.B)
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
	for ; b != nil; b = b.overflow(t) {
		for i, k := uintptr(0), b.keys(); i < bucketCnt; i, k = i+1, add(k, 4) {
			if *(*uint32)(k) == key && b.tophash[i] != empty {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*4+i*uintptr(t.valuesize))
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0])
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
		m := bucketMask(h.B)
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
	for ; b != nil; b = b.overflow(t) {
		for i, k := uintptr(0), b.keys(); i < bucketCnt; i, k = i+1, add(k, 4) {
			if *(*uint32)(k) == key && b.tophash[i] != empty {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*4+i*uintptr(t.valuesize)), true
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0]), false
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
		m := bucketMask(h.B)
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
	for ; b != nil; b = b.overflow(t) {
		for i, k := uintptr(0), b.keys(); i < bucketCnt; i, k = i+1, add(k, 8) {
			if *(*uint64)(k) == key && b.tophash[i] != empty {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*8+i*uintptr(t.valuesize))
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0])
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
		m := bucketMask(h.B)
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
	for ; b != nil; b = b.overflow(t) {
		for i, k := uintptr(0), b.keys(); i < bucketCnt; i, k = i+1, add(k, 8) {
			if *(*uint64)(k) == key && b.tophash[i] != empty {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*8+i*uintptr(t.valuesize)), true
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0]), false
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
			for i, kptr := uintptr(0), b.keys(); i < bucketCnt; i, kptr = i+1, add(kptr, 2*sys.PtrSize) {
				k := (*stringStruct)(kptr)
				if k.len != key.len || b.tophash[i] == empty {
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
		for i, kptr := uintptr(0), b.keys(); i < bucketCnt; i, kptr = i+1, add(kptr, 2*sys.PtrSize) {
			k := (*stringStruct)(kptr)
			if k.len != key.len || b.tophash[i] == empty {
				continue
			}
			if k.str == key.str {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize))
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
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+keymaybe*uintptr(t.valuesize))
			}
		}
		return unsafe.Pointer(&zeroVal[0])
	}
dohash:
	hash := t.key.alg.hash(noescape(unsafe.Pointer(&ky)), uintptr(h.hash0))
	m := bucketMask(h.B)
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
	top := tophash(hash)
	for ; b != nil; b = b.overflow(t) {
		for i, kptr := uintptr(0), b.keys(); i < bucketCnt; i, kptr = i+1, add(kptr, 2*sys.PtrSize) {
			k := (*stringStruct)(kptr)
			if k.len != key.len || b.tophash[i] != top {
				continue
			}
			if k.str == key.str || memequal(k.str, key.str, uintptr(key.len)) {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize))
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0])
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
			for i, kptr := uintptr(0), b.keys(); i < bucketCnt; i, kptr = i+1, add(kptr, 2*sys.PtrSize) {
				k := (*stringStruct)(kptr)
				if k.len != key.len || b.tophash[i] == empty {
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
		for i, kptr := uintptr(0), b.keys(); i < bucketCnt; i, kptr = i+1, add(kptr, 2*sys.PtrSize) {
			k := (*stringStruct)(kptr)
			if k.len != key.len || b.tophash[i] == empty {
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
	m := bucketMask(h.B)
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
	top := tophash(hash)
	for ; b != nil; b = b.overflow(t) {
		for i, kptr := uintptr(0), b.keys(); i < bucketCnt; i, kptr = i+1, add(kptr, 2*sys.PtrSize) {
			k := (*stringStruct)(kptr)
			if k.len != key.len || b.tophash[i] != top {
				continue
			}
			if k.str == key.str || memequal(k.str, key.str, uintptr(key.len)) {
				return add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize)), true
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0]), false
}

func mapassign_fast32(t *maptype, h *hmap, key uint32) unsafe.Pointer {
	if h == nil {
		panic(plainError("assignment to entry in nil map"))
	}
	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racewritepc(unsafe.Pointer(h), callerpc, funcPC(mapassign_fast32))
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}
	hash := t.key.alg.hash(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))

	// Set hashWriting after calling alg.hash for consistency with mapassign.
	h.flags |= hashWriting

	if h.buckets == nil {
		h.buckets = newobject(t.bucket) // newarray(t.bucket, 1)
	}

again:
	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork(t, h, bucket)
	}
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + bucket*uintptr(t.bucketsize)))
	top := tophash(hash)

	var inserti *uint8
	var insertk unsafe.Pointer
	var val unsafe.Pointer
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == empty && inserti == nil {
					inserti = &b.tophash[i]
					insertk = add(unsafe.Pointer(b), dataOffset+i*4)
					val = add(unsafe.Pointer(b), dataOffset+bucketCnt*4+i*uintptr(t.valuesize))
				}
				continue
			}
			k := *((*uint32)(add(unsafe.Pointer(b), dataOffset+i*4)))
			if k != key {
				continue
			}
			val = add(unsafe.Pointer(b), dataOffset+bucketCnt*4+i*uintptr(t.valuesize))
			goto done
		}
		ovf := b.overflow(t)
		if ovf == nil {
			break
		}
		b = ovf
	}

	// Did not find mapping for key. Allocate new cell & add entry.

	// If we hit the max load factor or we have too many overflow buckets,
	// and we're not already in the middle of growing, start growing.
	if !h.growing() && (overLoadFactor(h.count, h.B) || tooManyOverflowBuckets(h.noverflow, h.B)) {
		hashGrow(t, h)
		goto again // Growing the table invalidates everything, so try again
	}

	if inserti == nil {
		// all current buckets are full, allocate a new one.
		newb := h.newoverflow(t, b)
		inserti = &newb.tophash[0]
		insertk = add(unsafe.Pointer(newb), dataOffset)
		val = add(insertk, bucketCnt*4)
	}

	// store new key at insert position
	if sys.PtrSize == 4 && t.key.kind&kindNoPointers == 0 && writeBarrier.enabled {
		writebarrierptr((*uintptr)(insertk), uintptr(key))
	} else {
		*(*uint32)(insertk) = key
	}
	*inserti = top
	h.count++

done:
	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
	return val
}

func mapassign_fast64(t *maptype, h *hmap, key uint64) unsafe.Pointer {
	if h == nil {
		panic(plainError("assignment to entry in nil map"))
	}
	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racewritepc(unsafe.Pointer(h), callerpc, funcPC(mapassign_fast64))
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}
	hash := t.key.alg.hash(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))

	// Set hashWriting after calling alg.hash for consistency with mapassign.
	h.flags |= hashWriting

	if h.buckets == nil {
		h.buckets = newobject(t.bucket) // newarray(t.bucket, 1)
	}

again:
	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork(t, h, bucket)
	}
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + bucket*uintptr(t.bucketsize)))
	top := tophash(hash)

	var inserti *uint8
	var insertk unsafe.Pointer
	var val unsafe.Pointer
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == empty && inserti == nil {
					inserti = &b.tophash[i]
					insertk = add(unsafe.Pointer(b), dataOffset+i*8)
					val = add(unsafe.Pointer(b), dataOffset+bucketCnt*8+i*uintptr(t.valuesize))
				}
				continue
			}
			k := *((*uint64)(add(unsafe.Pointer(b), dataOffset+i*8)))
			if k != key {
				continue
			}
			val = add(unsafe.Pointer(b), dataOffset+bucketCnt*8+i*uintptr(t.valuesize))
			goto done
		}
		ovf := b.overflow(t)
		if ovf == nil {
			break
		}
		b = ovf
	}

	// Did not find mapping for key. Allocate new cell & add entry.

	// If we hit the max load factor or we have too many overflow buckets,
	// and we're not already in the middle of growing, start growing.
	if !h.growing() && (overLoadFactor(h.count, h.B) || tooManyOverflowBuckets(h.noverflow, h.B)) {
		hashGrow(t, h)
		goto again // Growing the table invalidates everything, so try again
	}

	if inserti == nil {
		// all current buckets are full, allocate a new one.
		newb := h.newoverflow(t, b)
		inserti = &newb.tophash[0]
		insertk = add(unsafe.Pointer(newb), dataOffset)
		val = add(insertk, bucketCnt*8)
	}

	// store new key at insert position
	if t.key.kind&kindNoPointers == 0 && writeBarrier.enabled {
		if sys.PtrSize == 8 {
			writebarrierptr((*uintptr)(insertk), uintptr(key))
		} else {
			// There are three ways to squeeze at least one 32 bit pointer into 64 bits.
			// Give up and call typedmemmove.
			typedmemmove(t.key, insertk, unsafe.Pointer(&key))
		}
	} else {
		*(*uint64)(insertk) = key
	}

	*inserti = top
	h.count++

done:
	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
	return val
}

func mapassign_faststr(t *maptype, h *hmap, ky string) unsafe.Pointer {
	if h == nil {
		panic(plainError("assignment to entry in nil map"))
	}
	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racewritepc(unsafe.Pointer(h), callerpc, funcPC(mapassign_faststr))
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}
	key := stringStructOf(&ky)
	hash := t.key.alg.hash(noescape(unsafe.Pointer(&ky)), uintptr(h.hash0))

	// Set hashWriting after calling alg.hash for consistency with mapassign.
	h.flags |= hashWriting

	if h.buckets == nil {
		h.buckets = newobject(t.bucket) // newarray(t.bucket, 1)
	}

again:
	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork(t, h, bucket)
	}
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + bucket*uintptr(t.bucketsize)))
	top := tophash(hash)

	var inserti *uint8
	var insertk unsafe.Pointer
	var val unsafe.Pointer
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == empty && inserti == nil {
					inserti = &b.tophash[i]
					insertk = add(unsafe.Pointer(b), dataOffset+i*2*sys.PtrSize)
					val = add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize))
				}
				continue
			}
			k := (*stringStruct)(add(unsafe.Pointer(b), dataOffset+i*2*sys.PtrSize))
			if k.len != key.len {
				continue
			}
			if k.str != key.str && !memequal(k.str, key.str, uintptr(key.len)) {
				continue
			}
			// already have a mapping for key. Update it.
			val = add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize))
			goto done
		}
		ovf := b.overflow(t)
		if ovf == nil {
			break
		}
		b = ovf
	}

	// Did not find mapping for key. Allocate new cell & add entry.

	// If we hit the max load factor or we have too many overflow buckets,
	// and we're not already in the middle of growing, start growing.
	if !h.growing() && (overLoadFactor(h.count, h.B) || tooManyOverflowBuckets(h.noverflow, h.B)) {
		hashGrow(t, h)
		goto again // Growing the table invalidates everything, so try again
	}

	if inserti == nil {
		// all current buckets are full, allocate a new one.
		newb := h.newoverflow(t, b)
		inserti = &newb.tophash[0]
		insertk = add(unsafe.Pointer(newb), dataOffset)
		val = add(insertk, bucketCnt*2*sys.PtrSize)
	}

	// store new key at insert position
	*((*stringStruct)(insertk)) = *key
	*inserti = top
	h.count++

done:
	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
	return val
}

func mapdelete_fast32(t *maptype, h *hmap, key uint32) {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racewritepc(unsafe.Pointer(h), callerpc, funcPC(mapdelete_fast32))
	}
	if h == nil || h.count == 0 {
		return
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}

	hash := t.key.alg.hash(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))

	// Set hashWriting after calling alg.hash for consistency with mapdelete
	h.flags |= hashWriting

	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork(t, h, bucket)
	}
	b := (*bmap)(add(h.buckets, bucket*uintptr(t.bucketsize)))
search:
	for ; b != nil; b = b.overflow(t) {
		for i, k := uintptr(0), b.keys(); i < bucketCnt; i, k = i+1, add(k, 4) {
			if key != *(*uint32)(k) || b.tophash[i] == empty {
				continue
			}
			// Only clear key if there are pointers in it.
			if t.key.kind&kindNoPointers == 0 {
				memclrHasPointers(k, t.key.size)
			}
			// Only clear value if there are pointers in it.
			if t.elem.kind&kindNoPointers == 0 {
				v := add(unsafe.Pointer(b), dataOffset+bucketCnt*4+i*uintptr(t.valuesize))
				memclrHasPointers(v, t.elem.size)
			}
			b.tophash[i] = empty
			h.count--
			break search
		}
	}

	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
}

func mapdelete_fast64(t *maptype, h *hmap, key uint64) {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racewritepc(unsafe.Pointer(h), callerpc, funcPC(mapdelete_fast64))
	}
	if h == nil || h.count == 0 {
		return
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}

	hash := t.key.alg.hash(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))

	// Set hashWriting after calling alg.hash for consistency with mapdelete
	h.flags |= hashWriting

	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork(t, h, bucket)
	}
	b := (*bmap)(add(h.buckets, bucket*uintptr(t.bucketsize)))
search:
	for ; b != nil; b = b.overflow(t) {
		for i, k := uintptr(0), b.keys(); i < bucketCnt; i, k = i+1, add(k, 8) {
			if key != *(*uint64)(k) || b.tophash[i] == empty {
				continue
			}
			// Only clear key if there are pointers in it.
			if t.key.kind&kindNoPointers == 0 {
				memclrHasPointers(k, t.key.size)
			}
			// Only clear value if there are pointers in it.
			if t.elem.kind&kindNoPointers == 0 {
				v := add(unsafe.Pointer(b), dataOffset+bucketCnt*8+i*uintptr(t.valuesize))
				memclrHasPointers(v, t.elem.size)
			}
			b.tophash[i] = empty
			h.count--
			break search
		}
	}

	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
}

func mapdelete_faststr(t *maptype, h *hmap, ky string) {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racewritepc(unsafe.Pointer(h), callerpc, funcPC(mapdelete_faststr))
	}
	if h == nil || h.count == 0 {
		return
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}

	key := stringStructOf(&ky)
	hash := t.key.alg.hash(noescape(unsafe.Pointer(&ky)), uintptr(h.hash0))

	// Set hashWriting after calling alg.hash for consistency with mapdelete
	h.flags |= hashWriting

	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork(t, h, bucket)
	}
	b := (*bmap)(add(h.buckets, bucket*uintptr(t.bucketsize)))
	top := tophash(hash)
search:
	for ; b != nil; b = b.overflow(t) {
		for i, kptr := uintptr(0), b.keys(); i < bucketCnt; i, kptr = i+1, add(kptr, 2*sys.PtrSize) {
			k := (*stringStruct)(kptr)
			if k.len != key.len || b.tophash[i] != top {
				continue
			}
			if k.str != key.str && !memequal(k.str, key.str, uintptr(key.len)) {
				continue
			}
			*(*string)(kptr) = ""
			// Only clear value if there are pointers in it.
			if t.elem.kind&kindNoPointers == 0 {
				v := add(unsafe.Pointer(b), dataOffset+bucketCnt*2*sys.PtrSize+i*uintptr(t.valuesize))
				memclrHasPointers(v, t.elem.size)
			}
			b.tophash[i] = empty
			h.count--
			break search
		}
	}

	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
}
