// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

func mapaccess1_fast64(t *maptype, h *hmap, key uint64) unsafe.Pointer {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, abi.FuncPCABIInternal(mapaccess1_fast64))
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0])
	}
	if h.flags&hashWriting != 0 {
		fatal("concurrent map read and map write")
	}
	var b *bmap
	if h.B == 0 {
		// One-bucket table. No need to hash.
		b = (*bmap)(h.buckets)
	} else {
		hash := t.Hasher(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))
		m := bucketMask(h.B)
		b = (*bmap)(add(h.buckets, (hash&m)*uintptr(t.BucketSize)))
		if c := h.oldbuckets; c != nil {
			if !h.sameSizeGrow() {
				// There used to be half as many buckets; mask down one more power of two.
				m >>= 1
			}
			oldb := (*bmap)(add(c, (hash&m)*uintptr(t.BucketSize)))
			if !evacuated(oldb) {
				b = oldb
			}
		}
	}
	for ; b != nil; b = b.overflow(t) {
		for i, k := uintptr(0), b.keys(); i < abi.MapBucketCount; i, k = i+1, add(k, 8) {
			if *(*uint64)(k) == key && !isEmpty(b.tophash[i]) {
				return add(unsafe.Pointer(b), dataOffset+abi.MapBucketCount*8+i*uintptr(t.ValueSize))
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0])
}

func mapaccess2_fast64(t *maptype, h *hmap, key uint64) (unsafe.Pointer, bool) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, abi.FuncPCABIInternal(mapaccess2_fast64))
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0]), false
	}
	if h.flags&hashWriting != 0 {
		fatal("concurrent map read and map write")
	}
	var b *bmap
	if h.B == 0 {
		// One-bucket table. No need to hash.
		b = (*bmap)(h.buckets)
	} else {
		hash := t.Hasher(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))
		m := bucketMask(h.B)
		b = (*bmap)(add(h.buckets, (hash&m)*uintptr(t.BucketSize)))
		if c := h.oldbuckets; c != nil {
			if !h.sameSizeGrow() {
				// There used to be half as many buckets; mask down one more power of two.
				m >>= 1
			}
			oldb := (*bmap)(add(c, (hash&m)*uintptr(t.BucketSize)))
			if !evacuated(oldb) {
				b = oldb
			}
		}
	}
	for ; b != nil; b = b.overflow(t) {
		for i, k := uintptr(0), b.keys(); i < abi.MapBucketCount; i, k = i+1, add(k, 8) {
			if *(*uint64)(k) == key && !isEmpty(b.tophash[i]) {
				return add(unsafe.Pointer(b), dataOffset+abi.MapBucketCount*8+i*uintptr(t.ValueSize)), true
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0]), false
}

func mapassign_fast64(t *maptype, h *hmap, key uint64) unsafe.Pointer {
	if h == nil {
		panic(plainError("assignment to entry in nil map"))
	}
	if raceenabled {
		callerpc := getcallerpc()
		racewritepc(unsafe.Pointer(h), callerpc, abi.FuncPCABIInternal(mapassign_fast64))
	}
	if h.flags&hashWriting != 0 {
		fatal("concurrent map writes")
	}
	hash := t.Hasher(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))

	// Set hashWriting after calling t.hasher for consistency with mapassign.
	h.flags ^= hashWriting

	if h.buckets == nil {
		h.buckets = newobject(t.Bucket) // newarray(t.bucket, 1)
	}

again:
	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork_fast64(t, h, bucket)
	}
	b := (*bmap)(add(h.buckets, bucket*uintptr(t.BucketSize)))

	var insertb *bmap
	var inserti uintptr
	var insertk unsafe.Pointer

bucketloop:
	for {
		for i := uintptr(0); i < abi.MapBucketCount; i++ {
			if isEmpty(b.tophash[i]) {
				if insertb == nil {
					insertb = b
					inserti = i
				}
				if b.tophash[i] == emptyRest {
					break bucketloop
				}
				continue
			}
			k := *((*uint64)(add(unsafe.Pointer(b), dataOffset+i*8)))
			if k != key {
				continue
			}
			insertb = b
			inserti = i
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
	if !h.growing() && (overLoadFactor(h.count+1, h.B) || tooManyOverflowBuckets(h.noverflow, h.B)) {
		hashGrow(t, h)
		goto again // Growing the table invalidates everything, so try again
	}

	if insertb == nil {
		// The current bucket and all the overflow buckets connected to it are full, allocate a new one.
		insertb = h.newoverflow(t, b)
		inserti = 0 // not necessary, but avoids needlessly spilling inserti
	}
	insertb.tophash[inserti&(abi.MapBucketCount-1)] = tophash(hash) // mask inserti to avoid bounds checks

	insertk = add(unsafe.Pointer(insertb), dataOffset+inserti*8)
	// store new key at insert position
	*(*uint64)(insertk) = key

	h.count++

done:
	elem := add(unsafe.Pointer(insertb), dataOffset+abi.MapBucketCount*8+inserti*uintptr(t.ValueSize))
	if h.flags&hashWriting == 0 {
		fatal("concurrent map writes")
	}
	h.flags &^= hashWriting
	return elem
}

func mapassign_fast64ptr(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	if h == nil {
		panic(plainError("assignment to entry in nil map"))
	}
	if raceenabled {
		callerpc := getcallerpc()
		racewritepc(unsafe.Pointer(h), callerpc, abi.FuncPCABIInternal(mapassign_fast64))
	}
	if h.flags&hashWriting != 0 {
		fatal("concurrent map writes")
	}
	hash := t.Hasher(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))

	// Set hashWriting after calling t.hasher for consistency with mapassign.
	h.flags ^= hashWriting

	if h.buckets == nil {
		h.buckets = newobject(t.Bucket) // newarray(t.bucket, 1)
	}

again:
	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork_fast64(t, h, bucket)
	}
	b := (*bmap)(add(h.buckets, bucket*uintptr(t.BucketSize)))

	var insertb *bmap
	var inserti uintptr
	var insertk unsafe.Pointer

bucketloop:
	for {
		for i := uintptr(0); i < abi.MapBucketCount; i++ {
			if isEmpty(b.tophash[i]) {
				if insertb == nil {
					insertb = b
					inserti = i
				}
				if b.tophash[i] == emptyRest {
					break bucketloop
				}
				continue
			}
			k := *((*unsafe.Pointer)(add(unsafe.Pointer(b), dataOffset+i*8)))
			if k != key {
				continue
			}
			insertb = b
			inserti = i
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
	if !h.growing() && (overLoadFactor(h.count+1, h.B) || tooManyOverflowBuckets(h.noverflow, h.B)) {
		hashGrow(t, h)
		goto again // Growing the table invalidates everything, so try again
	}

	if insertb == nil {
		// The current bucket and all the overflow buckets connected to it are full, allocate a new one.
		insertb = h.newoverflow(t, b)
		inserti = 0 // not necessary, but avoids needlessly spilling inserti
	}
	insertb.tophash[inserti&(abi.MapBucketCount-1)] = tophash(hash) // mask inserti to avoid bounds checks

	insertk = add(unsafe.Pointer(insertb), dataOffset+inserti*8)
	// store new key at insert position
	*(*unsafe.Pointer)(insertk) = key

	h.count++

done:
	elem := add(unsafe.Pointer(insertb), dataOffset+abi.MapBucketCount*8+inserti*uintptr(t.ValueSize))
	if h.flags&hashWriting == 0 {
		fatal("concurrent map writes")
	}
	h.flags &^= hashWriting
	return elem
}

func mapdelete_fast64(t *maptype, h *hmap, key uint64) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		racewritepc(unsafe.Pointer(h), callerpc, abi.FuncPCABIInternal(mapdelete_fast64))
	}
	if h == nil || h.count == 0 {
		return
	}
	if h.flags&hashWriting != 0 {
		fatal("concurrent map writes")
	}

	hash := t.Hasher(noescape(unsafe.Pointer(&key)), uintptr(h.hash0))

	// Set hashWriting after calling t.hasher for consistency with mapdelete
	h.flags ^= hashWriting

	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork_fast64(t, h, bucket)
	}
	b := (*bmap)(add(h.buckets, bucket*uintptr(t.BucketSize)))
	bOrig := b
search:
	for ; b != nil; b = b.overflow(t) {
		for i, k := uintptr(0), b.keys(); i < abi.MapBucketCount; i, k = i+1, add(k, 8) {
			if key != *(*uint64)(k) || isEmpty(b.tophash[i]) {
				continue
			}
			// Only clear key if there are pointers in it.
			if t.Key.Pointers() {
				if goarch.PtrSize == 8 {
					*(*unsafe.Pointer)(k) = nil
				} else {
					// There are three ways to squeeze at one or more 32 bit pointers into 64 bits.
					// Just call memclrHasPointers instead of trying to handle all cases here.
					memclrHasPointers(k, 8)
				}
			}
      e := add(unsafe.Pointer(b), dataOffset+abi.MapBucketCount*8+i*uintptr(t.ValueSize))
			if t.Elem.Pointers() {
				memclrHasPointers(e, t.Elem.Size_)
			} else {
				memclrNoHeapPointers(e, t.Elem.Size_)
			}
			b.tophash[i] = emptyOne
			// If the bucket now ends in a bunch of emptyOne states,
			// change those to emptyRest states.
			if i == abi.MapBucketCount-1 {
				if b.overflow(t) != nil && b.overflow(t).tophash[0] != emptyRest {
					goto notLast
				}
			} else {
				if b.tophash[i+1] != emptyRest {
					goto notLast
				}
			}
			for {
				b.tophash[i] = emptyRest
				if i == 0 {
					if b == bOrig {
						break // beginning of initial bucket, we're done.
					}
					// Find previous bucket, continue at its last entry.
					c := b
					for b = bOrig; b.overflow(t) != c; b = b.overflow(t) {
					}
					i = abi.MapBucketCount - 1
				} else {
					i--
				}
				if b.tophash[i] != emptyOne {
					break
				}
			}
		notLast:
			h.count--
			// Reset the hash seed to make it more difficult for attackers to
			// repeatedly trigger hash collisions. See issue 25237.
			if h.count == 0 {
				h.hash0 = uint32(rand())
			}
			break search
		}
	}

	if h.flags&hashWriting == 0 {
		fatal("concurrent map writes")
	}
	h.flags &^= hashWriting
}

func growWork_fast64(t *maptype, h *hmap, bucket uintptr) {
	// make sure we evacuate the oldbucket corresponding
	// to the bucket we're about to use
	evacuate_fast64(t, h, bucket&h.oldbucketmask())

	// evacuate one more oldbucket to make progress on growing
	if h.growing() {
		evacuate_fast64(t, h, h.nevacuate)
	}
}

func evacuate_fast64(t *maptype, h *hmap, oldbucket uintptr) {
	b := (*bmap)(add(h.oldbuckets, oldbucket*uintptr(t.BucketSize)))
	newbit := h.noldbuckets()
	if !evacuated(b) {
		// TODO: reuse overflow buckets instead of using new ones, if there
		// is no iterator using the old buckets.  (If !oldIterator.)

		// xy contains the x and y (low and high) evacuation destinations.
		var xy [2]evacDst
		x := &xy[0]
		x.b = (*bmap)(add(h.buckets, oldbucket*uintptr(t.BucketSize)))
		x.k = add(unsafe.Pointer(x.b), dataOffset)
		x.e = add(x.k, abi.MapBucketCount*8)

		if !h.sameSizeGrow() {
			// Only calculate y pointers if we're growing bigger.
			// Otherwise GC can see bad pointers.
			y := &xy[1]
			y.b = (*bmap)(add(h.buckets, (oldbucket+newbit)*uintptr(t.BucketSize)))
			y.k = add(unsafe.Pointer(y.b), dataOffset)
			y.e = add(y.k, abi.MapBucketCount*8)
		}

		for ; b != nil; b = b.overflow(t) {
			k := add(unsafe.Pointer(b), dataOffset)
			e := add(k, abi.MapBucketCount*8)
			for i := 0; i < abi.MapBucketCount; i, k, e = i+1, add(k, 8), add(e, uintptr(t.ValueSize)) {
				top := b.tophash[i]
				if isEmpty(top) {
					b.tophash[i] = evacuatedEmpty
					continue
				}
				if top < minTopHash {
					throw("bad map state")
				}
				var useY uint8
				if !h.sameSizeGrow() {
					// Compute hash to make our evacuation decision (whether we need
					// to send this key/elem to bucket x or bucket y).
					hash := t.Hasher(k, uintptr(h.hash0))
					if hash&newbit != 0 {
						useY = 1
					}
				}

				b.tophash[i] = evacuatedX + useY // evacuatedX + 1 == evacuatedY, enforced in makemap
				dst := &xy[useY]                 // evacuation destination

				if dst.i == abi.MapBucketCount {
					dst.b = h.newoverflow(t, dst.b)
					dst.i = 0
					dst.k = add(unsafe.Pointer(dst.b), dataOffset)
					dst.e = add(dst.k, abi.MapBucketCount*8)
				}
				dst.b.tophash[dst.i&(abi.MapBucketCount-1)] = top // mask dst.i as an optimization, to avoid a bounds check

				// Copy key.
				if t.Key.Pointers() && writeBarrier.enabled {
					if goarch.PtrSize == 8 {
						// Write with a write barrier.
						*(*unsafe.Pointer)(dst.k) = *(*unsafe.Pointer)(k)
					} else {
						// There are three ways to squeeze at least one 32 bit pointer into 64 bits.
						// Give up and call typedmemmove.
						typedmemmove(t.Key, dst.k, k)
					}
				} else {
					*(*uint64)(dst.k) = *(*uint64)(k)
				}

				typedmemmove(t.Elem, dst.e, e)
				dst.i++
				// These updates might push these pointers past the end of the
				// key or elem arrays.  That's ok, as we have the overflow pointer
				// at the end of the bucket to protect against pointing past the
				// end of the bucket.
				dst.k = add(dst.k, 8)
				dst.e = add(dst.e, uintptr(t.ValueSize))
			}
		}
		// Unlink the overflow buckets & clear key/elem to help GC.
		if h.flags&oldIterator == 0 && t.Bucket.Pointers() {
			b := add(h.oldbuckets, oldbucket*uintptr(t.BucketSize))
			// Preserve b.tophash because the evacuation
			// state is maintained there.
			ptr := add(b, dataOffset)
			n := uintptr(t.BucketSize) - dataOffset
			memclrHasPointers(ptr, n)
		}
	}

	if oldbucket == h.nevacuate {
		advanceEvacuationMark(h, t, newbit)
	}
}
