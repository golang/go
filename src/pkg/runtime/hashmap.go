// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// This file contains the implementation of Go's map type.
//
// A map is just a hash table.  The data is arranged
// into an array of buckets.  Each bucket contains up to
// 8 key/value pairs.  The low-order bits of the hash are
// used to select a bucket.  Each bucket contains a few
// high-order bits of each hash to distinguish the entries
// within a single bucket.
//
// If more than 8 keys hash to a bucket, we chain on
// extra buckets.
//
// When the hashtable grows, we allocate a new array
// of buckets twice as big.  Buckets are incrementally
// copied from the old bucket array to the new bucket array.
//
// Map iterators walk through the array of buckets and
// return the keys in walk order (bucket #, then overflow
// chain order, then bucket index).  To maintain iteration
// semantics, we never move keys within their bucket (if
// we did, keys might be returned 0 or 2 times).  When
// growing the table, iterators remain iterating through the
// old table and must check the new table if the bucket
// they are iterating through has been moved ("evacuated")
// to the new table.

// Picking loadFactor: too large and we have lots of overflow
// buckets, too small and we waste a lot of space.  I wrote
// a simple program to check some stats for different loads:
// (64-bit, 8 byte keys and values)
//  loadFactor    %overflow  bytes/entry     hitprobe    missprobe
//        4.00         2.13        20.77         3.00         4.00
//        4.50         4.05        17.30         3.25         4.50
//        5.00         6.85        14.77         3.50         5.00
//        5.50        10.55        12.94         3.75         5.50
//        6.00        15.27        11.67         4.00         6.00
//        6.50        20.90        10.79         4.25         6.50
//        7.00        27.14        10.15         4.50         7.00
//        7.50        34.03         9.73         4.75         7.50
//        8.00        41.10         9.40         5.00         8.00
//
// %overflow   = percentage of buckets which have an overflow bucket
// bytes/entry = overhead bytes used per key/value pair
// hitprobe    = # of entries to check when looking up a present key
// missprobe   = # of entries to check when looking up an absent key
//
// Keep in mind this data is for maximally loaded tables, i.e. just
// before the table grows.  Typical tables will be somewhat less loaded.

import (
	"unsafe"
)

const (
	// Maximum number of key/value pairs a bucket can hold.
	bucketCnt = 8

	// Maximum average load of a bucket that triggers growth.
	loadFactor = 6.5

	// Maximum key or value size to keep inline (instead of mallocing per element).
	// Must fit in a uint8.
	// Fast versions cannot handle big values - the cutoff size for
	// fast versions in ../../cmd/gc/walk.c must be at most this value.
	maxKeySize   = 128
	maxValueSize = 128

	// data offset should be the size of the bmap struct, but needs to be
	// aligned correctly.  For amd64p32 this means 64-bit alignment
	// even though pointers are 32 bit.
	dataOffset = unsafe.Offsetof(struct {
		b bmap
		v int64
	}{}.v)

	// Possible tophash values.  We reserve a few possibilities for special marks.
	// Each bucket (including its overflow buckets, if any) will have either all or none of its
	// entries in the evacuated* states (except during the evacuate() method, which only happens
	// during map writes and thus no one else can observe the map during that time).
	empty          = 0 // cell is empty
	evacuatedEmpty = 1 // cell is empty, bucket is evacuated.
	evacuatedX     = 2 // key/value is valid.  Entry has been evacuated to first half of larger table.
	evacuatedY     = 3 // same as above, but evacuated to second half of larger table.
	minTopHash     = 4 // minimum tophash for a normal filled cell.

	// flags
	indirectKey   = 1 // storing pointers to keys
	indirectValue = 2 // storing pointers to values
	iterator      = 4 // there may be an iterator using buckets
	oldIterator   = 8 // there may be an iterator using oldbuckets

	// sentinel bucket ID for iterator checks
	noCheck = 1<<(8*ptrSize) - 1

	// trigger a garbage collection at every alloc called from this code
	checkgc = false
)

// A header for a Go map.
type hmap struct {
	// Note: the format of the Hmap is encoded in ../../cmd/gc/reflect.c and
	// ../reflect/type.go.  Don't change this structure without also changing that code!
	count      int // # live cells == size of map.  Must be first (used by len() builtin)
	flags      uint32
	hash0      uint32 // hash seed
	B          uint8  // log_2 of # of buckets (can hold up to loadFactor * 2^B items)
	keysize    uint8  // key size in bytes
	valuesize  uint8  // value size in bytes
	bucketsize uint16 // bucket size in bytes

	buckets    unsafe.Pointer // array of 2^B Buckets. may be nil if count==0.
	oldbuckets unsafe.Pointer // previous bucket array of half the size, non-nil only when growing
	nevacuate  uintptr        // progress counter for evacuation (buckets less than this have been evacuated)
}

// A bucket for a Go map.
type bmap struct {
	tophash  [bucketCnt]uint8
	overflow *bmap
	// Followed by bucketCnt keys and then bucketCnt values.
	// NOTE: packing all the keys together and then all the values together makes the
	// code a bit more complicated than alternating key/value/key/value/... but it allows
	// us to eliminate padding which would be needed for, e.g., map[int64]int8.
}

// A hash iteration structure.
// If you modify hiter, also change cmd/gc/reflect.c to indicate
// the layout of this structure.
type hiter struct {
	key         unsafe.Pointer // Must be in first position.  Write nil to indicate iteration end (see cmd/gc/range.c).
	value       unsafe.Pointer // Must be in second position (see cmd/gc/range.c).
	t           *maptype
	h           *hmap
	buckets     unsafe.Pointer // bucket ptr at hash_iter initialization time
	bptr        *bmap          // current bucket
	offset      uint8          // intra-bucket offset to start from during iteration (should be big enough to hold bucketCnt-1)
	done        bool
	B           uint8
	bucket      uintptr
	i           uintptr
	checkBucket uintptr
}

func evacuated(b *bmap) bool {
	h := b.tophash[0]
	return h > empty && h < minTopHash
}

func makemap(t *maptype, hint int64) *hmap {
	if sz := unsafe.Sizeof(hmap{}); sz > 48 || sz != uintptr(t.hmap.size) {
		gothrow("bad hmap size")
	}

	if hint < 0 || int64(int32(hint)) != hint {
		panic("makemap: size out of range")
		// TODO: make hint an int, then none of this nonsense
	}

	if !ismapkey(t.key) {
		gothrow("runtime.makemap: unsupported map key type")
	}

	flags := uint32(0)

	// figure out how big we have to make everything
	keysize := uintptr(t.key.size)
	if keysize > maxKeySize {
		flags |= indirectKey
		keysize = ptrSize
	}
	valuesize := uintptr(t.elem.size)
	if valuesize > maxValueSize {
		flags |= indirectValue
		valuesize = ptrSize
	}
	bucketsize := dataOffset + bucketCnt*(keysize+valuesize)
	if bucketsize != uintptr(t.bucket.size) {
		gothrow("bucketsize wrong")
	}

	// invariants we depend on.  We should probably check these at compile time
	// somewhere, but for now we'll do it here.
	if t.key.align > bucketCnt {
		gothrow("key align too big")
	}
	if t.elem.align > bucketCnt {
		gothrow("value align too big")
	}
	if uintptr(t.key.size)%uintptr(t.key.align) != 0 {
		gothrow("key size not a multiple of key align")
	}
	if uintptr(t.elem.size)%uintptr(t.elem.align) != 0 {
		gothrow("value size not a multiple of value align")
	}
	if bucketCnt < 8 {
		gothrow("bucketsize too small for proper alignment")
	}
	if dataOffset%uintptr(t.key.align) != 0 {
		gothrow("need padding in bucket (key)")
	}
	if dataOffset%uintptr(t.elem.align) != 0 {
		gothrow("need padding in bucket (value)")
	}

	// find size parameter which will hold the requested # of elements
	B := uint8(0)
	for ; hint > bucketCnt && float32(hint) > loadFactor*float32(uintptr(1)<<B); B++ {
	}

	// allocate initial hash table
	// if B == 0, the buckets field is allocated lazily later (in mapassign)
	// If hint is large zeroing this memory could take a while.
	var buckets unsafe.Pointer
	if B != 0 {
		if checkgc {
			memstats.next_gc = memstats.heap_alloc
		}
		buckets = unsafe_NewArray(t.bucket, uintptr(1)<<B)
	}

	// initialize Hmap
	if checkgc {
		memstats.next_gc = memstats.heap_alloc
	}
	h := (*hmap)(unsafe_New(t.hmap))
	h.count = 0
	h.B = B
	h.flags = flags
	h.keysize = uint8(keysize)
	h.valuesize = uint8(valuesize)
	h.bucketsize = uint16(bucketsize)
	h.hash0 = fastrand2()
	h.buckets = buckets
	h.oldbuckets = nil
	h.nevacuate = 0

	return h
}

// mapaccess1 returns a pointer to h[key].  Never returns nil, instead
// it will return a reference to the zero object for the value type if
// the key is not in the map.
// NOTE: The returned pointer may keep the whole map live, so don't
// hold onto it for very long.
func mapaccess1(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	if raceenabled && h != nil {
		callerpc := gogetcallerpc(unsafe.Pointer(&t))
		fn := mapaccess1
		pc := **(**uintptr)(unsafe.Pointer(&fn))
		racereadpc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(t.elem.zero)
	}
	hash := gohash(t.key.alg, key, uintptr(t.key.size), uintptr(h.hash0))
	m := uintptr(1)<<h.B - 1
	b := (*bmap)(add(h.buckets, (hash&m)*uintptr(h.bucketsize)))
	if c := h.oldbuckets; c != nil {
		oldb := (*bmap)(add(c, (hash&(m>>1))*uintptr(h.bucketsize)))
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := uint8(hash >> (ptrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(h.keysize))
			if h.flags&indirectKey != 0 {
				k = *((*unsafe.Pointer)(k))
			}
			if goeq(t.key.alg, key, k, uintptr(t.key.size)) {
				v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(h.keysize)+i*uintptr(h.valuesize))
				if h.flags&indirectValue != 0 {
					v = *((*unsafe.Pointer)(v))
				}
				return v
			}
		}
		b = b.overflow
		if b == nil {
			return unsafe.Pointer(t.elem.zero)
		}
	}
}

func mapaccess2(t *maptype, h *hmap, key unsafe.Pointer) (unsafe.Pointer, bool) {
	if raceenabled && h != nil {
		callerpc := gogetcallerpc(unsafe.Pointer(&t))
		fn := mapaccess2
		pc := **(**uintptr)(unsafe.Pointer(&fn))
		racereadpc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(t.elem.zero), false
	}
	hash := gohash(t.key.alg, key, uintptr(t.key.size), uintptr(h.hash0))
	m := uintptr(1)<<h.B - 1
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + (hash&m)*uintptr(h.bucketsize)))
	if c := h.oldbuckets; c != nil {
		oldb := (*bmap)(unsafe.Pointer(uintptr(c) + (hash&(m>>1))*uintptr(h.bucketsize)))
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := uint8(hash >> (ptrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(h.keysize))
			if h.flags&indirectKey != 0 {
				k = *((*unsafe.Pointer)(k))
			}
			if goeq(t.key.alg, key, k, uintptr(t.key.size)) {
				v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(h.keysize)+i*uintptr(h.valuesize))
				if h.flags&indirectValue != 0 {
					v = *((*unsafe.Pointer)(v))
				}
				return v, true
			}
		}
		b = b.overflow
		if b == nil {
			return unsafe.Pointer(t.elem.zero), false
		}
	}
}

// returns both key and value.  Used by map iterator
func mapaccessK(t *maptype, h *hmap, key unsafe.Pointer) (unsafe.Pointer, unsafe.Pointer) {
	if h == nil || h.count == 0 {
		return nil, nil
	}
	hash := gohash(t.key.alg, key, uintptr(t.key.size), uintptr(h.hash0))
	m := uintptr(1)<<h.B - 1
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + (hash&m)*uintptr(h.bucketsize)))
	if c := h.oldbuckets; c != nil {
		oldb := (*bmap)(unsafe.Pointer(uintptr(c) + (hash&(m>>1))*uintptr(h.bucketsize)))
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := uint8(hash >> (ptrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(h.keysize))
			if h.flags&indirectKey != 0 {
				k = *((*unsafe.Pointer)(k))
			}
			if goeq(t.key.alg, key, k, uintptr(t.key.size)) {
				v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(h.keysize)+i*uintptr(h.valuesize))
				if h.flags&indirectValue != 0 {
					v = *((*unsafe.Pointer)(v))
				}
				return k, v
			}
		}
		b = b.overflow
		if b == nil {
			return nil, nil
		}
	}
}

func mapassign1(t *maptype, h *hmap, key unsafe.Pointer, val unsafe.Pointer) {
	if h == nil {
		panic("assignment to entry in nil map")
	}
	if raceenabled {
		callerpc := gogetcallerpc(unsafe.Pointer(&t))
		fn := mapassign1
		pc := **(**uintptr)(unsafe.Pointer(&fn))
		racewritepc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
		raceReadObjectPC(t.elem, val, callerpc, pc)
	}

	hash := gohash(t.key.alg, key, uintptr(t.key.size), uintptr(h.hash0))

	if h.buckets == nil {
		if checkgc {
			memstats.next_gc = memstats.heap_alloc
		}
		h.buckets = unsafe_NewArray(t.bucket, 1)
	}

again:
	bucket := hash & (uintptr(1)<<h.B - 1)
	if h.oldbuckets != nil {
		growWork(t, h, bucket)
	}
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + bucket*uintptr(h.bucketsize)))
	top := uint8(hash >> (ptrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}

	var inserti *uint8
	var insertk unsafe.Pointer
	var insertv unsafe.Pointer
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == empty && inserti == nil {
					inserti = &b.tophash[i]
					insertk = add(unsafe.Pointer(b), dataOffset+i*uintptr(h.keysize))
					insertv = add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(h.keysize)+i*uintptr(h.valuesize))
				}
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(h.keysize))
			k2 := k
			if h.flags&indirectKey != 0 {
				k2 = *((*unsafe.Pointer)(k2))
			}
			if !goeq(t.key.alg, key, k2, uintptr(t.key.size)) {
				continue
			}
			// already have a mapping for key.  Update it.
			memmove(k2, key, uintptr(t.key.size))
			v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(h.keysize)+i*uintptr(h.valuesize))
			v2 := v
			if h.flags&indirectValue != 0 {
				v2 = *((*unsafe.Pointer)(v2))
			}
			memmove(v2, val, uintptr(t.elem.size))
			return
		}
		if b.overflow == nil {
			break
		}
		b = b.overflow
	}

	// did not find mapping for key.  Allocate new cell & add entry.
	if float32(h.count) >= loadFactor*float32((uintptr(1)<<h.B)) && h.count >= bucketCnt {
		hashGrow(t, h)
		goto again // Growing the table invalidates everything, so try again
	}

	if inserti == nil {
		// all current buckets are full, allocate a new one.
		if checkgc {
			memstats.next_gc = memstats.heap_alloc
		}
		newb := (*bmap)(unsafe_New(t.bucket))
		b.overflow = newb
		inserti = &newb.tophash[0]
		insertk = add(unsafe.Pointer(newb), dataOffset)
		insertv = add(insertk, bucketCnt*uintptr(h.keysize))
	}

	// store new key/value at insert position
	if h.flags&indirectKey != 0 {
		if checkgc {
			memstats.next_gc = memstats.heap_alloc
		}
		kmem := unsafe_New(t.key)
		*(*unsafe.Pointer)(insertk) = kmem
		insertk = kmem
	}
	if h.flags&indirectValue != 0 {
		if checkgc {
			memstats.next_gc = memstats.heap_alloc
		}
		vmem := unsafe_New(t.elem)
		*(*unsafe.Pointer)(insertv) = vmem
		insertv = vmem
	}
	memmove(insertk, key, uintptr(t.key.size))
	memmove(insertv, val, uintptr(t.elem.size))
	*inserti = top
	h.count++
}

func mapdelete(t *maptype, h *hmap, key unsafe.Pointer) {
	if raceenabled && h != nil {
		callerpc := gogetcallerpc(unsafe.Pointer(&t))
		fn := mapdelete
		pc := **(**uintptr)(unsafe.Pointer(&fn))
		racewritepc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if h == nil || h.count == 0 {
		return
	}
	hash := gohash(t.key.alg, key, uintptr(t.key.size), uintptr(h.hash0))
	bucket := hash & (uintptr(1)<<h.B - 1)
	if h.oldbuckets != nil {
		growWork(t, h, bucket)
	}
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + bucket*uintptr(h.bucketsize)))
	top := uint8(hash >> (ptrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(h.keysize))
			k2 := k
			if h.flags&indirectKey != 0 {
				k2 = *((*unsafe.Pointer)(k2))
			}
			if !goeq(t.key.alg, key, k2, uintptr(t.key.size)) {
				continue
			}
			memclr(k, uintptr(h.keysize))
			v := unsafe.Pointer(uintptr(unsafe.Pointer(b)) + dataOffset + bucketCnt*uintptr(h.keysize) + i*uintptr(h.valuesize))
			memclr(v, uintptr(h.valuesize))
			b.tophash[i] = empty
			h.count--
			return
		}
		b = b.overflow
		if b == nil {
			return
		}
	}
}

func mapiterinit(t *maptype, h *hmap, it *hiter) {
	// Clear pointer fields so garbage collector does not complain.
	it.key = nil
	it.value = nil
	it.t = nil
	it.h = nil
	it.buckets = nil
	it.bptr = nil

	if raceenabled && h != nil {
		callerpc := gogetcallerpc(unsafe.Pointer(&t))
		fn := mapiterinit
		pc := **(**uintptr)(unsafe.Pointer(&fn))
		racereadpc(unsafe.Pointer(h), callerpc, pc)
	}

	if h == nil || h.count == 0 {
		it.key = nil
		it.value = nil
		return
	}

	if unsafe.Sizeof(hiter{})/ptrSize != 10 {
		gothrow("hash_iter size incorrect") // see ../../cmd/gc/reflect.c
	}
	it.t = t
	it.h = h

	// grab snapshot of bucket state
	it.B = h.B
	it.buckets = h.buckets

	// iterator state
	it.bucket = 0
	it.offset = uint8(fastrand2() & (bucketCnt - 1))
	it.done = false
	it.bptr = nil

	// Remember we have an iterator.
	// Can run concurrently with another hash_iter_init().
	for {
		old := h.flags
		if old == old|iterator|oldIterator {
			break
		}
		if gocas(&h.flags, old, old|iterator|oldIterator) {
			break
		}
	}

	mapiternext(it)
}

func mapiternext(it *hiter) {
	h := it.h
	if raceenabled {
		callerpc := gogetcallerpc(unsafe.Pointer(&it))
		fn := mapiternext
		pc := **(**uintptr)(unsafe.Pointer(&fn))
		racereadpc(unsafe.Pointer(h), callerpc, pc)
	}
	t := it.t
	bucket := it.bucket
	b := it.bptr
	i := it.i
	checkBucket := it.checkBucket

next:
	if b == nil {
		if it.done {
			// end of iteration
			it.key = nil
			it.value = nil
			return
		}
		if h.oldbuckets != nil && it.B == h.B {
			// Iterator was started in the middle of a grow, and the grow isn't done yet.
			// If the bucket we're looking at hasn't been filled in yet (i.e. the old
			// bucket hasn't been evacuated) then we need to iterate through the old
			// bucket and only return the ones that will be migrated to this bucket.
			oldbucket := bucket & (uintptr(1)<<(it.B-1) - 1)
			b = (*bmap)(add(h.oldbuckets, oldbucket*uintptr(h.bucketsize)))
			if !evacuated(b) {
				checkBucket = bucket
			} else {
				b = (*bmap)(add(it.buckets, bucket*uintptr(h.bucketsize)))
				checkBucket = noCheck
			}
		} else {
			b = (*bmap)(add(it.buckets, bucket*uintptr(h.bucketsize)))
			checkBucket = noCheck
		}
		bucket++
		if bucket == uintptr(1)<<it.B {
			bucket = 0
			it.done = true
		}
		i = 0
	}
	for ; i < bucketCnt; i++ {
		offi := (i + uintptr(it.offset)) & (bucketCnt - 1)
		k := add(unsafe.Pointer(b), dataOffset+offi*uintptr(h.keysize))
		v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(h.keysize)+offi*uintptr(h.valuesize))
		if b.tophash[offi] != empty && b.tophash[offi] != evacuatedEmpty {
			if checkBucket != noCheck {
				// Special case: iterator was started during a grow and the
				// grow is not done yet.  We're working on a bucket whose
				// oldbucket has not been evacuated yet.  Or at least, it wasn't
				// evacuated when we started the bucket.  So we're iterating
				// through the oldbucket, skipping any keys that will go
				// to the other new bucket (each oldbucket expands to two
				// buckets during a grow).
				k2 := k
				if h.flags&indirectKey != 0 {
					k2 = *((*unsafe.Pointer)(k2))
				}
				if goeq(t.key.alg, k2, k2, uintptr(t.key.size)) {
					// If the item in the oldbucket is not destined for
					// the current new bucket in the iteration, skip it.
					hash := gohash(t.key.alg, k2, uintptr(t.key.size), uintptr(h.hash0))
					if hash&(uintptr(1)<<it.B-1) != checkBucket {
						continue
					}
				} else {
					// Hash isn't repeatable if k != k (NaNs).  We need a
					// repeatable and randomish choice of which direction
					// to send NaNs during evacuation.  We'll use the low
					// bit of tophash to decide which way NaNs go.
					// NOTE: this case is why we need two evacuate tophash
					// values, evacuatedX and evacuatedY, that differ in
					// their low bit.
					if checkBucket>>(it.B-1) != uintptr(b.tophash[offi]&1) {
						continue
					}
				}
			}
			if b.tophash[offi] != evacuatedX && b.tophash[offi] != evacuatedY {
				// this is the golden data, we can return it.
				if h.flags&indirectKey != 0 {
					k = *((*unsafe.Pointer)(k))
				}
				it.key = k
				if h.flags&indirectValue != 0 {
					v = *((*unsafe.Pointer)(v))
				}
				it.value = v
			} else {
				// The hash table has grown since the iterator was started.
				// The golden data for this key is now somewhere else.
				k2 := k
				if h.flags&indirectKey != 0 {
					k2 = *((*unsafe.Pointer)(k2))
				}
				if goeq(t.key.alg, k2, k2, uintptr(t.key.size)) {
					// Check the current hash table for the data.
					// This code handles the case where the key
					// has been deleted, updated, or deleted and reinserted.
					// NOTE: we need to regrab the key as it has potentially been
					// updated to an equal() but not identical key (e.g. +0.0 vs -0.0).
					rk, rv := mapaccessK(t, h, k2)
					if rk == nil {
						continue // key has been deleted
					}
					it.key = rk
					it.value = rv
				} else {
					// if key!=key then the entry can't be deleted or
					// updated, so we can just return it.  That's lucky for
					// us because when key!=key we can't look it up
					// successfully in the current table.
					it.key = k2
					if h.flags&indirectValue != 0 {
						v = *((*unsafe.Pointer)(v))
					}
					it.value = v
				}
			}
			it.bucket = bucket
			it.bptr = b
			it.i = i + 1
			it.checkBucket = checkBucket
			return
		}
	}
	b = b.overflow
	i = 0
	goto next
}

func hashGrow(t *maptype, h *hmap) {
	if h.oldbuckets != nil {
		gothrow("evacuation not done in time")
	}
	oldbuckets := h.buckets
	if checkgc {
		memstats.next_gc = memstats.heap_alloc
	}
	newbuckets := unsafe_NewArray(t.bucket, uintptr(1)<<(h.B+1))
	flags := h.flags &^ (iterator | oldIterator)
	if h.flags&iterator != 0 {
		flags |= oldIterator
	}
	// commit the grow (atomic wrt gc)
	h.B++
	h.flags = flags
	h.oldbuckets = oldbuckets
	h.buckets = newbuckets
	h.nevacuate = 0

	// the actual copying of the hash table data is done incrementally
	// by growWork() and evacuate().
}

func growWork(t *maptype, h *hmap, bucket uintptr) {
	noldbuckets := uintptr(1) << (h.B - 1)

	// make sure we evacuate the oldbucket corresponding
	// to the bucket we're about to use
	evacuate(t, h, bucket&(noldbuckets-1))

	// evacuate one more oldbucket to make progress on growing
	if h.oldbuckets != nil {
		evacuate(t, h, h.nevacuate)
	}
}

func evacuate(t *maptype, h *hmap, oldbucket uintptr) {
	b := (*bmap)(add(h.oldbuckets, oldbucket*uintptr(h.bucketsize)))
	newbit := uintptr(1) << (h.B - 1)
	if !evacuated(b) {
		// TODO: reuse overflow buckets instead of using new ones, if there
		// is no iterator using the old buckets.  (If !oldIterator.)

		x := (*bmap)(add(h.buckets, oldbucket*uintptr(h.bucketsize)))
		y := (*bmap)(add(h.buckets, (oldbucket+newbit)*uintptr(h.bucketsize)))
		xi := 0
		yi := 0
		xk := add(unsafe.Pointer(x), dataOffset)
		yk := add(unsafe.Pointer(y), dataOffset)
		xv := add(xk, bucketCnt*uintptr(h.keysize))
		yv := add(yk, bucketCnt*uintptr(h.keysize))
		for ; b != nil; b = b.overflow {
			k := add(unsafe.Pointer(b), dataOffset)
			v := add(k, bucketCnt*uintptr(h.keysize))
			for i := 0; i < bucketCnt; i, k, v = i+1, add(k, uintptr(h.keysize)), add(v, uintptr(h.valuesize)) {
				top := b.tophash[i]
				if top == empty {
					b.tophash[i] = evacuatedEmpty
					continue
				}
				if top < minTopHash {
					gothrow("bad map state")
				}
				k2 := k
				if h.flags&indirectKey != 0 {
					k2 = *((*unsafe.Pointer)(k2))
				}
				// Compute hash to make our evacuation decision (whether we need
				// to send this key/value to bucket x or bucket y).
				hash := gohash(t.key.alg, k2, uintptr(t.key.size), uintptr(h.hash0))
				if h.flags&iterator != 0 {
					if !goeq(t.key.alg, k2, k2, uintptr(t.key.size)) {
						// If key != key (NaNs), then the hash could be (and probably
						// will be) entirely different from the old hash.  Moreover,
						// it isn't reproducible.  Reproducibility is required in the
						// presence of iterators, as our evacuation decision must
						// match whatever decision the iterator made.
						// Fortunately, we have the freedom to send these keys either
						// way.  Also, tophash is meaningless for these kinds of keys.
						// We let the low bit of tophash drive the evacuation decision.
						// We recompute a new random tophash for the next level so
						// these keys will get evenly distributed across all buckets
						// after multiple grows.
						if (top & 1) != 0 {
							hash |= newbit
						} else {
							hash &^= newbit
						}
						top = uint8(hash >> (ptrSize*8 - 8))
						if top < minTopHash {
							top += minTopHash
						}
					}
				}
				if (hash & newbit) == 0 {
					b.tophash[i] = evacuatedX
					if xi == bucketCnt {
						if checkgc {
							memstats.next_gc = memstats.heap_alloc
						}
						newx := (*bmap)(unsafe_New(t.bucket))
						x.overflow = newx
						x = newx
						xi = 0
						xk = add(unsafe.Pointer(x), dataOffset)
						xv = add(xk, bucketCnt*uintptr(h.keysize))
					}
					x.tophash[xi] = top
					if h.flags&indirectKey != 0 {
						*(*unsafe.Pointer)(xk) = k2 // copy pointer
					} else {
						memmove(xk, k, uintptr(t.key.size)) // copy value
					}
					if h.flags&indirectValue != 0 {
						*(*unsafe.Pointer)(xv) = *(*unsafe.Pointer)(v)
					} else {
						memmove(xv, v, uintptr(t.elem.size))
					}
					xi++
					xk = add(xk, uintptr(h.keysize))
					xv = add(xv, uintptr(h.valuesize))
				} else {
					b.tophash[i] = evacuatedY
					if yi == bucketCnt {
						if checkgc {
							memstats.next_gc = memstats.heap_alloc
						}
						newy := (*bmap)(unsafe_New(t.bucket))
						y.overflow = newy
						y = newy
						yi = 0
						yk = add(unsafe.Pointer(y), dataOffset)
						yv = add(yk, bucketCnt*uintptr(h.keysize))
					}
					y.tophash[yi] = top
					if h.flags&indirectKey != 0 {
						*(*unsafe.Pointer)(yk) = k2
					} else {
						memmove(yk, k, uintptr(t.key.size))
					}
					if h.flags&indirectValue != 0 {
						*(*unsafe.Pointer)(yv) = *(*unsafe.Pointer)(v)
					} else {
						memmove(yv, v, uintptr(t.elem.size))
					}
					yi++
					yk = add(yk, uintptr(h.keysize))
					yv = add(yv, uintptr(h.valuesize))
				}
			}
		}
		// Unlink the overflow buckets & clear key/value to help GC.
		if h.flags&oldIterator == 0 {
			b = (*bmap)(add(h.oldbuckets, oldbucket*uintptr(h.bucketsize)))
			b.overflow = nil
			memclr(add(unsafe.Pointer(b), dataOffset), uintptr(h.bucketsize)-dataOffset)
		}
	}

	// Advance evacuation mark
	if oldbucket == h.nevacuate {
		h.nevacuate = oldbucket + 1
		if oldbucket+1 == newbit { // newbit == # of oldbuckets
			// Growing is all done.  Free old main bucket array.
			h.oldbuckets = nil
		}
	}
}

func ismapkey(t *_type) bool {
	return *(*uintptr)(unsafe.Pointer(&t.alg.hash)) != nohashcode
}

// Reflect stubs.  Called from ../reflect/asm_*.s

func reflect_makemap(t *maptype) *hmap {
	return makemap(t, 0)
}

func reflect_mapaccess(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	val, ok := mapaccess2(t, h, key)
	if !ok {
		// reflect wants nil for a missing element
		val = nil
	}
	return val
}

func reflect_mapassign(t *maptype, h *hmap, key unsafe.Pointer, val unsafe.Pointer) {
	mapassign1(t, h, key, val)
}

func reflect_mapdelete(t *maptype, h *hmap, key unsafe.Pointer) {
	mapdelete(t, h, key)
}

func reflect_mapiterinit(t *maptype, h *hmap) *hiter {
	it := new(hiter)
	mapiterinit(t, h, it)
	return it
}

func reflect_mapiternext(it *hiter) {
	mapiternext(it)
}

func reflect_mapiterkey(it *hiter) unsafe.Pointer {
	return it.key
}

func reflect_maplen(h *hmap) int {
	if h == nil {
		return 0
	}
	if raceenabled {
		callerpc := gogetcallerpc(unsafe.Pointer(&h))
		fn := reflect_maplen
		pc := **(**uintptr)(unsafe.Pointer(&fn))
		racereadpc(unsafe.Pointer(h), callerpc, pc)
	}
	return h.count
}

func reflect_ismapkey(t *_type) bool {
	return ismapkey(t)
}
