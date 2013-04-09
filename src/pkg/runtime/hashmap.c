// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "hashmap.h"
#include "type.h"
#include "race.h"

// This file contains the implementation of Go's map type.
//
// The map is just a hash table.  The data is arranged
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

// Maximum number of key/value pairs a bucket can hold.
#define BUCKETSIZE 8

// Maximum average load of a bucket that triggers growth.
#define LOAD 6.5

// Picking LOAD: too large and we have lots of overflow
// buckets, too small and we waste a lot of space.  I wrote
// a simple program to check some stats for different loads:
// (64-bit, 8 byte keys and values)
//        LOAD    %overflow  bytes/entry     hitprobe    missprobe
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

// Maximum key or value size to keep inline (instead of mallocing per element).
// Must fit in a uint8.
// Fast versions cannot handle big values - the cutoff size for
// fast versions in ../../cmd/gc/walk.c must be at most this value.
#define MAXKEYSIZE 128
#define MAXVALUESIZE 128

typedef struct Bucket Bucket;
struct Bucket
{
	uint8  tophash[BUCKETSIZE]; // top 8 bits of hash of each entry (0 = empty)
	Bucket *overflow;           // overflow bucket, if any
	byte   data[1];             // BUCKETSIZE keys followed by BUCKETSIZE values
};
// NOTE: packing all the keys together and then all the values together makes the
// code a bit more complicated than alternating key/value/key/value/... but it allows
// us to eliminate padding which would be needed for, e.g., map[int64]int8.

// Low-order bit of overflow field is used to mark a bucket as already evacuated
// without destroying the overflow pointer.
// Only buckets in oldbuckets will be marked as evacuated.
// Evacuated bit will be set identically on the base bucket and any overflow buckets.
#define evacuated(b) (((uintptr)(b)->overflow & 1) != 0)
#define overflowptr(b) ((Bucket*)((uintptr)(b)->overflow & ~(uintptr)1))

// Initialize bucket to the empty state.  This only works if BUCKETSIZE==8!
#define clearbucket(b) { *(uint64*)((b)->tophash) = 0; (b)->overflow = nil; }

struct Hmap
{
	uintgo  count;        // # live cells == size of map.  Must be first (used by len() builtin)
	uint32  flags;
	uint8   B;            // log_2 of # of buckets (can hold up to LOAD * 2^B items)
	uint8   keysize;      // key size in bytes
	uint8   valuesize;    // value size in bytes
	uint16  bucketsize;   // bucket size in bytes

	uintptr hash0;        // hash seed
	byte    *buckets;     // array of 2^B Buckets. may be nil if count==0.
	byte    *oldbuckets;  // previous bucket array of half the size, non-nil only when growing
	uintptr nevacuate;    // progress counter for evacuation (buckets less than this have been evacuated)
};

// possible flags
enum
{
	IndirectKey = 1,    // storing pointers to keys
	IndirectValue = 2,  // storing pointers to values
	Iterator = 4,       // there may be an iterator using buckets
	OldIterator = 8,    // there may be an iterator using oldbuckets
	CanFreeBucket = 16, // ok to free buckets
	CanFreeKey = 32,    // keys are indirect and ok to free keys
};

// Macros for dereferencing indirect keys
#define IK(h, p) (((h)->flags & IndirectKey) != 0 ? *(byte**)(p) : (p))
#define IV(h, p) (((h)->flags & IndirectValue) != 0 ? *(byte**)(p) : (p))

enum
{
	docheck = 0,  // check invariants before and after every op.  Slow!!!
	debug = 0,    // print every operation
	checkgc = 0 || docheck,  // check interaction of mallocgc() with the garbage collector
};
static void
check(MapType *t, Hmap *h)
{
	uintptr bucket, oldbucket;
	Bucket *b;
	uintptr i;
	uintptr hash;
	uintgo cnt;
	uint8 top;
	bool eq;
	byte *k, *v;

	cnt = 0;

	// check buckets
	for(bucket = 0; bucket < (uintptr)1 << h->B; bucket++) {
		if(h->oldbuckets != nil) {
			oldbucket = bucket & (((uintptr)1 << (h->B - 1)) - 1);
			b = (Bucket*)(h->oldbuckets + oldbucket * h->bucketsize);
			if(!evacuated(b))
				continue; // b is still uninitialized
		}
		for(b = (Bucket*)(h->buckets + bucket * h->bucketsize); b != nil; b = b->overflow) {
			for(i = 0, k = b->data, v = k + h->keysize * BUCKETSIZE; i < BUCKETSIZE; i++, k += h->keysize, v += h->valuesize) {
				if(b->tophash[i] == 0)
					continue;
				cnt++;
				t->key->alg->equal(&eq, t->key->size, IK(h, k), IK(h, k));
				if(!eq)
					continue; // NaN!
				hash = h->hash0;
				t->key->alg->hash(&hash, t->key->size, IK(h, k));
				top = hash >> (8*sizeof(uintptr) - 8);
				if(top == 0)
					top = 1;
				if(top != b->tophash[i])
					runtime·throw("bad hash");
			}
		}
	}

	// check oldbuckets
	if(h->oldbuckets != nil) {
		for(oldbucket = 0; oldbucket < (uintptr)1 << (h->B - 1); oldbucket++) {
			b = (Bucket*)(h->oldbuckets + oldbucket * h->bucketsize);
			if(evacuated(b))
				continue;
			if(oldbucket < h->nevacuate)
				runtime·throw("bucket became unevacuated");
			for(; b != nil; b = overflowptr(b)) {
				for(i = 0, k = b->data, v = k + h->keysize * BUCKETSIZE; i < BUCKETSIZE; i++, k += h->keysize, v += h->valuesize) {
					if(b->tophash[i] == 0)
						continue;
					cnt++;
					t->key->alg->equal(&eq, t->key->size, IK(h, k), IK(h, k));
					if(!eq)
						continue; // NaN!
					hash = h->hash0;
					t->key->alg->hash(&hash, t->key->size, IK(h, k));
					top = hash >> (8*sizeof(uintptr) - 8);
					if(top == 0)
						top = 1;
					if(top != b->tophash[i])
						runtime·throw("bad hash (old)");
				}
			}
		}
	}

	if(cnt != h->count) {
		runtime·printf("%D %D\n", (uint64)cnt, (uint64)h->count);
		runtime·throw("entries missing");
	}
}

static void
hash_init(MapType *t, Hmap *h, uint32 hint)
{
	uint8 B;
	byte *buckets;
	uintptr i;
	uintptr keysize, valuesize, bucketsize;
	uint8 flags;
	Bucket *b;

	flags = CanFreeBucket;

	// figure out how big we have to make everything
	keysize = t->key->size;
	if(keysize > MAXKEYSIZE) {
		flags |= IndirectKey | CanFreeKey;
		keysize = sizeof(byte*);
	}
	valuesize = t->elem->size;
	if(valuesize > MAXVALUESIZE) {
		flags |= IndirectValue;
		valuesize = sizeof(byte*);
	}
	bucketsize = offsetof(Bucket, data[0]) + (keysize + valuesize) * BUCKETSIZE;

	// invariants we depend on.  We should probably check these at compile time
	// somewhere, but for now we'll do it here.
	if(t->key->align > BUCKETSIZE)
		runtime·throw("key align too big");
	if(t->elem->align > BUCKETSIZE)
		runtime·throw("value align too big");
	if(t->key->size % t->key->align != 0)
		runtime·throw("key size not a multiple of key align");
	if(t->elem->size % t->elem->align != 0)
		runtime·throw("value size not a multiple of value align");
	if(BUCKETSIZE < 8)
		runtime·throw("bucketsize too small for proper alignment");
	if(BUCKETSIZE != 8)
		runtime·throw("must redo clearbucket");
	if(sizeof(void*) == 4 && t->key->align > 4)
		runtime·throw("need padding in bucket (key)");
	if(sizeof(void*) == 4 && t->elem->align > 4)
		runtime·throw("need padding in bucket (value)");

	// find size parameter which will hold the requested # of elements
	B = 0;
	while(hint > BUCKETSIZE && hint > LOAD * ((uintptr)1 << B))
		B++;

	// allocate initial hash table
	// If hint is large zeroing this memory could take a while.
	if(checkgc) mstats.next_gc = mstats.heap_alloc;
	if(B == 0) {
		// done lazily later.
		buckets = nil;
	} else {
		buckets = runtime·mallocgc(bucketsize << B, 0, 1, 0);
		for(i = 0; i < (uintptr)1 << B; i++) {
			b = (Bucket*)(buckets + i * bucketsize);
			clearbucket(b);
		}
	}

	// initialize Hmap
	// Note: we save all these stores to the end so gciter doesn't see
	// a partially initialized map.
	h->count = 0;
	h->B = B;
	h->flags = flags;
	h->keysize = keysize;
	h->valuesize = valuesize;
	h->bucketsize = bucketsize;
	h->hash0 = runtime·fastrand1();
	h->buckets = buckets;
	h->oldbuckets = nil;
	h->nevacuate = 0;
	if(docheck)
		check(t, h);
}

// Moves entries in oldbuckets[i] to buckets[i] and buckets[i+2^k].
// We leave the original bucket intact, except for the evacuated marks, so that
// iterators can still iterate through the old buckets.
static void
evacuate(MapType *t, Hmap *h, uintptr oldbucket)
{
	Bucket *b;
	Bucket *nextb;
	Bucket *x, *y;
	Bucket *newx, *newy;
	uintptr xi, yi;
	uintptr newbit;
	uintptr hash;
	uintptr i;
	byte *k, *v;
	byte *xk, *yk, *xv, *yv;
	byte *ob;

	b = (Bucket*)(h->oldbuckets + oldbucket * h->bucketsize);
	newbit = (uintptr)1 << (h->B - 1);

	if(!evacuated(b)) {
		// TODO: reuse overflow buckets instead of using new ones, if there
		// is no iterator using the old buckets.  (If CanFreeBuckets and !OldIterator.)

		x = (Bucket*)(h->buckets + oldbucket * h->bucketsize);
		y = (Bucket*)(h->buckets + (oldbucket + newbit) * h->bucketsize);
		clearbucket(x);
		clearbucket(y);
		xi = 0;
		yi = 0;
		xk = x->data;
		yk = y->data;
		xv = xk + h->keysize * BUCKETSIZE;
		yv = yk + h->keysize * BUCKETSIZE;
		do {
			for(i = 0, k = b->data, v = k + h->keysize * BUCKETSIZE; i < BUCKETSIZE; i++, k += h->keysize, v += h->valuesize) {
				if(b->tophash[i] == 0)
					continue;
				hash = h->hash0;
				t->key->alg->hash(&hash, t->key->size, IK(h, k));
				// NOTE: if key != key, then this hash could be (and probably will be)
				// entirely different from the old hash.  We effectively only update
				// the B'th bit of the hash in this case.
				if((hash & newbit) == 0) {
					if(xi == BUCKETSIZE) {
						if(checkgc) mstats.next_gc = mstats.heap_alloc;
						newx = runtime·mallocgc(h->bucketsize, 0, 1, 0);
						clearbucket(newx);
						x->overflow = newx;
						x = newx;
						xi = 0;
						xk = x->data;
						xv = xk + h->keysize * BUCKETSIZE;
					}
					x->tophash[xi] = b->tophash[i];
					if((h->flags & IndirectKey) != 0) {
						*(byte**)xk = *(byte**)k;               // copy pointer
					} else {
						t->key->alg->copy(t->key->size, xk, k); // copy value
					}
					if((h->flags & IndirectValue) != 0) {
						*(byte**)xv = *(byte**)v;
					} else {
						t->elem->alg->copy(t->elem->size, xv, v);
					}
					xi++;
					xk += h->keysize;
					xv += h->valuesize;
				} else {
					if(yi == BUCKETSIZE) {
						if(checkgc) mstats.next_gc = mstats.heap_alloc;
						newy = runtime·mallocgc(h->bucketsize, 0, 1, 0);
						clearbucket(newy);
						y->overflow = newy;
						y = newy;
						yi = 0;
						yk = y->data;
						yv = yk + h->keysize * BUCKETSIZE;
					}
					y->tophash[yi] = b->tophash[i];
					if((h->flags & IndirectKey) != 0) {
						*(byte**)yk = *(byte**)k;
					} else {
						t->key->alg->copy(t->key->size, yk, k);
					}
					if((h->flags & IndirectValue) != 0) {
						*(byte**)yv = *(byte**)v;
					} else {
						t->elem->alg->copy(t->elem->size, yv, v);
					}
					yi++;
					yk += h->keysize;
					yv += h->valuesize;
				}
			}

			// mark as evacuated so we don't do it again.
			// this also tells any iterators that this data isn't golden anymore.
			nextb = b->overflow;
			b->overflow = (Bucket*)((uintptr)nextb + 1);

			b = nextb;
		} while(b != nil);

		// Free old overflow buckets as much as we can.
		if((h->flags & OldIterator) == 0) {
			b = (Bucket*)(h->oldbuckets + oldbucket * h->bucketsize);
			if((h->flags & CanFreeBucket) != 0) {
				while((nextb = overflowptr(b)) != nil) {
					b->overflow = nextb->overflow;
					runtime·free(nextb);
				}
			} else {
				// can't explicitly free overflow buckets, but at least
				// we can unlink them.
				b->overflow = (Bucket*)1;
			}
		}
	}

	// advance evacuation mark
	if(oldbucket == h->nevacuate) {
		h->nevacuate = oldbucket + 1;
		if(oldbucket + 1 == newbit) { // newbit == # of oldbuckets
			// free main bucket array
			if((h->flags & (OldIterator | CanFreeBucket)) == CanFreeBucket) {
				ob = h->oldbuckets;
				h->oldbuckets = nil;
				runtime·free(ob);
			} else {
				h->oldbuckets = nil;
			}
		}
	}
	if(docheck)
		check(t, h);
}

static void
grow_work(MapType *t, Hmap *h, uintptr bucket)
{
	uintptr noldbuckets;

	noldbuckets = (uintptr)1 << (h->B - 1);

	// make sure we evacuate the oldbucket corresponding
	// to the bucket we're about to use
	evacuate(t, h, bucket & (noldbuckets - 1));

	// evacuate one more oldbucket to make progress on growing
	if(h->oldbuckets != nil)
		evacuate(t, h, h->nevacuate);
}

static void
hash_grow(MapType *t, Hmap *h)
{
	byte *old_buckets;
	byte *new_buckets;
	uint8 flags;

	// allocate a bigger hash table
	if(h->oldbuckets != nil)
		runtime·throw("evacuation not done in time");
	old_buckets = h->buckets;
	// NOTE: this could be a big malloc, but since we don't need zeroing it is probably fast.
	if(checkgc) mstats.next_gc = mstats.heap_alloc;
	new_buckets = runtime·mallocgc((uintptr)h->bucketsize << (h->B + 1), 0, 1, 0);
	flags = (h->flags & ~(Iterator | OldIterator));
	if((h->flags & Iterator) != 0) {
		flags |= OldIterator;
		// We can't free indirect keys any more, as
		// they are potentially aliased across buckets.
		flags &= ~CanFreeKey;
	}

	// commit the grow (atomic wrt gc)
	h->B++;
	h->flags = flags;
	h->oldbuckets = old_buckets;
	h->buckets = new_buckets;
	h->nevacuate = 0;

	// the actual copying of the hash table data is done incrementally
	// by grow_work() and evacuate().
	if(docheck)
		check(t, h);
}

// returns ptr to value associated with key *keyp, or nil if none.
// if it returns non-nil, updates *keyp to point to the currently stored key.
static byte*
hash_lookup(MapType *t, Hmap *h, byte **keyp)
{
	void *key;
	uintptr hash;
	uintptr bucket, oldbucket;
	Bucket *b;
	uint8 top;
	uintptr i;
	bool eq;
	byte *k, *k2, *v;

	key = *keyp;
	if(docheck)
		check(t, h);
	if(h->count == 0)
		return nil;
	hash = h->hash0;
	t->key->alg->hash(&hash, t->key->size, key);
	bucket = hash & (((uintptr)1 << h->B) - 1);
	if(h->oldbuckets != nil) {
		oldbucket = bucket & (((uintptr)1 << (h->B - 1)) - 1);
		b = (Bucket*)(h->oldbuckets + oldbucket * h->bucketsize);
		if(evacuated(b)) {
			b = (Bucket*)(h->buckets + bucket * h->bucketsize);
		}
	} else {
		b = (Bucket*)(h->buckets + bucket * h->bucketsize);
	}
	top = hash >> (sizeof(uintptr)*8 - 8);
	if(top == 0)
		top = 1;
	do {
		for(i = 0, k = b->data, v = k + h->keysize * BUCKETSIZE; i < BUCKETSIZE; i++, k += h->keysize, v += h->valuesize) {
			if(b->tophash[i] == top) {
				k2 = IK(h, k);
				t->key->alg->equal(&eq, t->key->size, key, k2);
				if(eq) {
					*keyp = k2;
					return IV(h, v);
				}
			}
		}
		b = b->overflow;
	} while(b != nil);
	return nil;
}

// When an item is not found, fast versions return a pointer to this zeroed memory.
static uint8 empty_value[MAXVALUESIZE];

// Specialized versions of mapaccess1 for specific types.
// See ./hashmap_fast.c and ../../cmd/gc/walk.c.
#define HASH_LOOKUP1 runtime·mapaccess1_fast32
#define HASH_LOOKUP2 runtime·mapaccess2_fast32
#define KEYTYPE uint32
#define HASHFUNC runtime·algarray[AMEM32].hash
#define EQFUNC(x,y) ((x) == (y))
#define EQMAYBE(x,y) ((x) == (y))
#define HASMAYBE false
#define QUICKEQ(x) true
#include "hashmap_fast.c"

#undef HASH_LOOKUP1
#undef HASH_LOOKUP2
#undef KEYTYPE
#undef HASHFUNC
#undef EQFUNC
#undef EQMAYBE
#undef HASMAYBE
#undef QUICKEQ

#define HASH_LOOKUP1 runtime·mapaccess1_fast64
#define HASH_LOOKUP2 runtime·mapaccess2_fast64
#define KEYTYPE uint64
#define HASHFUNC runtime·algarray[AMEM64].hash
#define EQFUNC(x,y) ((x) == (y))
#define EQMAYBE(x,y) ((x) == (y))
#define HASMAYBE false
#define QUICKEQ(x) true
#include "hashmap_fast.c"

#undef HASH_LOOKUP1
#undef HASH_LOOKUP2
#undef KEYTYPE
#undef HASHFUNC
#undef EQFUNC
#undef EQMAYBE
#undef HASMAYBE
#undef QUICKEQ

#define HASH_LOOKUP1 runtime·mapaccess1_faststr
#define HASH_LOOKUP2 runtime·mapaccess2_faststr
#define KEYTYPE String
#define HASHFUNC runtime·algarray[ASTRING].hash
#define EQFUNC(x,y) ((x).len == (y).len && ((x).str == (y).str || runtime·memeq((x).str, (y).str, (x).len)))
#define EQMAYBE(x,y) ((x).len == (y).len)
#define HASMAYBE true
#define QUICKEQ(x) ((x).len < 32)
#include "hashmap_fast.c"

static void
hash_insert(MapType *t, Hmap *h, void *key, void *value)
{
	uintptr hash;
	uintptr bucket;
	uintptr i;
	bool eq;
	Bucket *b;
	Bucket *newb;
	uint8 *inserti;
	byte *insertk, *insertv;
	uint8 top;
	byte *k, *v;
	byte *kmem, *vmem;

	if(docheck)
		check(t, h);
	hash = h->hash0;
	t->key->alg->hash(&hash, t->key->size, key);
	if(h->buckets == nil) {
		h->buckets = runtime·mallocgc(h->bucketsize, 0, 1, 0);
		b = (Bucket*)(h->buckets);
		clearbucket(b);
	}

 again:
	bucket = hash & (((uintptr)1 << h->B) - 1);
	if(h->oldbuckets != nil)
		grow_work(t, h, bucket);
	b = (Bucket*)(h->buckets + bucket * h->bucketsize);
	top = hash >> (sizeof(uintptr)*8 - 8);
	if(top == 0)
		top = 1;
	inserti = 0;
	insertk = nil;
	insertv = nil;
	while(true) {
		for(i = 0, k = b->data, v = k + h->keysize * BUCKETSIZE; i < BUCKETSIZE; i++, k += h->keysize, v += h->valuesize) {
			if(b->tophash[i] != top) {
				if(b->tophash[i] == 0 && inserti == nil) {
					inserti = &b->tophash[i];
					insertk = k;
					insertv = v;
				}
				continue;
			}
			t->key->alg->equal(&eq, t->key->size, key, IK(h, k));
			if(!eq)
				continue;
			// already have a mapping for key.  Update it.
			t->key->alg->copy(t->key->size, IK(h, k), key); // Need to update key for keys which are distinct but equal (e.g. +0.0 and -0.0)
			t->elem->alg->copy(t->elem->size, IV(h, v), value);
			if(docheck)
				check(t, h);
			return;
		}
		if(b->overflow == nil)
			break;
		b = b->overflow;
	}

	// did not find mapping for key.  Allocate new cell & add entry.
	if(h->count >= LOAD * ((uintptr)1 << h->B) && h->count >= BUCKETSIZE) {
		hash_grow(t, h);
		goto again; // Growing the table invalidates everything, so try again
	}

	if(inserti == nil) {
		// all current buckets are full, allocate a new one.
		if(checkgc) mstats.next_gc = mstats.heap_alloc;
		newb = runtime·mallocgc(h->bucketsize, 0, 1, 0);
		clearbucket(newb);
		b->overflow = newb;
		inserti = newb->tophash;
		insertk = newb->data;
		insertv = insertk + h->keysize * BUCKETSIZE;
	}

	// store new key/value at insert position
	if((h->flags & IndirectKey) != 0) {
		if(checkgc) mstats.next_gc = mstats.heap_alloc;
		kmem = runtime·mallocgc(t->key->size, 0, 1, 0);
		*(byte**)insertk = kmem;
		insertk = kmem;
	}
	if((h->flags & IndirectValue) != 0) {
		if(checkgc) mstats.next_gc = mstats.heap_alloc;
		vmem = runtime·mallocgc(t->elem->size, 0, 1, 0);
		*(byte**)insertv = vmem;
		insertv = vmem;
	}
	t->key->alg->copy(t->key->size, insertk, key);
	t->elem->alg->copy(t->elem->size, insertv, value);
	*inserti = top;
	h->count++;
	if(docheck)
		check(t, h);
}

static void
hash_remove(MapType *t, Hmap *h, void *key)
{
	uintptr hash;
	uintptr bucket;
	Bucket *b;
	uint8 top;
	uintptr i;
	byte *k, *v;
	bool eq;
	
	if(docheck)
		check(t, h);
	if(h->count == 0)
		return;
	hash = h->hash0;
	t->key->alg->hash(&hash, t->key->size, key);
	bucket = hash & (((uintptr)1 << h->B) - 1);
	if(h->oldbuckets != nil)
		grow_work(t, h, bucket);
	b = (Bucket*)(h->buckets + bucket * h->bucketsize);
	top = hash >> (sizeof(uintptr)*8 - 8);
	if(top == 0)
		top = 1;
	do {
		for(i = 0, k = b->data, v = k + h->keysize * BUCKETSIZE; i < BUCKETSIZE; i++, k += h->keysize, v += h->valuesize) {
			if(b->tophash[i] != top)
				continue;
			t->key->alg->equal(&eq, t->key->size, key, IK(h, k));
			if(!eq)
				continue;

			if((h->flags & CanFreeKey) != 0) {
				k = *(byte**)k;
			}
			if((h->flags & IndirectValue) != 0) {
				v = *(byte**)v;
			}

			b->tophash[i] = 0;
			h->count--;
			
			if((h->flags & CanFreeKey) != 0) {
				runtime·free(k);
			}
			if((h->flags & IndirectValue) != 0) {
				runtime·free(v);
			}
			// TODO: consolidate buckets if they are mostly empty
			// can only consolidate if there are no live iterators at this size.
			if(docheck)
				check(t, h);
			return;
		}
		b = b->overflow;
	} while(b != nil);
}

// TODO: shrink the map, the same way we grow it.

// If you modify hash_iter, also change cmd/gc/range.c to indicate
// the size of this structure.
struct hash_iter
{
	uint8* key; // Must be in first position.  Write nil to indicate iteration end (see cmd/gc/range.c).
	uint8* value;

	MapType *t;
	Hmap *h;

	// end point for iteration
	uintptr endbucket;
	bool wrapped;

	// state of table at time iterator is initialized
	uint8 B;
	byte *buckets;

	// iter state
	uintptr bucket;
	struct Bucket *bptr;
	uintptr i;
	intptr check_bucket;
};

// iterator state:
// bucket: the current bucket ID
// b: the current Bucket in the chain
// i: the next offset to check in the current bucket
static void
hash_iter_init(MapType *t, Hmap *h, struct hash_iter *it)
{
	uint32 old;

	if(sizeof(struct hash_iter) / sizeof(uintptr) != 11) {
		runtime·throw("hash_iter size incorrect"); // see ../../cmd/gc/range.c
	}
	it->t = t;
	it->h = h;

	// grab snapshot of bucket state
	it->B = h->B;
	it->buckets = h->buckets;

	// iterator state
	it->bucket = it->endbucket = runtime·fastrand1() & (((uintptr)1 << h->B) - 1);
	it->wrapped = false;
	it->bptr = nil;

	// Remember we have an iterator.
	// Can run concurrently with another hash_iter_init() and with reflect·mapiterinit().
	for(;;) {
		old = h->flags;
		if((old&(Iterator|OldIterator)) == (Iterator|OldIterator))
			break;
		if(runtime·cas(&h->flags, old, old|Iterator|OldIterator))
			break;
	}

	if(h->buckets == nil) {
		// Empty map. Force next hash_next to exit without
		// evalulating h->bucket.
		it->wrapped = true;
	}
}

// initializes it->key and it->value to the next key/value pair
// in the iteration, or nil if we've reached the end.
static void
hash_next(struct hash_iter *it)
{
	Hmap *h;
	MapType *t;
	uintptr bucket, oldbucket;
	uintptr hash;
	Bucket *b;
	uintptr i;
	intptr check_bucket;
	bool eq;
	byte *k, *v;
	byte *rk, *rv;

	h = it->h;
	t = it->t;
	bucket = it->bucket;
	b = it->bptr;
	i = it->i;
	check_bucket = it->check_bucket;

next:
	if(b == nil) {
		if(bucket == it->endbucket && it->wrapped) {
			// end of iteration
			it->key = nil;
			it->value = nil;
			return;
		}
		if(h->oldbuckets != nil && it->B == h->B) {
			// Iterator was started in the middle of a grow, and the grow isn't done yet.
			// If the bucket we're looking at hasn't been filled in yet (i.e. the old
			// bucket hasn't been evacuated) then we need to iterate through the old
			// bucket and only return the ones that will be migrated to this bucket.
			oldbucket = bucket & (((uintptr)1 << (it->B - 1)) - 1);
			b = (Bucket*)(h->oldbuckets + oldbucket * h->bucketsize);
			if(!evacuated(b)) {
				check_bucket = bucket;
			} else {
				b = (Bucket*)(it->buckets + bucket * h->bucketsize);
				check_bucket = -1;
			}
		} else {
			b = (Bucket*)(it->buckets + bucket * h->bucketsize);
			check_bucket = -1;
		}
		bucket++;
		if(bucket == ((uintptr)1 << it->B)) {
			bucket = 0;
			it->wrapped = true;
		}
		i = 0;
	}
	k = b->data + h->keysize * i;
	v = b->data + h->keysize * BUCKETSIZE + h->valuesize * i;
	for(; i < BUCKETSIZE; i++, k += h->keysize, v += h->valuesize) {
		if(b->tophash[i] != 0) {
			if(check_bucket >= 0) {
				// Special case: iterator was started during a grow and the
				// grow is not done yet.  We're working on a bucket whose
				// oldbucket has not been evacuated yet.  So we iterate
				// through the oldbucket, skipping any keys that will go
				// to the other new bucket (each oldbucket expands to two
				// buckets during a grow).
				t->key->alg->equal(&eq, t->key->size, IK(h, k), IK(h, k));
				if(!eq) {
					// Hash is meaningless if k != k (NaNs).  Return all
					// NaNs during the first of the two new buckets.
					if(bucket >= ((uintptr)1 << (it->B - 1))) {
						continue;
					}
				} else {
					// If the item in the oldbucket is not destined for
					// the current new bucket in the iteration, skip it.
					hash = h->hash0;
					t->key->alg->hash(&hash, t->key->size, IK(h, k));
					if((hash & (((uintptr)1 << it->B) - 1)) != check_bucket) {
						continue;
					}
				}
			}
			if(!evacuated(b)) {
				// this is the golden data, we can return it.
				it->key = IK(h, k);
				it->value = IV(h, v);
			} else {
				// The hash table has grown since the iterator was started.
				// The golden data for this key is now somewhere else.
				t->key->alg->equal(&eq, t->key->size, IK(h, k), IK(h, k));
				if(eq) {
					// Check the current hash table for the data.
					// This code handles the case where the key
					// has been deleted, updated, or deleted and reinserted.
					// NOTE: we need to regrab the key as it has potentially been
					// updated to an equal() but not identical key (e.g. +0.0 vs -0.0).
					rk = IK(h, k);
					rv = hash_lookup(t, it->h, &rk);
					if(rv == nil)
						continue; // key has been deleted
					it->key = rk;
					it->value = rv;
				} else {
					// if key!=key then the entry can't be deleted or
					// updated, so we can just return it.  That's lucky for
					// us because when key!=key we can't look it up
					// successfully in the current table.
					it->key = IK(h, k);
					it->value = IV(h, v);
				}
			}
			it->bucket = bucket;
			it->bptr = b;
			it->i = i + 1;
			it->check_bucket = check_bucket;
			return;
		}
	}
	b = overflowptr(b);
	i = 0;
	goto next;
}


#define PHASE_BUCKETS      0
#define PHASE_OLD_BUCKETS  1
#define PHASE_TABLE        2
#define PHASE_OLD_TABLE    3
#define PHASE_DONE         4

// Initialize the iterator.
// Returns false if Hmap contains no pointers (in which case the iterator is not initialized).
bool
hash_gciter_init (Hmap *h, struct hash_gciter *it)
{
	// GC during map initialization or on an empty map.
	if(h->buckets == nil)
		return false;

	it->h = h;
	it->phase = PHASE_BUCKETS;
	it->bucket = 0;
	it->b = nil;

	// TODO: finish evacuating oldbuckets so that we can collect
	// oldbuckets?  We don't want to keep a partially evacuated
	// table around forever, so each gc could make at least some
	// evacuation progress.  Need to be careful about concurrent
	// access if we do concurrent gc.  Even if not, we don't want
	// to make the gc pause any longer than it has to be.

	return true;
}

// Returns true and fills *data with internal structure/key/value data,
// or returns false if the iterator has terminated.
// Ugh, this interface is really annoying.  I want a callback fn!
bool
hash_gciter_next(struct hash_gciter *it, struct hash_gciter_data *data)
{
	Hmap *h;
	uintptr bucket, oldbucket;
	Bucket *b, *oldb;
	uintptr i;
	byte *k, *v;

	h = it->h;
	bucket = it->bucket;
	b = it->b;
	i = it->i;

	data->st = nil;
	data->key_data = nil;
	data->val_data = nil;
	data->indirectkey = (h->flags & IndirectKey) != 0;
	data->indirectval = (h->flags & IndirectValue) != 0;

next:
	switch (it->phase) {
	case PHASE_BUCKETS:
		if(b != nil) {
			k = b->data + h->keysize * i;
			v = b->data + h->keysize * BUCKETSIZE + h->valuesize * i;
			for(; i < BUCKETSIZE; i++, k += h->keysize, v += h->valuesize) {
				if(b->tophash[i] != 0) {
					data->key_data = k;
					data->val_data = v;
					it->bucket = bucket;
					it->b = b;
					it->i = i + 1;
					return true;
				}
			}
			b = b->overflow;
			if(b != nil) {
				data->st = (byte*)b;
				it->bucket = bucket;
				it->b = b;
				it->i = 0;
				return true;
			}
		}
		while(bucket < ((uintptr)1 << h->B)) {
			if(h->oldbuckets != nil) {
				oldbucket = bucket & (((uintptr)1 << (h->B - 1)) - 1);
				oldb = (Bucket*)(h->oldbuckets + oldbucket * h->bucketsize);
				if(!evacuated(oldb)) {
					// new bucket isn't valid yet
					bucket++;
					continue;
				}
			}
			b = (Bucket*)(h->buckets + bucket * h->bucketsize);
			i = 0;
			bucket++;
			goto next;
		}
		it->phase = PHASE_OLD_BUCKETS;
		bucket = 0;
		b = nil;
		goto next;
	case PHASE_OLD_BUCKETS:
		if(h->oldbuckets == nil) {
			it->phase = PHASE_TABLE;
			goto next;
		}
		if(b != nil) {
			k = b->data + h->keysize * i;
			v = b->data + h->keysize * BUCKETSIZE + h->valuesize * i;
			for(; i < BUCKETSIZE; i++, k += h->keysize, v += h->valuesize) {
				if(b->tophash[i] != 0) {
					data->key_data = k;
					data->val_data = v;
					it->bucket = bucket;
					it->b = b;
					it->i = i + 1;
					return true;
				}
			}
			b = overflowptr(b);
			if(b != nil) {
				data->st = (byte*)b;
				it->bucket = bucket;
				it->b = b;
				it->i = 0;
				return true;
			}
		}
		if(bucket < ((uintptr)1 << (h->B - 1))) {
			b = (Bucket*)(h->oldbuckets + bucket * h->bucketsize);
			bucket++;
			i = 0;
			goto next;
		}
		it->phase = PHASE_TABLE;
		goto next;
	case PHASE_TABLE:
		it->phase = PHASE_OLD_TABLE;
		data->st = h->buckets;
		return true;
	case PHASE_OLD_TABLE:
		it->phase = PHASE_DONE;
		if(h->oldbuckets != nil) {
			data->st = h->oldbuckets;
			return true;
		} else {
			goto next;
		}
	}
	if(it->phase != PHASE_DONE)
		runtime·throw("bad phase at done");
	return false;
}

//
/// interfaces to go runtime
//

void
reflect·ismapkey(Type *typ, bool ret)
{
	ret = typ != nil && typ->alg->hash != runtime·nohash;
	FLUSH(&ret);
}

Hmap*
runtime·makemap_c(MapType *typ, int64 hint)
{
	Hmap *h;
	Type *key;

	key = typ->key;

	if(hint < 0 || (int32)hint != hint)
		runtime·panicstring("makemap: size out of range");

	if(key->alg->hash == runtime·nohash)
		runtime·throw("runtime.makemap: unsupported map key type");

	h = runtime·mal(sizeof(*h));

	if(UseSpanType) {
		if(false) {
			runtime·printf("makemap %S: %p\n", *typ->string, h);
		}
		runtime·settype(h, (uintptr)typ | TypeInfo_Map);
	}

	hash_init(typ, h, hint);

	// these calculations are compiler dependent.
	// figure out offsets of map call arguments.

	if(debug) {
		runtime·printf("makemap: map=%p; keysize=%p; valsize=%p; keyalg=%p; valalg=%p\n",
			       h, key->size, typ->elem->size, key->alg, typ->elem->alg);
	}

	return h;
}

// makemap(key, val *Type, hint int64) (hmap *map[any]any);
void
runtime·makemap(MapType *typ, int64 hint, Hmap *ret)
{
	ret = runtime·makemap_c(typ, hint);
	FLUSH(&ret);
}

// For reflect:
//	func makemap(Type *mapType) (hmap *map)
void
reflect·makemap(MapType *t, Hmap *ret)
{
	ret = runtime·makemap_c(t, 0);
	FLUSH(&ret);
}

void
runtime·mapaccess(MapType *t, Hmap *h, byte *ak, byte *av, bool *pres)
{
	byte *res;
	Type *elem;

	elem = t->elem;
	if(h == nil || h->count == 0) {
		elem->alg->copy(elem->size, av, nil);
		*pres = false;
		return;
	}

	if(runtime·gcwaiting)
		runtime·gosched();

	res = hash_lookup(t, h, &ak);

	if(res != nil) {
		*pres = true;
		elem->alg->copy(elem->size, av, res);
	} else {
		*pres = false;
		elem->alg->copy(elem->size, av, nil);
	}
}

// mapaccess1(hmap *map[any]any, key any) (val any);
#pragma textflag 7
void
runtime·mapaccess1(MapType *t, Hmap *h, ...)
{
	byte *ak, *av;
	byte *res;

	if(raceenabled && h != nil)
		runtime·racereadpc(h, runtime·getcallerpc(&t), runtime·mapaccess1);

	ak = (byte*)(&h + 1);
	av = ak + ROUND(t->key->size, Structrnd);

	if(h == nil || h->count == 0) {
		t->elem->alg->copy(t->elem->size, av, nil);
	} else {
		res = hash_lookup(t, h, &ak);
		t->elem->alg->copy(t->elem->size, av, res);
	}

	if(debug) {
		runtime·prints("runtime.mapaccess1: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, ak);
		runtime·prints("; val=");
		t->elem->alg->print(t->elem->size, av);
		runtime·prints("\n");
	}
}

// mapaccess2(hmap *map[any]any, key any) (val any, pres bool);
#pragma textflag 7
void
runtime·mapaccess2(MapType *t, Hmap *h, ...)
{
	byte *ak, *av, *ap;

	if(raceenabled && h != nil)
		runtime·racereadpc(h, runtime·getcallerpc(&t), runtime·mapaccess2);

	ak = (byte*)(&h + 1);
	av = ak + ROUND(t->key->size, Structrnd);
	ap = av + t->elem->size;

	runtime·mapaccess(t, h, ak, av, ap);

	if(debug) {
		runtime·prints("runtime.mapaccess2: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, ak);
		runtime·prints("; val=");
		t->elem->alg->print(t->elem->size, av);
		runtime·prints("; pres=");
		runtime·printbool(*ap);
		runtime·prints("\n");
	}
}

// For reflect:
//	func mapaccess(t type, h map, key iword) (val iword, pres bool)
// where an iword is the same word an interface value would use:
// the actual data if it fits, or else a pointer to the data.
void
reflect·mapaccess(MapType *t, Hmap *h, uintptr key, uintptr val, bool pres)
{
	byte *ak, *av;

	if(raceenabled && h != nil)
		runtime·racereadpc(h, runtime·getcallerpc(&t), reflect·mapaccess);

	if(t->key->size <= sizeof(key))
		ak = (byte*)&key;
	else
		ak = (byte*)key;
	val = 0;
	pres = false;
	if(t->elem->size <= sizeof(val))
		av = (byte*)&val;
	else {
		av = runtime·mal(t->elem->size);
		val = (uintptr)av;
	}
	runtime·mapaccess(t, h, ak, av, &pres);
	FLUSH(&val);
	FLUSH(&pres);
}

void
runtime·mapassign(MapType *t, Hmap *h, byte *ak, byte *av)
{
	if(h == nil)
		runtime·panicstring("assignment to entry in nil map");

	if(runtime·gcwaiting)
		runtime·gosched();

	if(av == nil) {
		hash_remove(t, h, ak);
	} else {
		hash_insert(t, h, ak, av);
	}

	if(debug) {
		runtime·prints("mapassign: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, ak);
		runtime·prints("; val=");
		t->elem->alg->print(t->elem->size, av);
		runtime·prints("\n");
	}
}

// mapassign1(mapType *type, hmap *map[any]any, key any, val any);
#pragma textflag 7
void
runtime·mapassign1(MapType *t, Hmap *h, ...)
{
	byte *ak, *av;

	if(h == nil)
		runtime·panicstring("assignment to entry in nil map");

	if(raceenabled)
		runtime·racewritepc(h, runtime·getcallerpc(&t), runtime·mapassign1);
	ak = (byte*)(&h + 1);
	av = ak + ROUND(t->key->size, t->elem->align);

	runtime·mapassign(t, h, ak, av);
}

// mapdelete(mapType *type, hmap *map[any]any, key any)
#pragma textflag 7
void
runtime·mapdelete(MapType *t, Hmap *h, ...)
{
	byte *ak;

	if(h == nil)
		return;

	if(raceenabled)
		runtime·racewritepc(h, runtime·getcallerpc(&t), runtime·mapdelete);
	ak = (byte*)(&h + 1);
	runtime·mapassign(t, h, ak, nil);

	if(debug) {
		runtime·prints("mapdelete: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, ak);
		runtime·prints("\n");
	}
}

// For reflect:
//	func mapassign(t type h map, key, val iword, pres bool)
// where an iword is the same word an interface value would use:
// the actual data if it fits, or else a pointer to the data.
void
reflect·mapassign(MapType *t, Hmap *h, uintptr key, uintptr val, bool pres)
{
	byte *ak, *av;

	if(h == nil)
		runtime·panicstring("assignment to entry in nil map");
	if(raceenabled)
		runtime·racewritepc(h, runtime·getcallerpc(&t), reflect·mapassign);
	if(t->key->size <= sizeof(key))
		ak = (byte*)&key;
	else
		ak = (byte*)key;
	if(t->elem->size <= sizeof(val))
		av = (byte*)&val;
	else
		av = (byte*)val;
	if(!pres)
		av = nil;
	runtime·mapassign(t, h, ak, av);
}

// mapiterinit(mapType *type, hmap *map[any]any, hiter *any);
void
runtime·mapiterinit(MapType *t, Hmap *h, struct hash_iter *it)
{
	if(h == nil) {
		it->key = nil;
		return;
	}
	if(raceenabled)
		runtime·racereadpc(h, runtime·getcallerpc(&t), runtime·mapiterinit);
	hash_iter_init(t, h, it);
	hash_next(it);
	if(debug) {
		runtime·prints("runtime.mapiterinit: map=");
		runtime·printpointer(h);
		runtime·prints("; iter=");
		runtime·printpointer(it);
		runtime·prints("; key=");
		runtime·printpointer(it->key);
		runtime·prints("\n");
	}
}

// For reflect:
//	func mapiterinit(h map) (it iter)
void
reflect·mapiterinit(MapType *t, Hmap *h, struct hash_iter *it)
{
	uint32 old, new;

	if(h != nil && t->key->size > sizeof(void*)) {
		// reflect·mapiterkey returns pointers to key data,
		// and reflect holds them, so we cannot free key data
		// eagerly anymore.
		// Can run concurrently with another reflect·mapiterinit() and with hash_iter_init().
		for(;;) {
			old = h->flags;
			if(old & IndirectKey)
				new = old & ~CanFreeKey;
			else
				new = old & ~CanFreeBucket;
			if(new == old)
				break;
			if(runtime·cas(&h->flags, old, new))
				break;
		}
	}

	it = runtime·mal(sizeof *it);
	FLUSH(&it);
	runtime·mapiterinit(t, h, it);
}

// mapiternext(hiter *any);
void
runtime·mapiternext(struct hash_iter *it)
{
	if(raceenabled)
		runtime·racereadpc(it->h, runtime·getcallerpc(&it), runtime·mapiternext);
	if(runtime·gcwaiting)
		runtime·gosched();

	hash_next(it);
	if(debug) {
		runtime·prints("runtime.mapiternext: iter=");
		runtime·printpointer(it);
		runtime·prints("; key=");
		runtime·printpointer(it->key);
		runtime·prints("\n");
	}
}

// For reflect:
//	func mapiternext(it iter)
void
reflect·mapiternext(struct hash_iter *it)
{
	runtime·mapiternext(it);
}

// mapiter1(hiter *any) (key any);
#pragma textflag 7
void
runtime·mapiter1(struct hash_iter *it, ...)
{
	byte *ak, *res;
	Type *key;

	ak = (byte*)(&it + 1);

	res = it->key;
	if(res == nil)
		runtime·throw("runtime.mapiter1: key:val nil pointer");

	key = it->t->key;
	key->alg->copy(key->size, ak, res);

	if(debug) {
		runtime·prints("mapiter1: iter=");
		runtime·printpointer(it);
		runtime·prints("; map=");
		runtime·printpointer(it->h);
		runtime·prints("\n");
	}
}

bool
runtime·mapiterkey(struct hash_iter *it, void *ak)
{
	byte *res;
	Type *key;

	res = it->key;
	if(res == nil)
		return false;
	key = it->t->key;
	key->alg->copy(key->size, ak, res);
	return true;
}

// For reflect:
//	func mapiterkey(h map) (key iword, ok bool)
// where an iword is the same word an interface value would use:
// the actual data if it fits, or else a pointer to the data.
void
reflect·mapiterkey(struct hash_iter *it, uintptr key, bool ok)
{
	byte *res;
	Type *tkey;

	key = 0;
	ok = false;
	res = it->key;
	if(res == nil) {
		key = 0;
		ok = false;
	} else {
		tkey = it->t->key;
		key = 0;
		if(tkey->size <= sizeof(key))
			tkey->alg->copy(tkey->size, (byte*)&key, res);
		else
			key = (uintptr)res;
		ok = true;
	}
	FLUSH(&key);
	FLUSH(&ok);
}

// For reflect:
//	func maplen(h map) (len int)
// Like len(m) in the actual language, we treat the nil map as length 0.
void
reflect·maplen(Hmap *h, intgo len)
{
	if(h == nil)
		len = 0;
	else {
		len = h->count;
		if(raceenabled)
			runtime·racereadpc(h, runtime·getcallerpc(&h), reflect·maplen);
	}
	FLUSH(&len);
}

// mapiter2(hiter *any) (key any, val any);
#pragma textflag 7
void
runtime·mapiter2(struct hash_iter *it, ...)
{
	byte *ak, *av, *res;
	MapType *t;

	t = it->t;
	ak = (byte*)(&it + 1);
	av = ak + ROUND(t->key->size, t->elem->align);

	res = it->key;
	if(res == nil)
		runtime·throw("runtime.mapiter2: key:val nil pointer");

	t->key->alg->copy(t->key->size, ak, res);
	t->elem->alg->copy(t->elem->size, av, it->value);

	if(debug) {
		runtime·prints("mapiter2: iter=");
		runtime·printpointer(it);
		runtime·prints("; map=");
		runtime·printpointer(it->h);
		runtime·prints("\n");
	}
}
