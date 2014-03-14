// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fast hashmap lookup specialized to a specific key type.
// Included by hashmap.c once for each specialized type.

// +build ignore

// Because this file is #included, it cannot be processed by goc2c,
// so we have to handle the Go resuts ourselves.

#pragma textflag NOSPLIT
void
HASH_LOOKUP1(MapType *t, Hmap *h, KEYTYPE key, GoOutput base, ...)
{
	uintptr bucket, i;
	Bucket *b;
	KEYTYPE *k;
	byte *v, **valueptr;
	uint8 top;
	int8 keymaybe;

	valueptr = (byte**)&base;
	if(debug) {
		runtime·prints("runtime.mapaccess1_fastXXX: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, &key);
		runtime·prints("\n");
	}
	if(h == nil || h->count == 0) {
		*valueptr = t->elem->zero;
		return;
	}
	if(raceenabled)
		runtime·racereadpc(h, runtime·getcallerpc(&t), HASH_LOOKUP1);
	if(docheck)
		check(t, h);

	if(h->B == 0) {
		// One-bucket table. Don't hash, just check each bucket entry.
		b = (Bucket*)h->buckets;
		if(FASTKEY(key)) {
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == Empty)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k) || SLOW_EQ(key, *k)) {
					*valueptr = v;
					return;
				}
			}
		} else {
			keymaybe = -1;
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == Empty)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k)) {
					*valueptr = v;
					return;
				}
				if(MAYBE_EQ(key, *k)) {
					if(keymaybe >= 0) {
						// Two same-length strings in this bucket.
						// use slow path.
						// TODO: keep track of more than just 1.  We could
						// afford about 3 equals calls before it would be more
						// expensive than 1 hash + 1 equals.
						goto dohash;
					}
					keymaybe = i;
				}
			}
			if(keymaybe >= 0) {
				k = (KEYTYPE*)b->data + keymaybe;
				if(SLOW_EQ(key, *k)) {
					*valueptr = (byte*)((KEYTYPE*)b->data + BUCKETSIZE) + keymaybe * h->valuesize;
					return;
				}
			}
		}
	} else {
dohash:
		bucket = h->hash0;
		HASHFUNC(&bucket, sizeof(KEYTYPE), &key);
		top = bucket >> (sizeof(uintptr)*8 - 8);
		if(top < MinTopHash)
			top += MinTopHash;
		bucket &= (((uintptr)1 << h->B) - 1);
		if(h->oldbuckets != nil) {
			i = bucket & (((uintptr)1 << (h->B - 1)) - 1);
			b = (Bucket*)(h->oldbuckets + i * h->bucketsize);
			if(evacuated(b)) {
				b = (Bucket*)(h->buckets + bucket * h->bucketsize);
			}
		} else {
			b = (Bucket*)(h->buckets + bucket * h->bucketsize);
		}
		do {
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] != top)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k) || SLOW_EQ(key, *k)) {
					*valueptr = v;
					return;
				}
			}
			b = b->overflow;
		} while(b != nil);
	}
	*valueptr = t->elem->zero;
}

#pragma textflag NOSPLIT
void
HASH_LOOKUP2(MapType *t, Hmap *h, KEYTYPE key, GoOutput base, ...)
{
	uintptr bucket, i;
	Bucket *b;
	KEYTYPE *k;
	byte *v, **valueptr;
	uint8 top;
	int8 keymaybe;
	bool *okptr;

	valueptr = (byte**)&base;
	okptr = (bool*)(valueptr+1);
	if(debug) {
		runtime·prints("runtime.mapaccess2_fastXXX: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, &key);
		runtime·prints("\n");
	}
	if(h == nil || h->count == 0) {
		*valueptr = t->elem->zero;
		*okptr = false;
		return;
	}
	if(raceenabled)
		runtime·racereadpc(h, runtime·getcallerpc(&t), HASH_LOOKUP2);
	if(docheck)
		check(t, h);

	if(h->B == 0) {
		// One-bucket table. Don't hash, just check each bucket entry.
		b = (Bucket*)h->buckets;
		if(FASTKEY(key)) {
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == Empty)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k) || SLOW_EQ(key, *k)) {
					*valueptr = v;
					*okptr = true;
					return;
				}
			}
		} else {
			keymaybe = -1;
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == Empty)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k)) {
					*valueptr = v;
					*okptr = true;
					return;
				}
				if(MAYBE_EQ(key, *k)) {
					if(keymaybe >= 0) {
						// Two same-length strings in this bucket.
						// use slow path.
						// TODO: keep track of more than just 1.  We could
						// afford about 3 equals calls before it would be more
						// expensive than 1 hash + 1 equals.
						goto dohash;
					}
					keymaybe = i;
				}
			}
			if(keymaybe >= 0) {
				k = (KEYTYPE*)b->data + keymaybe;
				if(SLOW_EQ(key, *k)) {
					*valueptr = (byte*)((KEYTYPE*)b->data + BUCKETSIZE) + keymaybe * h->valuesize;
					*okptr = true;
					return;
				}
			}
		}
	} else {
dohash:
		bucket = h->hash0;
		HASHFUNC(&bucket, sizeof(KEYTYPE), &key);
		top = bucket >> (sizeof(uintptr)*8 - 8);
		if(top < MinTopHash)
			top += MinTopHash;
		bucket &= (((uintptr)1 << h->B) - 1);
		if(h->oldbuckets != nil) {
			i = bucket & (((uintptr)1 << (h->B - 1)) - 1);
			b = (Bucket*)(h->oldbuckets + i * h->bucketsize);
			if(evacuated(b)) {
				b = (Bucket*)(h->buckets + bucket * h->bucketsize);
			}
		} else {
			b = (Bucket*)(h->buckets + bucket * h->bucketsize);
		}
		do {
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] != top)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k) || SLOW_EQ(key, *k)) {
					*valueptr = v;
					*okptr = true;
					return;
				}
			}
			b = b->overflow;
		} while(b != nil);
	}
	*valueptr = t->elem->zero;
	*okptr = false;
}
