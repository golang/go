// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fast hashmap lookup specialized to a specific key type.
// Included by hashmap.c once for each specialized type.

// Note that this code differs from hash_lookup in that
// it returns a pointer to the result, not the result itself.
// The returned pointer is only valid until the next GC
// point, so the caller must dereference it before then.

// +build ignore

#pragma textflag 7
void
HASH_LOOKUP1(MapType *t, Hmap *h, KEYTYPE key, byte *value)
{
	uintptr hash;
	uintptr bucket, oldbucket;
	Bucket *b;
	uint8 top;
	uintptr i;
	KEYTYPE *k;
	byte *v;

	if(debug) {
		runtime·prints("runtime.mapaccess1_fastXXX: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, &key);
		runtime·prints("\n");
	}
	if(h == nil || h->count == 0) {
		value = empty_value;
		FLUSH(&value);
		return;
	}
	if(raceenabled)
		runtime·racereadpc(h, runtime·getcallerpc(&t), HASH_LOOKUP1);
	if(docheck)
		check(t, h);

	if(h->B == 0 && (h->count == 1 || QUICKEQ(key))) {
		// One-bucket table.  Don't hash, just check each bucket entry.
		b = (Bucket*)h->buckets;
		for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
			if(b->tophash[i] != 0 && EQFUNC(key, *k)) {
				value = v;
				FLUSH(&value);
				return;
			}
		}
	} else {
		hash = h->hash0;
		HASHFUNC(&hash, sizeof(KEYTYPE), &key);
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
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == top && EQFUNC(key, *k)) {
					value = v;
					FLUSH(&value);
					return;
				}
			}
			b = b->overflow;
		} while(b != nil);
	}
	value = empty_value;
	FLUSH(&value);
}

#pragma textflag 7
void
HASH_LOOKUP2(MapType *t, Hmap *h, KEYTYPE key, byte *value, bool res)
{
	uintptr hash;
	uintptr bucket, oldbucket;
	Bucket *b;
	uint8 top;
	uintptr i;
	KEYTYPE *k;
	byte *v;

	if(debug) {
		runtime·prints("runtime.mapaccess2_fastXXX: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, &key);
		runtime·prints("\n");
	}
	if(h == nil || h->count == 0) {
		value = empty_value;
		res = false;
		FLUSH(&value);
		FLUSH(&res);
		return;
	}
	if(raceenabled)
		runtime·racereadpc(h, runtime·getcallerpc(&t), HASH_LOOKUP2);
	if(docheck)
		check(t, h);

	if(h->B == 0 && (h->count == 1 || QUICKEQ(key))) {
		// One-bucket table.  Don't hash, just check each bucket entry.
		b = (Bucket*)h->buckets;
		for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
			if(b->tophash[i] != 0 && EQFUNC(key, *k)) {
				value = v;
				res = true;
				FLUSH(&value);
				FLUSH(&res);
				return;
			}
		}
	} else {
		hash = h->hash0;
		HASHFUNC(&hash, sizeof(KEYTYPE), &key);
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
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == top && EQFUNC(key, *k)) {
					value = v;
					res = true;
					FLUSH(&value);
					FLUSH(&res);
					return;
				}
			}
			b = b->overflow;
		} while(b != nil);
	}
	value = empty_value;
	res = false;
	FLUSH(&value);
	FLUSH(&res);
}
