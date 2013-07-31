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
	uintptr bucket, i;
	Bucket *b;
	KEYTYPE *k;
	byte *v;
	uint8 top;
	int8 keymaybe;

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

	if(h->B == 0) {
		// One-bucket table. Don't hash, just check each bucket entry.
		b = (Bucket*)h->buckets;
		if(FASTKEY(key)) {
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == 0)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k) || SLOW_EQ(key, *k)) {
					value = v;
					FLUSH(&value);
					return;
				}
			}
		} else {
			keymaybe = -1;
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == 0)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k)) {
					value = v;
					FLUSH(&value);
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
					value = (byte*)((KEYTYPE*)b->data + BUCKETSIZE) + keymaybe * h->valuesize;
					FLUSH(&value);
					return;
				}
			}
		}
	} else {
dohash:
		bucket = h->hash0;
		HASHFUNC(&bucket, sizeof(KEYTYPE), &key);
		top = bucket >> (sizeof(uintptr)*8 - 8);
		if(top == 0)
			top = 1;
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
	uintptr bucket, i;
	Bucket *b;
	KEYTYPE *k;
	byte *v;
	uint8 top;
	int8 keymaybe;

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

	if(h->B == 0) {
		// One-bucket table. Don't hash, just check each bucket entry.
		b = (Bucket*)h->buckets;
		if(FASTKEY(key)) {
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == 0)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k) || SLOW_EQ(key, *k)) {
					value = v;
					res = true;
					FLUSH(&value);
					FLUSH(&res);
					return;
				}
			}
		} else {
			keymaybe = -1;
			for(i = 0, k = (KEYTYPE*)b->data, v = (byte*)(k + BUCKETSIZE); i < BUCKETSIZE; i++, k++, v += h->valuesize) {
				if(b->tophash[i] == 0)
					continue;
				if(QUICK_NE(key, *k))
					continue;
				if(QUICK_EQ(key, *k)) {
					value = v;
					res = true;
					FLUSH(&value);
					FLUSH(&res);
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
					value = (byte*)((KEYTYPE*)b->data + BUCKETSIZE) + keymaybe * h->valuesize;
					res = true;
					FLUSH(&value);
					FLUSH(&res);
					return;
				}
			}
		}
	} else {
dohash:
		bucket = h->hash0;
		HASHFUNC(&bucket, sizeof(KEYTYPE), &key);
		top = bucket >> (sizeof(uintptr)*8 - 8);
		if(top == 0)
			top = 1;
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
