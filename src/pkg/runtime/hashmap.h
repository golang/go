// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

struct Hmap;		/* opaque */

/* Used by the garbage collector */
struct hash_gciter
{
	Hmap *h;
	int32 phase;
	uintptr bucket;
	struct Bucket *b;
	uintptr i;
};

// this data is used by the garbage collector to keep the map's
// internal structures from being reclaimed.  The iterator must
// return in st every live object (ones returned by mallocgc) so
// that those objects won't be collected, and it must return
// every key & value in key_data/val_data so they can get scanned
// for pointers they point to.  Note that if you malloc storage
// for keys and values, you need to do both.
struct hash_gciter_data
{
	uint8 *st;			/* internal structure, or nil */
	uint8 *key_data;		/* key data, or nil */
	uint8 *val_data;		/* value data, or nil */
	bool indirectkey;		/* storing pointers to keys */
	bool indirectval;		/* storing pointers to values */
};
bool hash_gciter_init (struct Hmap *h, struct hash_gciter *it);
bool hash_gciter_next (struct hash_gciter *it, struct hash_gciter_data *data);
