// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "hashmap.h"
#include "type.h"
#include "race.h"

/* Hmap flag values */
#define IndirectVal  (1<<0)	/* storing pointers to values */
#define IndirectKey (1<<1)	/* storing pointers to keys */
#define CanFreeTable (1<<2)	/* okay to free subtables */
#define CanFreeKey (1<<3)	/* okay to free pointers to keys */

struct Hmap {	   /* a hash table; initialize with hash_init() */
	uintgo count;	  /* elements in table - must be first */
	uint8 datasize;   /* amount of data to store in entry */
	uint8 flag;
	uint8 valoff;	/* offset of value in key+value data block */
	int32 changes;	      /* inc'ed whenever a subtable is created/grown */
	uintptr hash0;      /* hash seed */
	struct hash_subtable *st;    /* first-level table */
};

#define MaxData 255

struct hash_entry {
	hash_hash_t hash;     /* hash value of data */
	byte data[1];	 /* user data has "datasize" bytes */
};

struct hash_subtable {
	uint8 power;	 /* bits used to index this table */
	uint8 used;	  /* bits in hash used before reaching this table */
	uint8 datasize;      /* bytes of client data in an entry */
	uint8 max_probes;    /* max number of probes when searching */
	int16 limit_bytes;	   /* max_probes * (datasize+sizeof (hash_hash_t)) */
	struct hash_entry *last;      /* points to last element of entry[] */
	struct hash_entry entry[1];  /* 2**power+max_probes-1 elements of elemsize bytes */
};

#define HASH_DATA_EQ(eq, t, h,x,y) ((eq)=0, (*t->key->alg->equal) (&(eq), t->key->size, (x), (y)), (eq))

#define HASH_REHASH 0x2       /* an internal flag */
/* the number of bits used is stored in the flags word too */
#define HASH_USED(x)      ((x) >> 2)
#define HASH_MAKE_USED(x) ((x) << 2)

#define HASH_LOW	6
#define HASH_ONE	(((hash_hash_t)1) << HASH_LOW)
#define HASH_MASK       (HASH_ONE - 1)
#define HASH_ADJUST(x)  (((x) < HASH_ONE) << HASH_LOW)

#define HASH_BITS       (sizeof (hash_hash_t) * 8)

#define HASH_SUBHASH    HASH_MASK
#define HASH_NIL	0
#define HASH_NIL_MEMSET 0

#define HASH_OFFSET(base, byte_offset) \
	  ((struct hash_entry *) (((byte *) (base)) + (byte_offset)))

#define HASH_MAX_PROBES	15 /* max entries to probe before rehashing */
#define HASH_MAX_POWER	12 /* max power of 2 to create sub-tables */

/* return a hash layer with 2**power empty entries */
static struct hash_subtable *
hash_subtable_new (Hmap *h, int32 power, int32 used)
{
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	int32 bytes = elemsize << power;
	struct hash_subtable *st;
	int32 limit_bytes = HASH_MAX_PROBES * elemsize;
	int32 max_probes = HASH_MAX_PROBES;

	if (bytes < limit_bytes) {
		limit_bytes = bytes;
		max_probes = 1 << power;
	}
	bytes += limit_bytes - elemsize;
	st = malloc (offsetof (struct hash_subtable, entry[0]) + bytes);
	st->power = power;
	st->used = used;
	st->datasize = h->datasize;
	st->max_probes = max_probes;
	st->limit_bytes = limit_bytes;
	st->last = HASH_OFFSET (st->entry, bytes) - 1;
	memset (st->entry, HASH_NIL_MEMSET, bytes);
	return (st);
}

static void
init_sizes (int64 hint, int32 *init_power)
{
	int32 log = 0;
	int32 i;

	for (i = 32; i != 0; i >>= 1) {
		if ((hint >> (log + i)) != 0) {
			log += i;
		}
	}
	log += 1 + (((hint << 3) >> log) >= 11);  /* round up for utilization */
	if (log <= 14) {
		*init_power = log;
	} else {
		*init_power = 12;
	}
}

static void
hash_init (Hmap *h, int32 datasize, int64 hint)
{
	int32 init_power;

	if(datasize < sizeof (void *))
		datasize = sizeof (void *);
	datasize = ROUND(datasize, sizeof (void *));
	init_sizes (hint, &init_power);
	h->datasize = datasize;
	assert (h->datasize == datasize);
	assert (sizeof (void *) <= h->datasize);
	h->count = 0;
	h->changes = 0;
	h->st = hash_subtable_new (h, init_power, 0);
	h->hash0 = runtime·fastrand1();
}

static void
hash_remove_n (struct hash_subtable *st, struct hash_entry *dst_e, int32 n)
{
	int32 elemsize = st->datasize + offsetof (struct hash_entry, data[0]);
	struct hash_entry *src_e = HASH_OFFSET (dst_e, n * elemsize);
	struct hash_entry *last_e = st->last;
	int32 shift = HASH_BITS - (st->power + st->used);
	int32 index_mask = (((hash_hash_t)1) << st->power) - 1;
	int32 dst_i = (((byte *) dst_e) - ((byte *) st->entry)) / elemsize;
	int32 src_i = dst_i + n;
	hash_hash_t hash;
	int32 skip;
	int32 bytes;

	while (dst_e != src_e) {
		if (src_e <= last_e) {
			struct hash_entry *cp_e = src_e;
			int32 save_dst_i = dst_i;
			while (cp_e <= last_e && (hash = cp_e->hash) != HASH_NIL &&
			     ((hash >> shift) & index_mask) <= dst_i) {
				cp_e = HASH_OFFSET (cp_e, elemsize);
				dst_i++;
			}
			bytes = ((byte *) cp_e) - (byte *) src_e;
			memmove (dst_e, src_e, bytes);
			dst_e = HASH_OFFSET (dst_e, bytes);
			src_e = cp_e;
			src_i += dst_i - save_dst_i;
			if (src_e <= last_e && (hash = src_e->hash) != HASH_NIL) {
				skip = ((hash >> shift) & index_mask) - dst_i;
			} else {
				skip = src_i - dst_i;
			}
		} else {
			skip = src_i - dst_i;
		}
		bytes = skip * elemsize;
		memset (dst_e, HASH_NIL_MEMSET, bytes);
		dst_e = HASH_OFFSET (dst_e, bytes);
		dst_i += skip;
	}
}

static int32
hash_insert_internal (MapType*, struct hash_subtable **pst, int32 flags, hash_hash_t hash,
		Hmap *h, void *data, void **pres);

static void
hash_conv (MapType *t, Hmap *h,
		struct hash_subtable *st, int32 flags,
		hash_hash_t hash,
		struct hash_entry *e)
{
	int32 new_flags = (flags + HASH_MAKE_USED (st->power)) | HASH_REHASH;
	int32 shift = HASH_BITS - HASH_USED (new_flags);
	hash_hash_t prefix_mask = (-(hash_hash_t)1) << shift;
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	void *dummy_result;
	struct hash_entry *de;
	int32 index_mask = (1 << st->power) - 1;
	hash_hash_t e_hash;
	struct hash_entry *pe = HASH_OFFSET (e, -elemsize);

	while (e != st->entry && (e_hash = pe->hash) != HASH_NIL && (e_hash & HASH_MASK) != HASH_SUBHASH) {
		e = pe;
		pe = HASH_OFFSET (pe, -elemsize);
	}

	de = e;
	while (e <= st->last &&
	    (e_hash = e->hash) != HASH_NIL &&
	    (e_hash & HASH_MASK) != HASH_SUBHASH) {
		struct hash_entry *target_e = HASH_OFFSET (st->entry, ((e_hash >> shift) & index_mask) * elemsize);
		struct hash_entry *ne = HASH_OFFSET (e, elemsize);
		hash_hash_t current = e_hash & prefix_mask;
		if (de < target_e) {
			memset (de, HASH_NIL_MEMSET, ((byte *) target_e) - (byte *) de);
			de = target_e;
		}
		if ((hash & prefix_mask) == current ||
		   (ne <= st->last && (e_hash = ne->hash) != HASH_NIL &&
		   (e_hash & prefix_mask) == current)) {
			struct hash_subtable *new_st = hash_subtable_new (h, 1, HASH_USED (new_flags));
			int32 rc = hash_insert_internal (t, &new_st, new_flags, e->hash, h, e->data, &dummy_result);
			assert (rc == 0);
			memcpy(dummy_result, e->data, h->datasize);
			e = ne;
			while (e <= st->last && (e_hash = e->hash) != HASH_NIL && (e_hash & prefix_mask) == current) {
				assert ((e_hash & HASH_MASK) != HASH_SUBHASH);
				rc = hash_insert_internal (t, &new_st, new_flags, e_hash, h, e->data, &dummy_result);
				assert (rc == 0);
				memcpy(dummy_result, e->data, h->datasize);
				e = HASH_OFFSET (e, elemsize);
			}
			memset (de->data, HASH_NIL_MEMSET, h->datasize);
			*(struct hash_subtable **)de->data = new_st;
			de->hash = current | HASH_SUBHASH;
		} else {
			if (e != de) {
				memcpy (de, e, elemsize);
			}
			e = HASH_OFFSET (e, elemsize);
		}
		de = HASH_OFFSET (de, elemsize);
	}
	if (e != de) {
		hash_remove_n (st, de, (((byte *) e) - (byte *) de) / elemsize);
	}
}

static void
hash_grow (MapType *t, Hmap *h, struct hash_subtable **pst, int32 flags)
{
	struct hash_subtable *old_st = *pst;
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	*pst = hash_subtable_new (h, old_st->power + 1, HASH_USED (flags));
	struct hash_entry *last_e = old_st->last;
	struct hash_entry *e;
	void *dummy_result;
	int32 used = 0;

	flags |= HASH_REHASH;
	for (e = old_st->entry; e <= last_e; e = HASH_OFFSET (e, elemsize)) {
		hash_hash_t hash = e->hash;
		if (hash != HASH_NIL) {
			int32 rc = hash_insert_internal (t, pst, flags, e->hash, h, e->data, &dummy_result);
			assert (rc == 0);
			memcpy(dummy_result, e->data, h->datasize);
			used++;
		}
	}
	if (h->flag & CanFreeTable)
		free (old_st);
}

static int32
hash_lookup (MapType *t, Hmap *h, void *data, void **pres)
{
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	hash_hash_t hash;
	struct hash_subtable *st = h->st;
	int32 used = 0;
	hash_hash_t e_hash;
	struct hash_entry *e;
	struct hash_entry *end_e;
	void *key;
	bool eq;

	hash = h->hash0;
	(*t->key->alg->hash) (&hash, t->key->size, data);
	hash &= ~HASH_MASK;
	hash += HASH_ADJUST (hash);
	for (;;) {
		int32 shift = HASH_BITS - (st->power + used);
		int32 index_mask = (1 << st->power) - 1;
		int32 i = (hash >> shift) & index_mask;	   /* i is the natural position of hash */

		e = HASH_OFFSET (st->entry, i * elemsize); /* e points to element i */
		e_hash = e->hash;
		if ((e_hash & HASH_MASK) != HASH_SUBHASH) {      /* a subtable */
			break;
		}
		used += st->power;
		st = *(struct hash_subtable **)e->data;
	}
	end_e = HASH_OFFSET (e, st->limit_bytes);
	while (e != end_e && (e_hash = e->hash) != HASH_NIL && e_hash < hash) {
		e = HASH_OFFSET (e, elemsize);
	}
	while (e != end_e && ((e_hash = e->hash) ^ hash) < HASH_SUBHASH) {
		key = e->data;
		if (h->flag & IndirectKey)
			key = *(void**)e->data;
		if (HASH_DATA_EQ (eq, t, h, data, key)) {    /* a match */
			*pres = e->data;
			return (1);
		}
		e = HASH_OFFSET (e, elemsize);
	}
	USED(e_hash);
	*pres = 0;
	return (0);
}

static int32
hash_remove (MapType *t, Hmap *h, void *data)
{
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	hash_hash_t hash;
	struct hash_subtable *st = h->st;
	int32 used = 0;
	hash_hash_t e_hash;
	struct hash_entry *e;
	struct hash_entry *end_e;
	bool eq;
	void *key;

	hash = h->hash0;
	(*t->key->alg->hash) (&hash, t->key->size, data);
	hash &= ~HASH_MASK;
	hash += HASH_ADJUST (hash);
	for (;;) {
		int32 shift = HASH_BITS - (st->power + used);
		int32 index_mask = (1 << st->power) - 1;
		int32 i = (hash >> shift) & index_mask;	   /* i is the natural position of hash */

		e = HASH_OFFSET (st->entry, i * elemsize); /* e points to element i */
		e_hash = e->hash;
		if ((e_hash & HASH_MASK) != HASH_SUBHASH) {      /* a subtable */
			break;
		}
		used += st->power;
		st = *(struct hash_subtable **)e->data;
	}
	end_e = HASH_OFFSET (e, st->limit_bytes);
	while (e != end_e && (e_hash = e->hash) != HASH_NIL && e_hash < hash) {
		e = HASH_OFFSET (e, elemsize);
	}
	while (e != end_e && ((e_hash = e->hash) ^ hash) < HASH_SUBHASH) {
		key = e->data;
		if (h->flag & IndirectKey)
			key = *(void**)e->data;
		if (HASH_DATA_EQ (eq, t, h, data, key)) {    /* a match */
			// Free key if indirect, but only if reflect can't be
			// holding a pointer to it.  Deletions are rare,
			// indirect (large) keys are rare, reflect on maps
			// is rare.  So in the rare, rare, rare case of deleting
			// an indirect key from a map that has been reflected on,
			// we leave the key for garbage collection instead of
			// freeing it here.
			if (h->flag & CanFreeKey)
				free (key);
			if (h->flag & IndirectVal)
				free (*(void**)((byte*)e->data + h->valoff));
			hash_remove_n (st, e, 1);
			h->count--;
			return (1);
		}
		e = HASH_OFFSET (e, elemsize);
	}
	USED(e_hash);
	return (0);
}

static int32
hash_insert_internal (MapType *t, struct hash_subtable **pst, int32 flags, hash_hash_t hash,
				 Hmap *h, void *data, void **pres)
{
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	bool eq;

	if ((flags & HASH_REHASH) == 0) {
		hash += HASH_ADJUST (hash);
		hash &= ~HASH_MASK;
	}
	for (;;) {
		struct hash_subtable *st = *pst;
		int32 shift = HASH_BITS - (st->power + HASH_USED (flags));
		int32 index_mask = (1 << st->power) - 1;
		int32 i = (hash >> shift) & index_mask;	   /* i is the natural position of hash */
		struct hash_entry *start_e =
			HASH_OFFSET (st->entry, i * elemsize);    /* start_e is the pointer to element i */
		struct hash_entry *e = start_e;		   /* e is going to range over [start_e, end_e) */
		struct hash_entry *end_e;
		hash_hash_t e_hash = e->hash;

		if ((e_hash & HASH_MASK) == HASH_SUBHASH) {      /* a subtable */
			pst = (struct hash_subtable **) e->data;
			flags += HASH_MAKE_USED (st->power);
			continue;
		}
		end_e = HASH_OFFSET (start_e, st->limit_bytes);
		while (e != end_e && (e_hash = e->hash) != HASH_NIL && e_hash < hash) {
			e = HASH_OFFSET (e, elemsize);
			i++;
		}
		if (e != end_e && e_hash != HASH_NIL) {
			/* ins_e ranges over the elements that may match */
			struct hash_entry *ins_e = e;
			int32 ins_i = i;
			hash_hash_t ins_e_hash;
			void *key;
			while (ins_e != end_e && ((e_hash = ins_e->hash) ^ hash) < HASH_SUBHASH) {
				key = ins_e->data;
				if (h->flag & IndirectKey)
					key = *(void**)key;
				if (HASH_DATA_EQ (eq, t, h, data, key)) {    /* a match */
					*pres = ins_e->data;
					return (1);
				}
				if (e_hash == hash) {	   /* adjust hash if it collides */
					assert ((flags & HASH_REHASH) == 0);
					hash++;
					if ((hash & HASH_MASK) == HASH_SUBHASH)
						runtime·throw("runtime: map hash collision overflow");
				}
				ins_e = HASH_OFFSET (ins_e, elemsize);
				ins_i++;
				if (e_hash <= hash) {	       /* set e to insertion point */
					e = ins_e;
					i = ins_i;
				}
			}
			/* set ins_e to the insertion point for the new element */
			ins_e = e;
			ins_i = i;
			ins_e_hash = 0;
			/* move ins_e to point at the end of the contiguous block, but
			   stop if any element can't be moved by one up */
			while (ins_e <= st->last && (ins_e_hash = ins_e->hash) != HASH_NIL &&
			       ins_i + 1 - ((ins_e_hash >> shift) & index_mask) < st->max_probes &&
			       (ins_e_hash & HASH_MASK) != HASH_SUBHASH) {
				ins_e = HASH_OFFSET (ins_e, elemsize);
				ins_i++;
			}
			if (e == end_e || ins_e > st->last || ins_e_hash != HASH_NIL) {
				e = end_e;    /* can't insert; must grow or convert to subtable */
			} else {	      /* make space for element */
				memmove (HASH_OFFSET (e, elemsize), e, ((byte *) ins_e) - (byte *) e);
			}
		}
		if (e != end_e) {
			e->hash = hash;
			*pres = e->data;
			return (0);
		}
		h->changes++;
		if (st->power < HASH_MAX_POWER) {
			hash_grow (t, h, pst, flags);
		} else {
			hash_conv (t, h, st, flags, hash, start_e);
		}
	}
}

static int32
hash_insert (MapType *t, Hmap *h, void *data, void **pres)
{
	uintptr hash;
	int32 rc;

	hash = h->hash0;
	(*t->key->alg->hash) (&hash, t->key->size, data);
	rc = hash_insert_internal (t, &h->st, 0, hash, h, data, pres);

	h->count += (rc == 0);    /* increment count if element didn't previously exist */
	return (rc);
}

static uint32
hash_count (Hmap *h)
{
	return (h->count);
}

static void
iter_restart (struct hash_iter *it, struct hash_subtable *st, int32 used)
{
	int32 elemsize = it->elemsize;
	hash_hash_t last_hash = it->last_hash;
	struct hash_entry *e;
	hash_hash_t e_hash;
	struct hash_iter_sub *sub = &it->subtable_state[it->i];
	struct hash_entry *last;

	for (;;) {
		int32 shift = HASH_BITS - (st->power + used);
		int32 index_mask = (1 << st->power) - 1;
		int32 i = (last_hash >> shift) & index_mask;

		last = st->last;
		e = HASH_OFFSET (st->entry, i * elemsize);
		sub->start = st->entry;
		sub->last = last;

		if ((e->hash & HASH_MASK) != HASH_SUBHASH) {
			break;
		}
		sub->e = HASH_OFFSET (e, elemsize);
		sub = &it->subtable_state[++(it->i)];
		used += st->power;
		st = *(struct hash_subtable **)e->data;
	}
	while (e <= last && ((e_hash = e->hash) == HASH_NIL || e_hash <= last_hash)) {
		e = HASH_OFFSET (e, elemsize);
	}
	sub->e = e;
}

static void *
hash_next (struct hash_iter *it)
{
	int32 elemsize;
	struct hash_iter_sub *sub;
	struct hash_entry *e;
	struct hash_entry *last;
	hash_hash_t e_hash;

	if (it->changes != it->h->changes) {    /* hash table's structure changed; recompute */
		if (~it->last_hash == 0)
			return (0);
		it->changes = it->h->changes;
		it->i = 0;
		iter_restart (it, it->h->st, 0);
	}
	elemsize = it->elemsize;

Again:
	e_hash = 0;
	sub = &it->subtable_state[it->i];
	e = sub->e;
	last = sub->last;

	if (e != sub->start && it->last_hash != HASH_OFFSET (e, -elemsize)->hash) {
		struct hash_entry *start = HASH_OFFSET (e, -(elemsize * HASH_MAX_PROBES));
		struct hash_entry *pe = HASH_OFFSET (e, -elemsize);
		hash_hash_t last_hash = it->last_hash;
		if (start < sub->start) {
			start = sub->start;
		}
		while (e != start && ((e_hash = pe->hash) == HASH_NIL || last_hash < e_hash)) {
			e = pe;
			pe = HASH_OFFSET (pe, -elemsize);
		}
		while (e <= last && ((e_hash = e->hash) == HASH_NIL || e_hash <= last_hash)) {
			e = HASH_OFFSET (e, elemsize);
		}
	}

	for (;;) {
		while (e <= last && (e_hash = e->hash) == HASH_NIL) {
			e = HASH_OFFSET (e, elemsize);
		}
		if (e > last) {
			if (it->i == 0) {
				if(!it->cycled) {
					// Wrap to zero and iterate up until it->cycle.
					it->cycled = true;
					it->last_hash = 0;
					it->subtable_state[0].e = it->h->st->entry;
					it->subtable_state[0].start = it->h->st->entry;
					it->subtable_state[0].last = it->h->st->last;
					goto Again;
				}
				// Set last_hash to impossible value and
				// break it->changes, so that check at top of
				// hash_next will be used if we get called again.
				it->last_hash = ~(uintptr_t)0;
				it->changes--;
				return (0);
			} else {
				it->i--;
				sub = &it->subtable_state[it->i];
				e = sub->e;
				last = sub->last;
			}
		} else if ((e_hash & HASH_MASK) != HASH_SUBHASH) {
			if(it->cycled && e->hash > it->cycle) {
				// Already returned this.
				// Set last_hash to impossible value and
				// break it->changes, so that check at top of
				// hash_next will be used if we get called again.
				it->last_hash = ~(uintptr_t)0;
				it->changes--;
				return (0);
			}
			it->last_hash = e->hash;
			sub->e = HASH_OFFSET (e, elemsize);
			return (e->data);
		} else {
			struct hash_subtable *st =
				*(struct hash_subtable **)e->data;
			sub->e = HASH_OFFSET (e, elemsize);
			it->i++;
			assert (it->i < sizeof (it->subtable_state) /
					sizeof (it->subtable_state[0]));
			sub = &it->subtable_state[it->i];
			sub->e = e = st->entry;
			sub->start = st->entry;
			sub->last = last = st->last;
		}
	}
}

static void
hash_iter_init (MapType *t, Hmap *h, struct hash_iter *it)
{
	it->elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	it->changes = h->changes;
	it->i = 0;
	it->h = h;
	it->t = t;
	it->last_hash = 0;
	it->subtable_state[0].e = h->st->entry;
	it->subtable_state[0].start = h->st->entry;
	it->subtable_state[0].last = h->st->last;

	// fastrand1 returns 31 useful bits.
	// We don't care about not having a bottom bit but we
	// do want top bits.
	if(sizeof(void*) == 8)
		it->cycle = (uint64)runtime·fastrand1()<<33 | (uint64)runtime·fastrand1()<<2;
	else
		it->cycle = runtime·fastrand1()<<1;
	it->cycled = false;
	it->last_hash = it->cycle;
	iter_restart(it, it->h->st, 0);
}

static void
clean_st (Hmap *h, struct hash_subtable *st, int32 *slots, int32 *used)
{
	int32 elemsize = st->datasize + offsetof (struct hash_entry, data[0]);
	struct hash_entry *e = st->entry;
	struct hash_entry *last = st->last;
	int32 lslots = (((byte *) (last+1)) - (byte *) e) / elemsize;
	int32 lused = 0;

	while (e <= last) {
		hash_hash_t hash = e->hash;
		if ((hash & HASH_MASK) == HASH_SUBHASH) {
			clean_st (h, *(struct hash_subtable **)e->data, slots, used);
		} else {
			lused += (hash != HASH_NIL);
		}
		e = HASH_OFFSET (e, elemsize);
	}
	if (h->flag & CanFreeTable)
		free (st);
	*slots += lslots;
	*used += lused;
}

static void
hash_destroy (Hmap *h)
{
	int32 slots = 0;
	int32 used = 0;

	clean_st (h, h->st, &slots, &used);
	free (h);
}

static void
hash_visit_internal (struct hash_subtable *st,
		int32 used, int32 level,
		void (*data_visit) (void *arg, int32 level, void *data),
		void *arg)
{
	int32 elemsize = st->datasize + offsetof (struct hash_entry, data[0]);
	struct hash_entry *e = st->entry;
	int32 shift = HASH_BITS - (used + st->power);
	int32 i = 0;

	while (e <= st->last) {
		int32 index = ((e->hash >> (shift - 1)) >> 1) & ((1 << st->power) - 1);
		if ((e->hash & HASH_MASK) == HASH_SUBHASH) {
			  (*data_visit) (arg, level, e->data);
			  hash_visit_internal (*(struct hash_subtable **)e->data,
				used + st->power, level + 1, data_visit, arg);
		} else {
			  (*data_visit) (arg, level, e->data);
		}
		if (e->hash != HASH_NIL) {
			  assert (i < index + st->max_probes);
			  assert (index <= i);
		}
		e = HASH_OFFSET (e, elemsize);
		i++;
	}
}

static void
hash_visit (Hmap *h, void (*data_visit) (void *arg, int32 level, void *data), void *arg)
{
	hash_visit_internal (h->st, 0, 0, data_visit, arg);
}

//
/// interfaces to go runtime
//

static void**
hash_valptr(Hmap *h, void *p)
{
	p = (byte*)p + h->valoff;
	if(h->flag & IndirectVal)
		p = *(void**)p;
	return p;
}


static void**
hash_keyptr(Hmap *h, void *p)
{
	if(h->flag & IndirectKey)
		p = *(void**)p;
	return p;
}

static	int32	debug	= 0;

Hmap*
runtime·makemap_c(MapType *typ, int64 hint)
{
	Hmap *h;
	Type *key, *val;
	uintptr ksize, vsize;

	key = typ->key;
	val = typ->elem;

	if(hint < 0 || (int32)hint != hint)
		runtime·panicstring("makemap: size out of range");

	if(key->alg->hash == runtime·nohash)
		runtime·throw("runtime.makemap: unsupported map key type");

	h = runtime·mal(sizeof(*h));
	h->flag |= CanFreeTable;  /* until reflect gets involved, free is okay */

	if(UseSpanType) {
		if(false) {
			runtime·printf("makemap %S: %p\n", *typ->string, h);
		}
		runtime·settype(h, (uintptr)typ | TypeInfo_Map);
	}

	ksize = ROUND(key->size, sizeof(void*));
	vsize = ROUND(val->size, sizeof(void*));
	if(ksize > MaxData || vsize > MaxData || ksize+vsize > MaxData) {
		// Either key is too big, or value is, or combined they are.
		// Prefer to keep the key if possible, because we look at
		// keys more often than values.
		if(ksize > MaxData - sizeof(void*)) {
			// No choice but to indirect the key.
			h->flag |= IndirectKey;
			h->flag |= CanFreeKey;  /* until reflect gets involved, free is okay */
			ksize = sizeof(void*);
		}
		if(vsize > MaxData - ksize) {
			// Have to indirect the value.
			h->flag |= IndirectVal;
			vsize = sizeof(void*);
		}
	}

	h->valoff = ksize;
	hash_init(h, ksize+vsize, hint);

	// these calculations are compiler dependent.
	// figure out offsets of map call arguments.

	if(debug) {
		runtime·printf("makemap: map=%p; keysize=%p; valsize=%p; keyalg=%p; valalg=%p\n",
			h, key->size, val->size, key->alg, val->alg);
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
	if(h == nil) {
		elem->alg->copy(elem->size, av, nil);
		*pres = false;
		return;
	}

	if(runtime·gcwaiting)
		runtime·gosched();

	res = nil;
	if(hash_lookup(t, h, ak, (void**)&res)) {
		*pres = true;
		elem->alg->copy(elem->size, av, hash_valptr(h, res));
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
	bool pres;

	if(raceenabled && h != nil)
		runtime·racereadpc(h, runtime·getcallerpc(&t), runtime·mapaccess1);

	ak = (byte*)(&h + 1);
	av = ak + ROUND(t->key->size, Structrnd);

	runtime·mapaccess(t, h, ak, av, &pres);

	if(debug) {
		runtime·prints("runtime.mapaccess1: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, ak);
		runtime·prints("; val=");
		t->elem->alg->print(t->elem->size, av);
		runtime·prints("; pres=");
		runtime·printbool(pres);
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
		t->elem->alg->print(t->key->size, av);
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
	byte *res;
	int32 hit;

	if(h == nil)
		runtime·panicstring("assignment to entry in nil map");

	if(runtime·gcwaiting)
		runtime·gosched();

	if(av == nil) {
		hash_remove(t, h, ak);
		return;
	}

	res = nil;
	hit = hash_insert(t, h, ak, (void**)&res);
	if(!hit) {
		if(h->flag & IndirectKey)
			*(void**)res = runtime·mal(t->key->size);
		if(h->flag & IndirectVal)
			*(void**)(res+h->valoff) = runtime·mal(t->elem->size);
	}
	t->key->alg->copy(t->key->size, hash_keyptr(h, res), ak);
	t->elem->alg->copy(t->elem->size, hash_valptr(h, res), av);

	if(debug) {
		runtime·prints("mapassign: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		t->key->alg->print(t->key->size, ak);
		runtime·prints("; val=");
		t->elem->alg->print(t->elem->size, av);
		runtime·prints("; hit=");
		runtime·printint(hit);
		runtime·prints("; res=");
		runtime·printpointer(res);
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
		it->data = nil;
		return;
	}
	if(raceenabled)
		runtime·racereadpc(h, runtime·getcallerpc(&t), runtime·mapiterinit);
	hash_iter_init(t, h, it);
	it->data = hash_next(it);
	if(debug) {
		runtime·prints("runtime.mapiterinit: map=");
		runtime·printpointer(h);
		runtime·prints("; iter=");
		runtime·printpointer(it);
		runtime·prints("; data=");
		runtime·printpointer(it->data);
		runtime·prints("\n");
	}
}

// For reflect:
//	func mapiterinit(h map) (it iter)
void
reflect·mapiterinit(MapType *t, Hmap *h, struct hash_iter *it)
{
	uint8 flag;

	if(h != nil && t->key->size > sizeof(void*)) {
		// reflect·mapiterkey returns pointers to key data,
		// and reflect holds them, so we cannot free key data
		// eagerly anymore.  Updating h->flag now is racy,
		// but it's okay because this is the only possible store
		// after creation.
		flag = h->flag;
		if(flag & IndirectKey)
			flag &= ~CanFreeKey;
		else
			flag &= ~CanFreeTable;
		h->flag = flag;
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

	it->data = hash_next(it);
	if(debug) {
		runtime·prints("runtime.mapiternext: iter=");
		runtime·printpointer(it);
		runtime·prints("; data=");
		runtime·printpointer(it->data);
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
	Hmap *h;
	byte *ak, *res;
	Type *key;

	h = it->h;
	ak = (byte*)(&it + 1);

	res = it->data;
	if(res == nil)
		runtime·throw("runtime.mapiter1: key:val nil pointer");

	key = it->t->key;
	key->alg->copy(key->size, ak, hash_keyptr(h, res));

	if(debug) {
		runtime·prints("mapiter2: iter=");
		runtime·printpointer(it);
		runtime·prints("; map=");
		runtime·printpointer(h);
		runtime·prints("\n");
	}
}

bool
runtime·mapiterkey(struct hash_iter *it, void *ak)
{
	byte *res;
	Type *key;

	res = it->data;
	if(res == nil)
		return false;
	key = it->t->key;
	key->alg->copy(key->size, ak, hash_keyptr(it->h, res));
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
	res = it->data;
	if(res == nil) {
		key = 0;
		ok = false;
	} else {
		tkey = it->t->key;
		key = 0;
		res = (byte*)hash_keyptr(it->h, res);
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
	Hmap *h;
	byte *ak, *av, *res;
	MapType *t;

	t = it->t;
	ak = (byte*)(&it + 1);
	av = ak + ROUND(t->key->size, t->elem->align);

	res = it->data;
	if(res == nil)
		runtime·throw("runtime.mapiter2: key:val nil pointer");

	h = it->h;
	t->key->alg->copy(t->key->size, ak, hash_keyptr(h, res));
	t->elem->alg->copy(t->elem->size, av, hash_valptr(h, res));

	if(debug) {
		runtime·prints("mapiter2: iter=");
		runtime·printpointer(it);
		runtime·prints("; map=");
		runtime·printpointer(h);
		runtime·prints("\n");
	}
}
