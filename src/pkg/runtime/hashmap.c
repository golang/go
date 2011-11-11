// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "hashmap.h"
#include "type.h"

/* Return a pointer to the struct/union of type "type"
   whose "field" field is addressed by pointer "p". */

struct Hmap {	   /* a hash table; initialize with hash_init() */
	uint32 count;	  /* elements in table - must be first */

	uint8 datasize;   /* amount of data to store in entry */
	uint8 max_power;  /* max power of 2 to create sub-tables */
	uint8 max_probes; /* max entries to probe before rehashing */
	uint8 indirectval; /* storing pointers to values */
	int32 changes;	      /* inc'ed whenever a subtable is created/grown */
	hash_hash_t (*data_hash) (uint32, void *a);  /* return hash of *a */
	uint32 (*data_eq) (uint32, void *a, void *b);   /* return whether *a == *b */
	void (*data_del) (uint32, void *arg, void *data);  /* invoked on deletion */
	struct hash_subtable *st;    /* first-level table */

	uint32	keysize;
	uint32	valsize;
	uint32	datavo;

	// three sets of offsets: the digit counts how many
	// of key, value are passed as inputs:
	//	0 = func() (key, value)
	//	1 = func(key) (value)
	//	2 = func(key, value)
	uint32	ko0;
	uint32	vo0;
	uint32	ko1;
	uint32	vo1;
	uint32	po1;
	uint32	ko2;
	uint32	vo2;
	uint32	po2;
	Alg*	keyalg;
	Alg*	valalg;
};

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

#define HASH_DATA_EQ(h,x,y) ((*h->data_eq) (h->keysize, (x), (y)))

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


/* return a hash layer with 2**power empty entries */
static struct hash_subtable *
hash_subtable_new (Hmap *h, int32 power, int32 used)
{
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	int32 bytes = elemsize << power;
	struct hash_subtable *st;
	int32 limit_bytes = h->max_probes * elemsize;
	int32 max_probes = h->max_probes;

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
init_sizes (int64 hint, int32 *init_power, int32 *max_power)
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
	*max_power = 12;
}

static void
hash_init (Hmap *h,
		int32 datasize,
		hash_hash_t (*data_hash) (uint32, void *),
		uint32 (*data_eq) (uint32, void *, void *),
		void (*data_del) (uint32, void *, void *),
		int64 hint)
{
	int32 init_power;
	int32 max_power;

	if(datasize < sizeof (void *))
		datasize = sizeof (void *);
	datasize = runtime·rnd(datasize, sizeof (void *));
	init_sizes (hint, &init_power, &max_power);
	h->datasize = datasize;
	h->max_power = max_power;
	h->max_probes = 15;
	assert (h->datasize == datasize);
	assert (h->max_power == max_power);
	assert (sizeof (void *) <= h->datasize || h->max_power == 255);
	h->count = 0;
	h->changes = 0;
	h->data_hash = data_hash;
	h->data_eq = data_eq;
	h->data_del = data_del;
	h->st = hash_subtable_new (h, init_power, 0);
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
hash_insert_internal (struct hash_subtable **pst, int32 flags, hash_hash_t hash,
		Hmap *h, void *data, void **pres);

static void
hash_conv (Hmap *h,
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
			int32 rc = hash_insert_internal (&new_st, new_flags, e->hash, h, e->data, &dummy_result);
			assert (rc == 0);
			memcpy(dummy_result, e->data, h->datasize);
			e = ne;
			while (e <= st->last && (e_hash = e->hash) != HASH_NIL && (e_hash & prefix_mask) == current) {
				assert ((e_hash & HASH_MASK) != HASH_SUBHASH);
				rc = hash_insert_internal (&new_st, new_flags, e_hash, h, e->data, &dummy_result);
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
hash_grow (Hmap *h, struct hash_subtable **pst, int32 flags)
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
			int32 rc = hash_insert_internal (pst, flags, e->hash, h, e->data, &dummy_result);
			assert (rc == 0);
			memcpy(dummy_result, e->data, h->datasize);
			used++;
		}
	}
	free (old_st);
}

static int32
hash_lookup (Hmap *h, void *data, void **pres)
{
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	hash_hash_t hash = (*h->data_hash) (h->keysize, data) & ~HASH_MASK;
	struct hash_subtable *st = h->st;
	int32 used = 0;
	hash_hash_t e_hash;
	struct hash_entry *e;
	struct hash_entry *end_e;

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
		if (HASH_DATA_EQ (h, data, e->data)) {    /* a match */
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
hash_remove (Hmap *h, void *data, void *arg)
{
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	hash_hash_t hash = (*h->data_hash) (h->keysize, data) & ~HASH_MASK;
	struct hash_subtable *st = h->st;
	int32 used = 0;
	hash_hash_t e_hash;
	struct hash_entry *e;
	struct hash_entry *end_e;

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
		if (HASH_DATA_EQ (h, data, e->data)) {    /* a match */
			(*h->data_del) (h->datavo, arg, e->data);
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
hash_insert_internal (struct hash_subtable **pst, int32 flags, hash_hash_t hash,
				 Hmap *h, void *data, void **pres)
{
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);

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
			while (ins_e != end_e && ((e_hash = ins_e->hash) ^ hash) < HASH_SUBHASH) {
				if (HASH_DATA_EQ (h, data, ins_e->data)) {    /* a match */
					*pres = ins_e->data;
					return (1);
				}
				assert (e_hash != hash || (flags & HASH_REHASH) == 0);
				hash += (e_hash == hash);	   /* adjust hash if it collides */
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
		if (st->power < h->max_power) {
			hash_grow (h, pst, flags);
		} else {
			hash_conv (h, st, flags, hash, start_e);
		}
	}
}

static int32
hash_insert (Hmap *h, void *data, void **pres)
{
	int32 rc = hash_insert_internal (&h->st, 0, (*h->data_hash) (h->keysize, data), h, data, pres);

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
		struct hash_entry *start = HASH_OFFSET (e, -(elemsize * it->h->max_probes));
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
hash_iter_init (Hmap *h, struct hash_iter *it)
{
	it->elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	it->changes = h->changes;
	it->i = 0;
	it->h = h;
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
clean_st (struct hash_subtable *st, int32 *slots, int32 *used)
{
	int32 elemsize = st->datasize + offsetof (struct hash_entry, data[0]);
	struct hash_entry *e = st->entry;
	struct hash_entry *last = st->last;
	int32 lslots = (((byte *) (last+1)) - (byte *) e) / elemsize;
	int32 lused = 0;

	while (e <= last) {
		hash_hash_t hash = e->hash;
		if ((hash & HASH_MASK) == HASH_SUBHASH) {
			clean_st (*(struct hash_subtable **)e->data, slots, used);
		} else {
			lused += (hash != HASH_NIL);
		}
		e = HASH_OFFSET (e, elemsize);
	}
	free (st);
	*slots += lslots;
	*used += lused;
}

static void
hash_destroy (Hmap *h)
{
	int32 slots = 0;
	int32 used = 0;

	clean_st (h->st, &slots, &used);
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

// hash requires < 256 bytes of data (key+value) stored inline.
// Only basic types can be key - biggest is complex128 (16 bytes).
// Leave some room to grow, just in case.
enum {
	MaxValsize = 256 - 64
};

static void
donothing(uint32 s, void *a, void *b)
{
	USED(s);
	USED(a);
	USED(b);
}

static void
freedata(uint32 datavo, void *a, void *b)
{
	void *p;

	USED(a);
	p = *(void**)((byte*)b + datavo);
	free(p);
}

static void**
hash_indirect(Hmap *h, void *p)
{
	if(h->indirectval)
		p = *(void**)p;
	return p;
}	

static	int32	debug	= 0;

// makemap(typ *Type, hint uint32) (hmap *map[any]any);
Hmap*
runtime·makemap_c(MapType *typ, int64 hint)
{
	Hmap *h;
	int32 keyalg, valalg, keysize, valsize, valsize_in_hash;
	void (*data_del)(uint32, void*, void*);
	Type *key, *val;
	
	key = typ->key;
	val = typ->elem;

	if(hint < 0 || (int32)hint != hint)
		runtime·panicstring("makemap: size out of range");

	keyalg = key->alg;
	valalg = val->alg;
	keysize = key->size;
	valsize = val->size;

	if(keyalg >= nelem(runtime·algarray) || runtime·algarray[keyalg].hash == runtime·nohash) {
		runtime·printf("map(keyalg=%d)\n", keyalg);
		runtime·throw("runtime.makemap: unsupported map key type");
	}

	if(valalg >= nelem(runtime·algarray)) {
		runtime·printf("map(valalg=%d)\n", valalg);
		runtime·throw("runtime.makemap: unsupported map value type");
	}

	h = runtime·mal(sizeof(*h));

	valsize_in_hash = valsize;
	data_del = donothing;
	if (valsize > MaxValsize) {
		h->indirectval = 1;
		data_del = freedata;
		valsize_in_hash = sizeof(void*);
	} 

	// align value inside data so that mark-sweep gc can find it.
	// might remove in the future and just assume datavo == keysize.
	h->datavo = keysize;
	if(valsize_in_hash >= sizeof(void*))
		h->datavo = runtime·rnd(keysize, sizeof(void*));

	hash_init(h, h->datavo+valsize_in_hash,
		runtime·algarray[keyalg].hash,
		runtime·algarray[keyalg].equal,
		data_del,
		hint);

	h->keysize = keysize;
	h->valsize = valsize;
	h->keyalg = &runtime·algarray[keyalg];
	h->valalg = &runtime·algarray[valalg];

	// these calculations are compiler dependent.
	// figure out offsets of map call arguments.

	// func() (key, val)
	h->ko0 = runtime·rnd(sizeof(h), Structrnd);
	h->vo0 = runtime·rnd(h->ko0+keysize, val->align);

	// func(key) (val[, pres])
	h->ko1 = runtime·rnd(sizeof(h), key->align);
	h->vo1 = runtime·rnd(h->ko1+keysize, Structrnd);
	h->po1 = h->vo1 + valsize;

	// func(key, val[, pres])
	h->ko2 = runtime·rnd(sizeof(h), key->align);
	h->vo2 = runtime·rnd(h->ko2+keysize, val->align);
	h->po2 = h->vo2 + valsize;

	if(debug) {
		runtime·printf("makemap: map=%p; keysize=%d; valsize=%d; keyalg=%d; valalg=%d; offsets=%d,%d; %d,%d,%d; %d,%d,%d\n",
			h, keysize, valsize, keyalg, valalg, h->ko0, h->vo0, h->ko1, h->vo1, h->po1, h->ko2, h->vo2, h->po2);
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

	if(h == nil) {
		elem = t->elem;
		runtime·algarray[elem->alg].copy(elem->size, av, nil);
		*pres = false;
		return;
	}

	if(runtime·gcwaiting)
		runtime·gosched();

	res = nil;
	if(hash_lookup(h, ak, (void**)&res)) {
		*pres = true;
		h->valalg->copy(h->valsize, av, hash_indirect(h, res+h->datavo));
	} else {
		*pres = false;
		h->valalg->copy(h->valsize, av, nil);
	}
}

// mapaccess1(hmap *map[any]any, key any) (val any);
#pragma textflag 7
void
runtime·mapaccess1(MapType *t, Hmap *h, ...)
{
	byte *ak, *av;
	bool pres;

	if(h == nil) {
		ak = (byte*)(&h + 1);
		av = ak + runtime·rnd(t->key->size, Structrnd);
	} else {
		ak = (byte*)&h + h->ko1;
		av = (byte*)&h + h->vo1;
	}

	runtime·mapaccess(t, h, ak, av, &pres);

	if(debug) {
		runtime·prints("runtime.mapaccess1: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		h->keyalg->print(h->keysize, ak);
		runtime·prints("; val=");
		h->valalg->print(h->valsize, av);
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

	if(h == nil) {
		ak = (byte*)(&h + 1);
		av = ak + runtime·rnd(t->key->size, Structrnd);
		ap = av + t->elem->size;
	} else {
		ak = (byte*)&h + h->ko1;
		av = (byte*)&h + h->vo1;
		ap = (byte*)&h + h->po1;
	}

	runtime·mapaccess(t, h, ak, av, ap);

	if(debug) {
		runtime·prints("runtime.mapaccess2: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		h->keyalg->print(h->keysize, ak);
		runtime·prints("; val=");
		h->valalg->print(h->valsize, av);
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

	USED(t);

	if(h == nil)
		runtime·panicstring("assignment to entry in nil map");

	if(runtime·gcwaiting)
		runtime·gosched();

	res = nil;
	if(av == nil) {
		hash_remove(h, ak, (void**)&res);
		return;
	}

	hit = hash_insert(h, ak, (void**)&res);
	if(!hit && h->indirectval)
		*(void**)(res+h->datavo) = runtime·mal(h->valsize);
	h->keyalg->copy(h->keysize, res, ak);
	h->valalg->copy(h->valsize, hash_indirect(h, res+h->datavo), av);

	if(debug) {
		runtime·prints("mapassign: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		h->keyalg->print(h->keysize, ak);
		runtime·prints("; val=");
		h->valalg->print(h->valsize, av);
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

	ak = (byte*)&h + h->ko2;
	av = (byte*)&h + h->vo2;

	runtime·mapassign(t, h, ak, av);
}

// mapdelete(mapType *type, hmap *map[any]any, key any)
#pragma textflag 7
void
runtime·mapdelete(MapType *t, Hmap *h, ...)
{
	byte *ak;

	if(h == nil)
		runtime·panicstring("deletion of entry in nil map");

	ak = (byte*)&h + h->ko2;
	runtime·mapassign(t, h, ak, nil);

	if(debug) {
		runtime·prints("mapdelete: map=");
		runtime·printpointer(h);
		runtime·prints("; key=");
		h->keyalg->print(h->keysize, ak);
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
	if(h->keysize <= sizeof(key))
		ak = (byte*)&key;
	else
		ak = (byte*)key;
	if(h->valsize <= sizeof(val))
		av = (byte*)&val;
	else
		av = (byte*)val;
	if(!pres)
		av = nil;
	runtime·mapassign(t, h, ak, av);
}

// mapiterinit(mapType *type, hmap *map[any]any, hiter *any);
void
runtime·mapiterinit(MapType*, Hmap *h, struct hash_iter *it)
{
	if(h == nil) {
		it->data = nil;
		return;
	}
	hash_iter_init(h, it);
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
	it = runtime·mal(sizeof *it);
	FLUSH(&it);
	runtime·mapiterinit(t, h, it);
}

// mapiternext(hiter *any);
void
runtime·mapiternext(struct hash_iter *it)
{
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

	h = it->h;
	ak = (byte*)&it + h->ko0;

	res = it->data;
	if(res == nil)
		runtime·throw("runtime.mapiter1: key:val nil pointer");

	h->keyalg->copy(h->keysize, ak, res);

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
	Hmap *h;
	byte *res;

	h = it->h;
	res = it->data;
	if(res == nil)
		return false;
	h->keyalg->copy(h->keysize, ak, res);
	return true;
}

// For reflect:
//	func mapiterkey(h map) (key iword, ok bool)
// where an iword is the same word an interface value would use:
// the actual data if it fits, or else a pointer to the data.
void
reflect·mapiterkey(struct hash_iter *it, uintptr key, bool ok)
{
	Hmap *h;
	byte *res;

	key = 0;
	ok = false;
	h = it->h;
	res = it->data;
	if(res == nil) {
		key = 0;
		ok = false;
	} else {
		key = 0;
		if(h->keysize <= sizeof(key))
			h->keyalg->copy(h->keysize, (byte*)&key, res);
		else
			key = (uintptr)res;
		ok = true;
	}
	FLUSH(&key);
	FLUSH(&ok);
}

// For reflect:
//	func maplen(h map) (len int32)
// Like len(m) in the actual language, we treat the nil map as length 0.
void
reflect·maplen(Hmap *h, int32 len)
{
	if(h == nil)
		len = 0;
	else
		len = h->count;
	FLUSH(&len);
}

// mapiter2(hiter *any) (key any, val any);
#pragma textflag 7
void
runtime·mapiter2(struct hash_iter *it, ...)
{
	Hmap *h;
	byte *ak, *av, *res;

	h = it->h;
	ak = (byte*)&it + h->ko0;
	av = (byte*)&it + h->vo0;

	res = it->data;
	if(res == nil)
		runtime·throw("runtime.mapiter2: key:val nil pointer");

	h->keyalg->copy(h->keysize, ak, res);
	h->valalg->copy(h->valsize, av, hash_indirect(h, res+h->datavo));

	if(debug) {
		runtime·prints("mapiter2: iter=");
		runtime·printpointer(it);
		runtime·prints("; map=");
		runtime·printpointer(h);
		runtime·prints("\n");
	}
}
