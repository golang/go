// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "hashmap.h"

/* Return a pointer to the struct/union of type "type"
   whose "field" field is addressed by pointer "p". */


struct hash {	   /* a hash table; initialize with hash_init() */
	uint32 count;	  /* elements in table - must be first */

	uint8 datasize;   /* amount of data to store in entry */
	uint8 max_power;  /* max power of 2 to create sub-tables */
	uint8 max_probes; /* max entries to probe before rehashing */
	int32 changes;	      /* inc'ed whenever a subtable is created/grown */
	hash_hash_t (*data_hash) (uint32, void *a);  /* return hash of *a */
	uint32 (*data_eq) (uint32, void *a, void *b);   /* return whether *a == *b */
	void (*data_del) (uint32, void *arg, void *data);  /* invoked on deletion */
	struct hash_subtable *st;    /* first-level table */

	uint32	keysize;
	uint32	valsize;
	uint32	datavo;
	uint32	ko;
	uint32	vo;
	uint32	po;
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
	struct hash_entry *end;      /* points just past end of entry[] */
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
hash_subtable_new (struct hash *h, int32 power, int32 used)
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
	st->end = HASH_OFFSET (st->entry, bytes);
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
hash_init (struct hash *h,
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
	datasize = rnd(datasize, sizeof (void *));
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
	struct hash_entry *end_e = st->end;
	int32 shift = HASH_BITS - (st->power + st->used);
	int32 index_mask = (((hash_hash_t)1) << st->power) - 1;
	int32 dst_i = (((byte *) dst_e) - ((byte *) st->entry)) / elemsize;
	int32 src_i = dst_i + n;
	hash_hash_t hash;
	int32 skip;
	int32 bytes;

	while (dst_e != src_e) {
		if (src_e != end_e) {
			struct hash_entry *cp_e = src_e;
			int32 save_dst_i = dst_i;
			while (cp_e != end_e && (hash = cp_e->hash) != HASH_NIL &&
			     ((hash >> shift) & index_mask) <= dst_i) {
				cp_e = HASH_OFFSET (cp_e, elemsize);
				dst_i++;
			}
			bytes = ((byte *) cp_e) - (byte *) src_e;
			memmove (dst_e, src_e, bytes);
			dst_e = HASH_OFFSET (dst_e, bytes);
			src_e = cp_e;
			src_i += dst_i - save_dst_i;
			if (src_e != end_e && (hash = src_e->hash) != HASH_NIL) {
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
		struct hash *h, void *data, void **pres);

static void
hash_conv (struct hash *h,
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
	while (e != st->end &&
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
		   (ne != st->end && (e_hash = ne->hash) != HASH_NIL &&
		   (e_hash & prefix_mask) == current)) {
			struct hash_subtable *new_st = hash_subtable_new (h, 1, HASH_USED (new_flags));
			int32 rc = hash_insert_internal (&new_st, new_flags, e->hash, h, e->data, &dummy_result);
			assert (rc == 0);
			memcpy(dummy_result, e->data, h->datasize);
			e = ne;
			while (e != st->end && (e_hash = e->hash) != HASH_NIL && (e_hash & prefix_mask) == current) {
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
hash_grow (struct hash *h, struct hash_subtable **pst, int32 flags)
{
	struct hash_subtable *old_st = *pst;
	int32 elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	*pst = hash_subtable_new (h, old_st->power + 1, HASH_USED (flags));
	struct hash_entry *end_e = old_st->end;
	struct hash_entry *e;
	void *dummy_result;
	int32 used = 0;

	flags |= HASH_REHASH;
	for (e = old_st->entry; e != end_e; e = HASH_OFFSET (e, elemsize)) {
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

int32
hash_lookup (struct hash *h, void *data, void **pres)
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

int32
hash_remove (struct hash *h, void *data, void *arg)
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
			(*h->data_del) (h->keysize, arg, e->data);
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
				 struct hash *h, void *data, void **pres)
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
			while (ins_e != st->end && (ins_e_hash = ins_e->hash) != HASH_NIL &&
			       ins_i + 1 - ((ins_e_hash >> shift) & index_mask) < st->max_probes &&
			       (ins_e_hash & HASH_MASK) != HASH_SUBHASH) {
				ins_e = HASH_OFFSET (ins_e, elemsize);
				ins_i++;
			}
			if (e == end_e || ins_e == st->end || ins_e_hash != HASH_NIL) {
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

int32
hash_insert (struct hash *h, void *data, void **pres)
{
	int32 rc = hash_insert_internal (&h->st, 0, (*h->data_hash) (h->keysize, data), h, data, pres);

	h->count += (rc == 0);    /* increment count if element didn't previously exist */
	return (rc);
}

uint32
hash_count (struct hash *h)
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
	struct hash_entry *end;

	for (;;) {
		int32 shift = HASH_BITS - (st->power + used);
		int32 index_mask = (1 << st->power) - 1;
		int32 i = (last_hash >> shift) & index_mask;

		end = st->end;
		e = HASH_OFFSET (st->entry, i * elemsize);
		sub->start = st->entry;
		sub->end = end;

		if ((e->hash & HASH_MASK) != HASH_SUBHASH) {
			break;
		}
		sub->e = HASH_OFFSET (e, elemsize);
		sub = &it->subtable_state[++(it->i)];
		used += st->power;
		st = *(struct hash_subtable **)e->data;
	}
	while (e != end && ((e_hash = e->hash) == HASH_NIL || e_hash <= last_hash)) {
		e = HASH_OFFSET (e, elemsize);
	}
	sub->e = e;
}

void *
hash_next (struct hash_iter *it)
{
	int32 elemsize = it->elemsize;
	struct hash_iter_sub *sub = &it->subtable_state[it->i];
	struct hash_entry *e = sub->e;
	struct hash_entry *end = sub->end;
	hash_hash_t e_hash = 0;

	if (it->changes != it->h->changes) {    /* hash table's structure changed; recompute */
		it->changes = it->h->changes;
		it->i = 0;
		iter_restart (it, it->h->st, 0);
		sub = &it->subtable_state[it->i];
		e = sub->e;
		end = sub->end;
	}
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
		while (e != end && ((e_hash = e->hash) == HASH_NIL || e_hash <= last_hash)) {
			e = HASH_OFFSET (e, elemsize);
		}
	}

	for (;;) {
		while (e != end && (e_hash = e->hash) == HASH_NIL) {
			e = HASH_OFFSET (e, elemsize);
		}
		if (e == end) {
			if (it->i == 0) {
				it->last_hash = HASH_OFFSET (e, -elemsize)->hash;
				sub->e = e;
				return (0);
			} else {
				it->i--;
				sub = &it->subtable_state[it->i];
				e = sub->e;
				end = sub->end;
			}
		} else if ((e_hash & HASH_MASK) != HASH_SUBHASH) {
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
			sub->end = end = st->end;
		}
	}
}

void
hash_iter_init (struct hash *h, struct hash_iter *it)
{
	it->elemsize = h->datasize + offsetof (struct hash_entry, data[0]);
	it->changes = h->changes;
	it->i = 0;
	it->h = h;
	it->last_hash = 0;
	it->subtable_state[0].e = h->st->entry;
	it->subtable_state[0].start = h->st->entry;
	it->subtable_state[0].end = h->st->end;
}

static void
clean_st (struct hash_subtable *st, int32 *slots, int32 *used)
{
	int32 elemsize = st->datasize + offsetof (struct hash_entry, data[0]);
	struct hash_entry *e = st->entry;
	struct hash_entry *end = st->end;
	int32 lslots = (((byte *) end) - (byte *) e) / elemsize;
	int32 lused = 0;

	while (e != end) {
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

void
hash_destroy (struct hash *h)
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

	while (e != st->end) {
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

void
hash_visit (struct hash *h, void (*data_visit) (void *arg, int32 level, void *data), void *arg)
{
	hash_visit_internal (h->st, 0, 0, data_visit, arg);
}

//
/// interfaces to go runtime
//

static void
donothing(uint32 s, void *a, void *b)
{
	USED(s);
	USED(a);
	USED(b);
}

typedef	struct	hash	Hmap;
static	int32	debug	= 0;

// newmap(keysize uint32, valsize uint32,
//	keyalg uint32, valalg uint32,
//	hint uint32) (hmap *map[any]any);
void
sys·newmap(uint32 keysize, uint32 valsize,
	uint32 keyalg, uint32 valalg, uint32 hint,
	Hmap* ret)
{
	Hmap *h;

	if(keyalg >= nelem(algarray) || algarray[keyalg].hash == nohash) {
		printf("map(keyalg=%d)\n", keyalg);
		throw("sys·newmap: unsupported map key type");
	}

	if(valalg >= nelem(algarray)) {
		printf("map(valalg=%d)\n", valalg);
		throw("sys·newmap: unsupported map value type");
	}

	h = mal(sizeof(*h));
	
	// align value inside data so that mark-sweep gc can find it.
	// might remove in the future and just assume datavo == keysize.
	h->datavo = keysize;
	if(valsize >= sizeof(void*))
		h->datavo = rnd(keysize, sizeof(void*));

	hash_init(h, h->datavo+valsize,
		algarray[keyalg].hash,
		algarray[keyalg].equal,
		donothing,
		hint);

	h->keysize = keysize;
	h->valsize = valsize;
	h->keyalg = &algarray[keyalg];
	h->valalg = &algarray[valalg];
	
	// these calculations are compiler dependent.
	// figure out offsets of map call arguments.
	h->ko = rnd(sizeof(h), keysize);
	h->vo = rnd(h->ko+keysize, valsize);
	h->po = rnd(h->vo+valsize, 1);

	ret = h;
	FLUSH(&ret);

	if(debug) {
		prints("newmap: map=");
		sys·printpointer(h);
		prints("; keysize=");
		sys·printint(keysize);
		prints("; valsize=");
		sys·printint(valsize);
		prints("; keyalg=");
		sys·printint(keyalg);
		prints("; valalg=");
		sys·printint(valalg);
		prints("; ko=");
		sys·printint(h->ko);
		prints("; vo=");
		sys·printint(h->vo);
		prints("; po=");
		sys·printint(h->po);
		prints("\n");
	}
}

// mapaccess1(hmap *map[any]any, key any) (val any);
void
sys·mapaccess1(Hmap *h, ...)
{
	byte *ak, *av;
	byte *res;
	int32 hit;

	ak = (byte*)&h + h->ko;
	av = (byte*)&h + h->vo;

	res = nil;
	hit = hash_lookup(h, ak, (void**)&res);
	if(!hit)
		throw("sys·mapaccess1: key not in map");
	h->valalg->copy(h->valsize, av, res+h->datavo);

	if(debug) {
		prints("sys·mapaccess1: map=");
		sys·printpointer(h);
		prints("; key=");
		h->keyalg->print(h->keysize, ak);
		prints("; val=");
		h->valalg->print(h->valsize, av);
		prints("; hit=");
		sys·printint(hit);
		prints("; res=");
		sys·printpointer(res);
		prints("\n");
	}
}

// mapaccess2(hmap *map[any]any, key any) (val any, pres bool);
void
sys·mapaccess2(Hmap *h, ...)
{
	byte *ak, *av, *ap;
	byte *res;
	int32 hit;

	ak = (byte*)&h + h->ko;
	av = (byte*)&h + h->vo;
	ap = (byte*)&h + h->po;

	res = nil;
	hit = hash_lookup(h, ak, (void**)&res);
	if(!hit) {
		*ap = false;
		h->valalg->copy(h->valsize, av, nil);
	} else {
		*ap = true;
		h->valalg->copy(h->valsize, av, res+h->datavo);
	}

	if(debug) {
		prints("sys·mapaccess2: map=");
		sys·printpointer(h);
		prints("; key=");
		h->keyalg->print(h->keysize, ak);
		prints("; val=");
		h->valalg->print(h->valsize, av);
		prints("; hit=");
		sys·printint(hit);
		prints("; res=");
		sys·printpointer(res);
		prints("; pres=");
		sys·printbool(*ap);
		prints("\n");
	}
}

static void
mapassign(Hmap *h, byte *ak, byte *av)
{
	byte *res;
	int32 hit;

	res = nil;
	hit = hash_insert(h, ak, (void**)&res);
	h->keyalg->copy(h->keysize, res, ak);
	h->valalg->copy(h->valsize, res+h->datavo, av);

	if(debug) {
		prints("mapassign: map=");
		sys·printpointer(h);
		prints("; key=");
		h->keyalg->print(h->keysize, ak);
		prints("; val=");
		h->valalg->print(h->valsize, av);
		prints("; hit=");
		sys·printint(hit);
		prints("; res=");
		sys·printpointer(res);
		prints("\n");
	}
}

// mapassign1(hmap *map[any]any, key any, val any);
void
sys·mapassign1(Hmap *h, ...)
{
	byte *ak, *av;

	ak = (byte*)&h + h->ko;
	av = (byte*)&h + h->vo;

	mapassign(h, ak, av);
}

// mapassign2(hmap *map[any]any, key any, val any, pres bool);
void
sys·mapassign2(Hmap *h, ...)
{
	byte *ak, *av, *ap;
	byte *res;
	int32 hit;

	ak = (byte*)&h + h->ko;
	av = (byte*)&h + h->vo;
	ap = (byte*)&h + h->po;

	if(*ap == true) {
		// assign
		mapassign(h, ak, av);
		return;
	}

	// delete
	hit = hash_remove(h, ak, (void**)&res);

	if(debug) {
		prints("mapassign2: map=");
		sys·printpointer(h);
		prints("; key=");
		h->keyalg->print(h->keysize, ak);
		prints("; hit=");
		sys·printint(hit);
		prints("; res=");
		sys·printpointer(res);
		prints("\n");
	}
}

// mapiterinit(hmap *map[any]any, hiter *any);
void
sys·mapiterinit(Hmap *h, struct hash_iter *it)
{
	if(h == nil) {
		it->data = nil;
		return;
	}
	hash_iter_init(h, it);
	it->data = hash_next(it);
	if(debug) {
		prints("sys·mapiterinit: map=");
		sys·printpointer(h);
		prints("; iter=");
		sys·printpointer(it);
		prints("; data=");
		sys·printpointer(it->data);
		prints("\n");
	}
}

// mapiternext(hiter *any);
void
sys·mapiternext(struct hash_iter *it)
{
	it->data = hash_next(it);
	if(debug) {
		prints("sys·mapiternext: iter=");
		sys·printpointer(it);
		prints("; data=");
		sys·printpointer(it->data);
		prints("\n");
	}
}

// mapiter1(hiter *any) (key any);
void
sys·mapiter1(struct hash_iter *it, ...)
{
	Hmap *h;
	byte *ak, *res;

	h = it->h;
	ak = (byte*)&it + h->ko;

	res = it->data;
	if(res == nil)
		throw("sys·mapiter2: key:val nil pointer");

	h->keyalg->copy(h->keysize, ak, res);

	if(debug) {
		prints("mapiter2: iter=");
		sys·printpointer(it);
		prints("; map=");
		sys·printpointer(h);
		prints("\n");
	}
}

// mapiter2(hiter *any) (key any, val any);
void
sys·mapiter2(struct hash_iter *it, ...)
{
	Hmap *h;
	byte *ak, *av, *res;

	h = it->h;
	ak = (byte*)&it + h->ko;
	av = (byte*)&it + h->vo;

	res = it->data;
	if(res == nil)
		throw("sys·mapiter2: key:val nil pointer");

	h->keyalg->copy(h->keysize, ak, res);
	h->valalg->copy(h->valsize, av, res+h->datavo);

	if(debug) {
		prints("mapiter2: iter=");
		sys·printpointer(it);
		prints("; map=");
		sys·printpointer(h);
		prints("\n");
	}
}
