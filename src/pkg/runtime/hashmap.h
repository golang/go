// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/* A hash table.
   Example, hashing nul-terminated char*s:
	hash_hash_t str_hash (void *v) {
		char *s;
		hash_hash_t hash = 0;
		for (s = *(char **)v; *s != 0; s++) {
			hash = (hash ^ *s) * 2654435769U;
		}
		return (hash);
	}
	int str_eq (void *a, void *b) {
		return (strcmp (*(char **)a, *(char **)b) == 0);
	}
	void str_del (void *arg, void *data) {
		*(char **)arg = *(char **)data;
	}

	struct hash *h = hash_new (sizeof (char *), &str_hash, &str_eq, &str_del, 3, 12, 15);
	...  3=> 2**3  entries initial size
	... 12=> 2**12 entries before sprouting sub-tables
	... 15=> number of adjacent probes to attempt before growing

  Example lookup:
	char *key = "foobar";
	char **result_ptr;
	if (hash_lookup (h, &key, (void **) &result_ptr)) {
	      printf ("found in table: %s\n", *result_ptr);
	} else {
	      printf ("not found in table\n");
	}

  Example insertion:
	char *key = strdup ("foobar");
	char **result_ptr;
	if (hash_lookup (h, &key, (void **) &result_ptr)) {
	      printf ("found in table: %s\n", *result_ptr);
	      printf ("to overwrite, do   *result_ptr = key\n");
	} else {
	      printf ("not found in table; inserted as %s\n", *result_ptr);
	      assert (*result_ptr == key);
	}

  Example deletion:
	char *key = "foobar";
	char *result;
	if (hash_remove (h, &key, &result)) {
	      printf ("key found and deleted from table\n");
	      printf ("called str_del (&result, data) to copy data to result: %s\n", result);
	} else {
	      printf ("not found in table\n");
	}

  Example iteration over the elements of *h:
	char **data;
	struct hash_iter it;
	hash_iter_init (h, &it);
	for (data = hash_next (&it); data != 0; data = hash_next (&it)) {
	    printf ("%s\n", *data);
	}
 */

#define	malloc		mal
#define	free(a)		USED(a)
#define	offsetof(s,m)	(uint32)(&(((s*)0)->m))
#define	memset(a,b,c)	runtime·memclr((byte*)(a), (uint32)(c))
#define	memcpy(a,b,c)	mcpy((byte*)(a),(byte*)(b),(uint32)(c))
#define	assert(a)	if(!(a)) throw("assert")

struct hash;		/* opaque */
struct hash_subtable;	/* opaque */
struct hash_entry;	/* opaque */

typedef uintptr uintptr_t;
typedef uintptr_t hash_hash_t;

struct hash_iter {
	uint8*	data;		/* returned from next */
	int32	elemsize;	/* size of elements in table */
	int32	changes;	/* number of changes observed last time */
	int32	i;		/* stack pointer in subtable_state */
	hash_hash_t last_hash;	/* last hash value returned */
	struct hash *h;		/* the hash table */
	struct hash_iter_sub {
		struct hash_entry *e;		/* pointer into subtable */
		struct hash_entry *start;	/* start of subtable */
		struct hash_entry *end;		/* end of subtable */
	} subtable_state[4];	/* Should be large enough unless the hashing is
				   so bad that many distinct data values hash
				   to the same hash value.  */
};

/* Return a hashtable h 2**init_power empty entries, each with
   "datasize" data bytes.
   (*data_hash)(a) should return the hash value of data element *a.
   (*data_eq)(a,b) should return whether the data at "a" and the data at "b"
   are equal.
   (*data_del)(arg, a) will be invoked when data element *a is about to be removed
   from the table.  "arg" is the argument passed to "hash_remove()".

   Growing is accomplished by resizing if the current tables size is less than
   a threshold, and by adding subtables otherwise.  hint should be set
   the expected maximum size of the table.
   "datasize" should be in [sizeof (void*), ..., 255].  If you need a
   bigger "datasize", store a pointer to another piece of memory. */

//struct hash *hash_new (int32 datasize,
//		hash_hash_t (*data_hash) (void *),
//		int32 (*data_eq) (void *, void *),
//		void (*data_del) (void *, void *),
//		int64 hint);

/* Lookup *data in *h.   If the data is found, return 1 and place a pointer to
   the found element in *pres.   Otherwise return 0 and place 0 in *pres. */
int32 hash_lookup (struct hash *h, void *data, void **pres);

/* Lookup *data in *h.  If the data is found, execute (*data_del) (arg, p)
   where p points to the data in the table, then remove it from *h and return
   1.  Otherwise return 0.  */
int32 hash_remove (struct hash *h, void *data, void *arg);

/* Lookup *data in *h.   If the data is found, return 1, and place a pointer
   to the found element in *pres.   Otherwise, return 0, allocate a region
   for the data to be inserted, and place a pointer to the inserted element
   in *pres; it is the caller's responsibility to copy the data to be
   inserted to the pointer returned in *pres in this case.

   If using garbage collection, it is the caller's responsibility to
   add references for **pres if HASH_ADDED is returned. */
int32 hash_insert (struct hash *h, void *data, void **pres);

/* Return the number of elements in the table. */
uint32 hash_count (struct hash *h);

/* The following call is useful only if not using garbage collection on the
   table.
   Remove all sub-tables associated with *h.
   This undoes the effects of hash_init().
   If other memory pointed to by user data must be freed, the caller is
   responsible for doiing do by iterating over *h first; see
   hash_iter_init()/hash_next().  */
void hash_destroy (struct hash *h);

/*----- iteration -----*/

/* Initialize *it from *h. */
void hash_iter_init (struct hash *h, struct hash_iter *it);

/* Return the next used entry in the table which which *it was initialized. */
void *hash_next (struct hash_iter *it);

/*---- test interface ----*/
/* Call (*data_visit) (arg, level, data) for every data entry in the table,
   whether used or not.   "level" is the subtable level, 0 means first level. */
/* TESTING ONLY: DO NOT USE THIS ROUTINE IN NORMAL CODE */
void hash_visit (struct hash *h, void (*data_visit) (void *arg, int32 level, void *data), void *arg);
