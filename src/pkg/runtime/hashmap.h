// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	// Note: the format of the Bucket is encoded in ../../cmd/gc/reflect.c and
	// ../reflect/type.go.  Don't change this structure without also changing that code!
	uint8  tophash[BUCKETSIZE]; // top 8 bits of hash of each entry (or special mark below)
	Bucket *overflow;           // overflow bucket, if any
	uint64 data[1];             // BUCKETSIZE keys followed by BUCKETSIZE values
};
// NOTE: packing all the keys together and then all the values together makes the
// code a bit more complicated than alternating key/value/key/value/... but it allows
// us to eliminate padding which would be needed for, e.g., map[int64]int8.

// tophash values.  We reserve a few possibilities for special marks.
// Each bucket (including its overflow buckets, if any) will have either all or none of its
// entries in the Evacuated* states (except during the evacuate() method, which only happens
// during map writes and thus no one else can observe the map during that time).
enum
{
	Empty = 0,		// cell is empty
	EvacuatedEmpty = 1,	// cell is empty, bucket is evacuated.
	EvacuatedX = 2,		// key/value is valid.  Entry has been evacuated to first half of larger table.
	EvacuatedY = 3,		// same as above, but evacuated to second half of larger table.
	MinTopHash = 4, 	// minimum tophash for a normal filled cell.
};
#define evacuated(b) ((b)->tophash[0] > Empty && (b)->tophash[0] < MinTopHash)

struct Hmap
{
	// Note: the format of the Hmap is encoded in ../../cmd/gc/reflect.c and
	// ../reflect/type.go.  Don't change this structure without also changing that code!
	uintgo  count;        // # live cells == size of map.  Must be first (used by len() builtin)
	uint32  flags;
	uint32  hash0;        // hash seed
	uint8   B;            // log_2 of # of buckets (can hold up to LOAD * 2^B items)
	uint8   keysize;      // key size in bytes
	uint8   valuesize;    // value size in bytes
	uint16  bucketsize;   // bucket size in bytes

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
};

// Macros for dereferencing indirect keys
#define IK(h, p) (((h)->flags & IndirectKey) != 0 ? *(byte**)(p) : (p))
#define IV(h, p) (((h)->flags & IndirectValue) != 0 ? *(byte**)(p) : (p))

// If you modify Hiter, also change cmd/gc/reflect.c to indicate
// the layout of this structure.
struct Hiter
{
	uint8* key; // Must be in first position.  Write nil to indicate iteration end (see cmd/gc/range.c).
	uint8* value; // Must be in second position (see cmd/gc/range.c).

	MapType *t;
	Hmap *h;
	byte *buckets; // bucket ptr at hash_iter initialization time
	struct Bucket *bptr; // current bucket

	uint8 offset; // intra-bucket offset to start from during iteration (should be big enough to hold BUCKETSIZE-1)
	bool done;

	// state of table at time iterator is initialized
	uint8 B;

	// iter state
	uintptr bucket;
	uintptr i;
	intptr check_bucket;
};

