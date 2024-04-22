// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package maps implements Go's builtin map type.
package maps

// This package contains the implementation of Go's builtin map type.
//
// The map design is based on Abseil's "Swiss Table" map design
// (https://abseil.io/about/design/swisstables), with additional modifications
// to cover Go's additional requirements, discussed below.
//
// Terminology:
// - Slot: A storage location of a single key/element pair.
// - Group: A group of abi.SwissMapGroupSlots (8) slots, plus a control word.
// - Control word: An 8-byte word which denotes whether each slot is empty,
//   deleted, or used. If a slot is used, its control byte also contains the
//   lower 7 bits of the hash (H2).
// - H1: Upper 57 bits of a hash.
// - H2: Lower 7 bits of a hash.
// - Table: A complete "Swiss Table" hash table. A table consists of one or
//   more groups for storage plus metadata to handle operation and determining
//   when to grow.
//
// At its core, the table design is similar to a traditional open-addressed
// hash table. Storage consists of an array of groups, which effectively means
// an array of key/elem slots with some control words interspersed. Lookup uses
// the hash to determine an initial group to check. If, due to collisions, this
// group contains no match, the probe sequence selects the next group to check
// (see below for more detail about the probe sequence).
//
// The key difference occurs within a group. In a standard open-addressed
// linear probed hash table, we would check each slot one at a time to find a
// match. A swiss table utilizes the extra control word to check all 8 slots in
// parallel.
//
// Each byte in the control word corresponds to one of the slots in the group.
// In each byte, 1 bit is used to indicate whether the slot is in use, or if it
// is empty/deleted. The other 7 bits contain the lower 7 bits of the hash for
// the key in that slot. See [ctrl] for the exact encoding.
//
// During lookup, we can use some clever bitwise manipulation to compare all 8
// 7-bit hashes against the input hash in parallel (see [ctrlGroup.matchH2]).
// That is, we effectively perform 8 steps of probing in a single operation.
// With SIMD instructions, this could be extended to 16 slots with a 16-byte
// control word.
//
// Since we only use 7 bits of the 64 bit hash, there is a 1 in 128 (~0.7%)
// probability of false positive on each slot, but that's fine: we always need
// double check each match with a standard key comparison regardless.
//
// Probing
//
// Probing is done using the upper 57 bits (H1) of the hash as an index into
// the groups array. Probing walks through the groups using quadratic probing
// until it finds a group with a match or a group with an empty slot. See
// [probeSeq] for specifics about the probe sequence. Note the probe
// invariants: the number of groups must be a power of two, and the end of a
// probe sequence must be a group with an empty slot (the table can never be
// 100% full).
//
// Deletion
//
// Probing stops when it finds a group with an empty slot. This affects
// deletion: when deleting from a completely full group, we must not mark the
// slot as empty, as there could be more slots used later in a probe sequence
// and this deletion would cause probing to stop too early. Instead, we mark
// such slots as "deleted" with a tombstone. If the group still has an empty
// slot, we don't need a tombstone and directly mark the slot empty. Currently,
// tombstone are only cleared during grow, as an in-place cleanup complicates
// iteration.
//
// Growth
//
// When the table reaches the maximum load factor, it grows by allocating a new
// groups array twice as big as before and reinserting all keys (the probe
// sequence will differ with a larger array).
// NOTE: Spoiler alert: A later CL supporting incremental growth will make each
// table instance have an immutable group count. Growth will allocate a
// completely new (bigger) table instance.
//
// Iteration
//
// Iteration is the most complex part of the map due to Go's generous iteration
// semantics. A summary of semantics from the spec:
// 1. Adding and/or deleting entries during iteration MUST NOT cause iteration
//    to return the same entry more than once.
// 2. Entries added during iteration MAY be returned by iteration.
// 3. Entries modified during iteration MUST return their latest value.
// 4. Entries deleted during iteration MUST NOT be returned by iteration.
// 5. Iteration order is unspecified. In the implementation, it is explicitly
//    randomized.
//
// If the map never grows, these semantics are straightforward: just iterate
// over every group and every slot and these semantics all land as expected.
//
// If the map grows during iteration, things complicate significantly. First
// and foremost, we need to track which entries we already returned to satisfy
// (1), but the larger table has a completely different probe sequence and thus
// different entry layout.
//
// We handle that by having the iterator keep a reference to the original table
// groups array even after the table grows. We keep iterating over the original
// groups to maintain the iteration order and avoid violating (1). Any new
// entries added only to the new groups will be skipped (allowed by (2)). To
// avoid violating (3) or (4), while we use the original groups to select the
// keys, we must look them up again in the new groups to determine if they have
// been modified or deleted. There is yet another layer of complexity if the
// key does not compare equal itself. See [Iter.Next] for the gory details.
//
// NOTE: Spoiler alert: A later CL supporting incremental growth will make this
// even more complicated. Yay!

// Extracts the H1 portion of a hash: the 57 upper bits.
// TODO(prattmic): what about 32-bit systems?
func h1(h uintptr) uintptr {
	return h >> 7
}

// Extracts the H2 portion of a hash: the 7 bits not used for h1.
//
// These are used as an occupied control byte.
func h2(h uintptr) uintptr {
	return h & 0x7f
}

type Map = table
