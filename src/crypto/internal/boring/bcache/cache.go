// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bcache implements a GC-friendly cache (see [Cache]) for BoringCrypto.
package bcache

import (
	"sync/atomic"
	"unsafe"
)

// A Cache is a GC-friendly concurrent map from unsafe.Pointer to
// unsafe.Pointer. It is meant to be used for maintaining shadow
// BoringCrypto state associated with certain allocated structs, in
// particular public and private RSA and ECDSA keys.
//
// The cache is GC-friendly in the sense that the keys do not
// indefinitely prevent the garbage collector from collecting them.
// Instead, at the start of each GC, the cache is cleared entirely. That
// is, the cache is lossy, and the loss happens at the start of each GC.
// This means that clients need to be able to cope with cache entries
// disappearing, but it also means that clients don't need to worry about
// cache entries keeping the keys from being collected.
type Cache[K, V any] struct {
	// The runtime atomically stores nil to ptable at the start of each GC.
	ptable atomic.Pointer[cacheTable[K, V]]
}

type cacheTable[K, V any] [cacheSize]atomic.Pointer[cacheEntry[K, V]]

// A cacheEntry is a single entry in the linked list for a given hash table entry.
type cacheEntry[K, V any] struct {
	k    *K                // immutable once created
	v    atomic.Pointer[V] // read and written atomically to allow updates
	next *cacheEntry[K, V] // immutable once linked into table
}

func registerCache(unsafe.Pointer) // provided by runtime

// Register registers the cache with the runtime,
// so that c.ptable can be cleared at the start of each GC.
// Register must be called during package initialization.
func (c *Cache[K, V]) Register() {
	registerCache(unsafe.Pointer(&c.ptable))
}

// cacheSize is the number of entries in the hash table.
// The hash is the pointer value mod cacheSize, a prime.
// Collisions are resolved by maintaining a linked list in each hash slot.
const cacheSize = 1021

// table returns a pointer to the current cache hash table,
// coping with the possibility of the GC clearing it out from under us.
func (c *Cache[K, V]) table() *cacheTable[K, V] {
	for {
		p := c.ptable.Load()
		if p == nil {
			p = new(cacheTable[K, V])
			if !c.ptable.CompareAndSwap(nil, p) {
				continue
			}
		}
		return p
	}
}

// Clear clears the cache.
// The runtime does this automatically at each garbage collection;
// this method is exposed only for testing.
func (c *Cache[K, V]) Clear() {
	// The runtime does this at the start of every garbage collection
	// (itself, not by calling this function).
	c.ptable.Store(nil)
}

// Get returns the cached value associated with v,
// which is either the value v corresponding to the most recent call to Put(k, v)
// or nil if that cache entry has been dropped.
func (c *Cache[K, V]) Get(k *K) *V {
	head := &c.table()[uintptr(unsafe.Pointer(k))%cacheSize]
	e := head.Load()
	for ; e != nil; e = e.next {
		if e.k == k {
			return e.v.Load()
		}
	}
	return nil
}

// Put sets the cached value associated with k to v.
func (c *Cache[K, V]) Put(k *K, v *V) {
	head := &c.table()[uintptr(unsafe.Pointer(k))%cacheSize]

	// Strategy is to walk the linked list at head,
	// same as in Get, to look for existing entry.
	// If we find one, we update v atomically in place.
	// If not, then we race to replace the start = *head
	// we observed with a new k, v entry.
	// If we win that race, we're done.
	// Otherwise, we try the whole thing again,
	// with two optimizations:
	//
	//  1. We track in noK the start of the section of
	//     the list that we've confirmed has no entry for k.
	//     The next time down the list, we can stop at noK,
	//     because new entries are inserted at the front of the list.
	//     This guarantees we never traverse an entry
	//     multiple times.
	//
	//  2. We only allocate the entry to be added once,
	//     saving it in add for the next attempt.
	var add, noK *cacheEntry[K, V]
	n := 0
	for {
		e := head.Load()
		start := e
		for ; e != nil && e != noK; e = e.next {
			if e.k == k {
				e.v.Store(v)
				return
			}
			n++
		}
		if add == nil {
			add = &cacheEntry[K, V]{k: k}
			add.v.Store(v)
		}
		add.next = start
		if n >= 1000 {
			// If an individual list gets too long, which shouldn't happen,
			// throw it away to avoid quadratic lookup behavior.
			add.next = nil
		}
		if head.CompareAndSwap(start, add) {
			return
		}
		noK = start
	}
}
