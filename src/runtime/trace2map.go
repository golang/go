// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.exectracer2

// Simple append-only thread-safe hash map for tracing.
// Provides a mapping between variable-length data and a
// unique ID. Subsequent puts of the same data will return
// the same ID. The zero value is ready to use.
//
// Uses a region-based allocation scheme internally, and
// reset clears the whole map.
//
// It avoids doing any high-level Go operations so it's safe
// to use even in sensitive contexts.

package runtime

import (
	"internal/cpu"
	"internal/goarch"
	"internal/runtime/atomic"
	"runtime/internal/sys"
	"unsafe"
)

type traceMap struct {
	root atomic.UnsafePointer // *traceMapNode (can't use generics because it's notinheap)
	_    cpu.CacheLinePad
	seq  atomic.Uint64
	_    cpu.CacheLinePad
	mem  traceRegionAlloc
}

// traceMapNode is an implementation of a lock-free append-only hash-trie
// (a trie of the hash bits).
//
// Key features:
//   - 4-ary trie. Child nodes are indexed by the upper 2 (remaining) bits of the hash.
//     For example, top level uses bits [63:62], next level uses [61:60] and so on.
//   - New nodes are placed at the first empty level encountered.
//   - When the first child is added to a node, the existing value is not moved into a child.
//     This means that you must check the key at each level, not just at the leaf.
//   - No deletion or rebalancing.
//   - Intentionally devolves into a linked list on hash collisions (the hash bits will all
//     get shifted out during iteration, and new nodes will just be appended to the 0th child).
type traceMapNode struct {
	_ sys.NotInHeap

	children [4]atomic.UnsafePointer // *traceMapNode (can't use generics because it's notinheap)
	hash     uintptr
	id       uint64
	data     []byte
}

// stealID steals an ID from the table, ensuring that it will not
// appear in the table anymore.
func (tab *traceMap) stealID() uint64 {
	return tab.seq.Add(1)
}

// put inserts the data into the table.
//
// It's always safe for callers to noescape data because put copies its bytes.
//
// Returns a unique ID for the data and whether this is the first time
// the data has been added to the map.
func (tab *traceMap) put(data unsafe.Pointer, size uintptr) (uint64, bool) {
	if size == 0 {
		return 0, false
	}
	hash := memhash(data, 0, size)

	var newNode *traceMapNode
	m := &tab.root
	hashIter := hash
	for {
		n := (*traceMapNode)(m.Load())
		if n == nil {
			// Try to insert a new map node. We may end up discarding
			// this node if we fail to insert because it turns out the
			// value is already in the map.
			//
			// The discard will only happen if two threads race on inserting
			// the same value. Both might create nodes, but only one will
			// succeed on insertion. If two threads race to insert two
			// different values, then both nodes will *always* get inserted,
			// because the equality checking below will always fail.
			//
			// Performance note: contention on insertion is likely to be
			// higher for small maps, but since this data structure is
			// append-only, either the map stays small because there isn't
			// much activity, or the map gets big and races to insert on
			// the same node are much less likely.
			if newNode == nil {
				newNode = tab.newTraceMapNode(data, size, hash, tab.seq.Add(1))
			}
			if m.CompareAndSwapNoWB(nil, unsafe.Pointer(newNode)) {
				return newNode.id, true
			}
			// Reload n. Because pointers are only stored once,
			// we must have lost the race, and therefore n is not nil
			// anymore.
			n = (*traceMapNode)(m.Load())
		}
		if n.hash == hash && uintptr(len(n.data)) == size {
			if memequal(unsafe.Pointer(&n.data[0]), data, size) {
				return n.id, false
			}
		}
		m = &n.children[hashIter>>(8*goarch.PtrSize-2)]
		hashIter <<= 2
	}
}

func (tab *traceMap) newTraceMapNode(data unsafe.Pointer, size, hash uintptr, id uint64) *traceMapNode {
	// Create data array.
	sl := notInHeapSlice{
		array: tab.mem.alloc(size),
		len:   int(size),
		cap:   int(size),
	}
	memmove(unsafe.Pointer(sl.array), data, size)

	// Create metadata structure.
	meta := (*traceMapNode)(unsafe.Pointer(tab.mem.alloc(unsafe.Sizeof(traceMapNode{}))))
	*(*notInHeapSlice)(unsafe.Pointer(&meta.data)) = sl
	meta.id = id
	meta.hash = hash
	return meta
}

// reset drops all allocated memory from the table and resets it.
//
// The caller must ensure that there are no put operations executing concurrently
// with this function.
func (tab *traceMap) reset() {
	tab.root.Store(nil)
	tab.seq.Store(0)
	tab.mem.drop()
}
