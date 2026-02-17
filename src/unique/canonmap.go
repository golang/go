// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique

import (
	"internal/abi"
	"internal/goarch"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
	"weak"
)

// canonMap is a map of T -> *T. The map controls the creation
// of a canonical *T, and elements of the map are automatically
// deleted when the canonical *T is no longer referenced.
type canonMap[T comparable] struct {
	root atomic.Pointer[indirect[T]]
	hash func(unsafe.Pointer, uintptr) uintptr
	seed uintptr
}

func newCanonMap[T comparable]() *canonMap[T] {
	cm := new(canonMap[T])
	cm.root.Store(newIndirectNode[T](nil))

	var m map[T]struct{}
	mapType := abi.TypeOf(m).MapType()
	cm.hash = mapType.Hasher
	cm.seed = uintptr(runtime_rand())
	return cm
}

func (m *canonMap[T]) Load(key T) *T {
	hash := m.hash(abi.NoEscape(unsafe.Pointer(&key)), m.seed)

	i := m.root.Load()
	hashShift := 8 * goarch.PtrSize
	for hashShift != 0 {
		hashShift -= nChildrenLog2

		n := i.children[(hash>>hashShift)&nChildrenMask].Load()
		if n == nil {
			return nil
		}
		if n.isEntry {
			v, _ := n.entry().lookup(key)
			return v
		}
		i = n.indirect()
	}
	panic("unique.canonMap: ran out of hash bits while iterating")
}

func (m *canonMap[T]) LoadOrStore(key T) *T {
	hash := m.hash(abi.NoEscape(unsafe.Pointer(&key)), m.seed)

	var i *indirect[T]
	var hashShift uint
	var slot *atomic.Pointer[node[T]]
	var n *node[T]
	for {
		// Find the key or a candidate location for insertion.
		i = m.root.Load()
		hashShift = 8 * goarch.PtrSize
		haveInsertPoint := false
		for hashShift != 0 {
			hashShift -= nChildrenLog2

			slot = &i.children[(hash>>hashShift)&nChildrenMask]
			n = slot.Load()
			if n == nil {
				// We found a nil slot which is a candidate for insertion.
				haveInsertPoint = true
				break
			}
			if n.isEntry {
				// We found an existing entry, which is as far as we can go.
				// If it stays this way, we'll have to replace it with an
				// indirect node.
				if v, _ := n.entry().lookup(key); v != nil {
					return v
				}
				haveInsertPoint = true
				break
			}
			i = n.indirect()
		}
		if !haveInsertPoint {
			panic("unique.canonMap: ran out of hash bits while iterating")
		}

		// Grab the lock and double-check what we saw.
		i.mu.Lock()
		n = slot.Load()
		if (n == nil || n.isEntry) && !i.dead.Load() {
			// What we saw is still true, so we can continue with the insert.
			break
		}
		// We have to start over.
		i.mu.Unlock()
	}
	// N.B. This lock is held from when we broke out of the outer loop above.
	// We specifically break this out so that we can use defer here safely.
	// One option is to break this out into a new function instead, but
	// there's so much local iteration state used below that this turns out
	// to be cleaner.
	defer i.mu.Unlock()

	var oldEntry *entry[T]
	if n != nil {
		oldEntry = n.entry()
		if v, _ := oldEntry.lookup(key); v != nil {
			// Easy case: by loading again, it turns out exactly what we wanted is here!
			return v
		}
	}
	newEntry, canon, wp := newEntryNode(key, hash)
	// Prune dead pointers. This is to avoid O(n) lookups when we store the exact same
	// value in the set but the cleanup hasn't run yet because it got delayed for some
	// reason.
	oldEntry = oldEntry.prune()
	if oldEntry == nil {
		// Easy case: create a new entry and store it.
		slot.Store(&newEntry.node)
	} else {
		// We possibly need to expand the entry already there into one or more new nodes.
		//
		// Publish the node last, which will make both oldEntry and newEntry visible. We
		// don't want readers to be able to observe that oldEntry isn't in the tree.
		slot.Store(m.expand(oldEntry, newEntry, hash, hashShift, i))
	}
	runtime.AddCleanup(canon, func(_ struct{}) {
		m.cleanup(hash, wp)
	}, struct{}{})
	return canon
}

// expand takes oldEntry and newEntry whose hashes conflict from bit 64 down to hashShift and
// produces a subtree of indirect nodes to hold the two new entries. newHash is the hash of
// the value in the new entry.
func (m *canonMap[T]) expand(oldEntry, newEntry *entry[T], newHash uintptr, hashShift uint, parent *indirect[T]) *node[T] {
	// Check for a hash collision.
	oldHash := oldEntry.hash
	if oldHash == newHash {
		// Store the old entry in the new entry's overflow list, then store
		// the new entry.
		newEntry.overflow.Store(oldEntry)
		return &newEntry.node
	}
	// We have to add an indirect node. Worse still, we may need to add more than one.
	newIndirect := newIndirectNode(parent)
	top := newIndirect
	for {
		if hashShift == 0 {
			panic("unique.canonMap: ran out of hash bits while inserting")
		}
		hashShift -= nChildrenLog2 // hashShift is for the level parent is at. We need to go deeper.
		oi := (oldHash >> hashShift) & nChildrenMask
		ni := (newHash >> hashShift) & nChildrenMask
		if oi != ni {
			newIndirect.children[oi].Store(&oldEntry.node)
			newIndirect.children[ni].Store(&newEntry.node)
			break
		}
		nextIndirect := newIndirectNode(newIndirect)
		newIndirect.children[oi].Store(&nextIndirect.node)
		newIndirect = nextIndirect
	}
	return &top.node
}

// cleanup deletes the entry corresponding to wp in the canon map, if it's
// still in the map. wp must have a Value method that returns nil by the
// time this function is called. hash must be the hash of the value that
// wp once pointed to (that is, the hash of *wp.Value()).
func (m *canonMap[T]) cleanup(hash uintptr, wp weak.Pointer[T]) {
	var i *indirect[T]
	var hashShift uint
	var slot *atomic.Pointer[node[T]]
	var n *node[T]
	for {
		// Find wp in the map by following hash.
		i = m.root.Load()
		hashShift = 8 * goarch.PtrSize
		haveEntry := false
		for hashShift != 0 {
			hashShift -= nChildrenLog2

			slot = &i.children[(hash>>hashShift)&nChildrenMask]
			n = slot.Load()
			if n == nil {
				// We found a nil slot, already deleted.
				return
			}
			if n.isEntry {
				if !n.entry().hasWeakPointer(wp) {
					// The weak pointer was already pruned.
					return
				}
				haveEntry = true
				break
			}
			i = n.indirect()
		}
		if !haveEntry {
			panic("unique.canonMap: ran out of hash bits while iterating")
		}

		// Grab the lock and double-check what we saw.
		i.mu.Lock()
		n = slot.Load()
		if n != nil && n.isEntry {
			// Prune the entry node without thinking too hard. If we do
			// somebody else's work, such as someone trying to insert an
			// entry with the same hash (probably the same value) then
			// great, they'll back out without taking the lock.
			newEntry := n.entry().prune()
			if newEntry == nil {
				slot.Store(nil)
			} else {
				slot.Store(&newEntry.node)
			}

			// Delete interior nodes that are empty, up the tree.
			//
			// We'll hand-over-hand lock our way up the tree as we do this,
			// since we need to delete each empty node's link in its parent,
			// which requires the parents' lock.
			for i.parent != nil && i.empty() {
				if hashShift == 8*goarch.PtrSize {
					panic("unique.canonMap: ran out of hash bits while iterating")
				}
				hashShift += nChildrenLog2

				// Delete the current node in the parent.
				parent := i.parent
				parent.mu.Lock()
				i.dead.Store(true) // Could be done outside of parent's lock.
				parent.children[(hash>>hashShift)&nChildrenMask].Store(nil)
				i.mu.Unlock()
				i = parent
			}
			i.mu.Unlock()
			return
		}
		// We have to start over.
		i.mu.Unlock()
	}
}

// node is the header for a node. It's polymorphic and
// is actually either an entry or an indirect.
type node[T comparable] struct {
	isEntry bool
}

func (n *node[T]) entry() *entry[T] {
	if !n.isEntry {
		panic("called entry on non-entry node")
	}
	return (*entry[T])(unsafe.Pointer(n))
}

func (n *node[T]) indirect() *indirect[T] {
	if n.isEntry {
		panic("called indirect on entry node")
	}
	return (*indirect[T])(unsafe.Pointer(n))
}

const (
	// 16 children. This seems to be the sweet spot for
	// load performance: any smaller and we lose out on
	// 50% or more in CPU performance. Any larger and the
	// returns are minuscule (~1% improvement for 32 children).
	nChildrenLog2 = 4
	nChildren     = 1 << nChildrenLog2
	nChildrenMask = nChildren - 1
)

// indirect is an internal node in the hash-trie.
type indirect[T comparable] struct {
	node[T]
	dead     atomic.Bool
	parent   *indirect[T]
	mu       sync.Mutex // Protects mutation to children and any children that are entry nodes.
	children [nChildren]atomic.Pointer[node[T]]
}

func newIndirectNode[T comparable](parent *indirect[T]) *indirect[T] {
	return &indirect[T]{node: node[T]{isEntry: false}, parent: parent}
}

func (i *indirect[T]) empty() bool {
	for j := range i.children {
		if i.children[j].Load() != nil {
			return false
		}
	}
	return true
}

// entry is a leaf node in the hash-trie.
type entry[T comparable] struct {
	node[T]
	overflow atomic.Pointer[entry[T]] // Overflow for hash collisions.
	key      weak.Pointer[T]
	hash     uintptr
}

func newEntryNode[T comparable](key T, hash uintptr) (*entry[T], *T, weak.Pointer[T]) {
	k := new(T)
	*k = key
	wp := weak.Make(k)
	return &entry[T]{
		node: node[T]{isEntry: true},
		key:  wp,
		hash: hash,
	}, k, wp
}

// lookup finds the entry in the overflow chain that has the provided key.
//
// Returns the key's canonical pointer and the weak pointer for that canonical pointer.
func (e *entry[T]) lookup(key T) (*T, weak.Pointer[T]) {
	for e != nil {
		s := e.key.Value()
		if s != nil && *s == key {
			return s, e.key
		}
		e = e.overflow.Load()
	}
	return nil, weak.Pointer[T]{}
}

// hasWeakPointer returns true if the provided weak pointer can be found in the overflow chain.
func (e *entry[T]) hasWeakPointer(wp weak.Pointer[T]) bool {
	for e != nil {
		if e.key == wp {
			return true
		}
		e = e.overflow.Load()
	}
	return false
}

// prune removes all entries in the overflow chain whose keys are nil.
//
// The caller must hold the lock on e's parent node.
func (e *entry[T]) prune() *entry[T] {
	// Prune the head of the list.
	for e != nil {
		if e.key.Value() != nil {
			break
		}
		e = e.overflow.Load()
	}
	if e == nil {
		return nil
	}

	// Prune individual nodes in the list.
	newHead := e
	i := &e.overflow
	e = i.Load()
	for e != nil {
		if e.key.Value() != nil {
			i = &e.overflow
		} else {
			i.Store(e.overflow.Load())
		}
		e = e.overflow.Load()
	}
	return newHead
}

// Pull in runtime.rand so that we don't need to take a dependency
// on math/rand/v2.
//
//go:linkname runtime_rand runtime.rand
func runtime_rand() uint64
