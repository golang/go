// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"internal/abi"
	"internal/goarch"
	"sync/atomic"
	"unsafe"
)

// HashTrieMap is an implementation of a concurrent hash-trie. The implementation
// is designed around frequent loads, but offers decent performance for stores
// and deletes as well, especially if the map is larger. Its primary use-case is
// the unique package, but can be used elsewhere as well.
//
// The zero HashTrieMap is empty and ready to use.
// It must not be copied after first use.
type HashTrieMap[K comparable, V any] struct {
	inited   atomic.Uint32
	initMu   Mutex
	root     atomic.Pointer[indirect[K, V]]
	keyHash  hashFunc
	valEqual equalFunc
	seed     uintptr
}

func (ht *HashTrieMap[K, V]) init() {
	if ht.inited.Load() == 0 {
		ht.initSlow()
	}
}

//go:noinline
func (ht *HashTrieMap[K, V]) initSlow() {
	ht.initMu.Lock()
	defer ht.initMu.Unlock()

	if ht.inited.Load() != 0 {
		// Someone got to it while we were waiting.
		return
	}

	// Set up root node, derive the hash function for the key, and the
	// equal function for the value, if any.
	var m map[K]V
	mapType := abi.TypeOf(m).MapType()
	ht.root.Store(newIndirectNode[K, V](nil))
	ht.keyHash = mapType.Hasher
	ht.valEqual = mapType.Elem.Equal
	ht.seed = uintptr(runtime_rand())

	ht.inited.Store(1)
}

type hashFunc func(unsafe.Pointer, uintptr) uintptr
type equalFunc func(unsafe.Pointer, unsafe.Pointer) bool

// Load returns the value stored in the map for a key, or nil if no
// value is present.
// The ok result indicates whether value was found in the map.
func (ht *HashTrieMap[K, V]) Load(key K) (value V, ok bool) {
	ht.init()
	hash := ht.keyHash(abi.NoEscape(unsafe.Pointer(&key)), ht.seed)

	i := ht.root.Load()
	hashShift := 8 * goarch.PtrSize
	for hashShift != 0 {
		hashShift -= nChildrenLog2

		n := i.children[(hash>>hashShift)&nChildrenMask].Load()
		if n == nil {
			return *new(V), false
		}
		if n.isEntry {
			return n.entry().lookup(key)
		}
		i = n.indirect()
	}
	panic("internal/concurrent.HashMapTrie: ran out of hash bits while iterating")
}

// LoadOrStore returns the existing value for the key if present.
// Otherwise, it stores and returns the given value.
// The loaded result is true if the value was loaded, false if stored.
func (ht *HashTrieMap[K, V]) LoadOrStore(key K, value V) (result V, loaded bool) {
	ht.init()
	hash := ht.keyHash(abi.NoEscape(unsafe.Pointer(&key)), ht.seed)
	var i *indirect[K, V]
	var hashShift uint
	var slot *atomic.Pointer[node[K, V]]
	var n *node[K, V]
	for {
		// Find the key or a candidate location for insertion.
		i = ht.root.Load()
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
				if v, ok := n.entry().lookup(key); ok {
					return v, true
				}
				haveInsertPoint = true
				break
			}
			i = n.indirect()
		}
		if !haveInsertPoint {
			panic("internal/concurrent.HashMapTrie: ran out of hash bits while iterating")
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

	var oldEntry *entry[K, V]
	if n != nil {
		oldEntry = n.entry()
		if v, ok := oldEntry.lookup(key); ok {
			// Easy case: by loading again, it turns out exactly what we wanted is here!
			return v, true
		}
	}
	newEntry := newEntryNode(key, value)
	if oldEntry == nil {
		// Easy case: create a new entry and store it.
		slot.Store(&newEntry.node)
	} else {
		// We possibly need to expand the entry already there into one or more new nodes.
		//
		// Publish the node last, which will make both oldEntry and newEntry visible. We
		// don't want readers to be able to observe that oldEntry isn't in the tree.
		slot.Store(ht.expand(oldEntry, newEntry, hash, hashShift, i))
	}
	return value, false
}

// expand takes oldEntry and newEntry whose hashes conflict from bit 64 down to hashShift and
// produces a subtree of indirect nodes to hold the two new entries.
func (ht *HashTrieMap[K, V]) expand(oldEntry, newEntry *entry[K, V], newHash uintptr, hashShift uint, parent *indirect[K, V]) *node[K, V] {
	// Check for a hash collision.
	oldHash := ht.keyHash(unsafe.Pointer(&oldEntry.key), ht.seed)
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
			panic("internal/concurrent.HashMapTrie: ran out of hash bits while inserting")
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

// Store sets the value for a key.
func (ht *HashTrieMap[K, V]) Store(key K, old V) {
	_, _ = ht.Swap(key, old)
}

// Swap swaps the value for a key and returns the previous value if any.
// The loaded result reports whether the key was present.
func (ht *HashTrieMap[K, V]) Swap(key K, new V) (previous V, loaded bool) {
	ht.init()
	hash := ht.keyHash(abi.NoEscape(unsafe.Pointer(&key)), ht.seed)
	var i *indirect[K, V]
	var hashShift uint
	var slot *atomic.Pointer[node[K, V]]
	var n *node[K, V]
	for {
		// Find the key or a candidate location for insertion.
		i = ht.root.Load()
		hashShift = 8 * goarch.PtrSize
		haveInsertPoint := false
		for hashShift != 0 {
			hashShift -= nChildrenLog2

			slot = &i.children[(hash>>hashShift)&nChildrenMask]
			n = slot.Load()
			if n == nil {
				// We found a nil slot which is a candidate for insertion,
				// or an existing entry that we'll replace.
				haveInsertPoint = true
				break
			}
			if n.isEntry {
				// Swap if the keys compare.
				old, swapped := n.entry().swap(key, new)
				if swapped {
					return old, true
				}
				// If we fail, that means we should try to insert.
				haveInsertPoint = true
				break
			}
			i = n.indirect()
		}
		if !haveInsertPoint {
			panic("internal/concurrent.HashMapTrie: ran out of hash bits while iterating")
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

	var zero V
	var oldEntry *entry[K, V]
	if n != nil {
		// Between before and now, something got inserted. Swap if the keys compare.
		oldEntry = n.entry()
		old, swapped := oldEntry.swap(key, new)
		if swapped {
			return old, true
		}
	}
	// The keys didn't compare, so we're doing an insertion.
	newEntry := newEntryNode(key, new)
	if oldEntry == nil {
		// Easy case: create a new entry and store it.
		slot.Store(&newEntry.node)
	} else {
		// We possibly need to expand the entry already there into one or more new nodes.
		//
		// Publish the node last, which will make both oldEntry and newEntry visible. We
		// don't want readers to be able to observe that oldEntry isn't in the tree.
		slot.Store(ht.expand(oldEntry, newEntry, hash, hashShift, i))
	}
	return zero, false
}

// CompareAndSwap swaps the old and new values for key
// if the value stored in the map is equal to old.
// The value type must be of a comparable type, otherwise CompareAndSwap will panic.
func (ht *HashTrieMap[K, V]) CompareAndSwap(key K, old, new V) (swapped bool) {
	ht.init()
	if ht.valEqual == nil {
		panic("called CompareAndSwap when value is not of comparable type")
	}
	hash := ht.keyHash(abi.NoEscape(unsafe.Pointer(&key)), ht.seed)
	for {
		// Find the key or return if it's not there.
		i := ht.root.Load()
		hashShift := 8 * goarch.PtrSize
		found := false
		for hashShift != 0 {
			hashShift -= nChildrenLog2

			slot := &i.children[(hash>>hashShift)&nChildrenMask]
			n := slot.Load()
			if n == nil {
				// Nothing to compare with. Give up.
				return false
			}
			if n.isEntry {
				// We found an entry. Try to compare and swap directly.
				return n.entry().compareAndSwap(key, old, new, ht.valEqual)
			}
			i = n.indirect()
		}
		if !found {
			panic("internal/concurrent.HashMapTrie: ran out of hash bits while iterating")
		}
	}
}

// LoadAndDelete deletes the value for a key, returning the previous value if any.
// The loaded result reports whether the key was present.
func (ht *HashTrieMap[K, V]) LoadAndDelete(key K) (value V, loaded bool) {
	ht.init()
	hash := ht.keyHash(abi.NoEscape(unsafe.Pointer(&key)), ht.seed)

	// Find a node with the key and compare with it. n != nil if we found the node.
	i, hashShift, slot, n := ht.find(key, hash, nil, *new(V))
	if n == nil {
		if i != nil {
			i.mu.Unlock()
		}
		return *new(V), false
	}

	// Try to delete the entry.
	v, e, loaded := n.entry().loadAndDelete(key)
	if !loaded {
		// Nothing was actually deleted, which means the node is no longer there.
		i.mu.Unlock()
		return *new(V), false
	}
	if e != nil {
		// We didn't actually delete the whole entry, just one entry in the chain.
		// Nothing else to do, since the parent is definitely not empty.
		slot.Store(&e.node)
		i.mu.Unlock()
		return v, true
	}
	// Delete the entry.
	slot.Store(nil)

	// Check if the node is now empty (and isn't the root), and delete it if able.
	for i.parent != nil && i.empty() {
		if hashShift == 8*goarch.PtrSize {
			panic("internal/concurrent.HashMapTrie: ran out of hash bits while iterating")
		}
		hashShift += nChildrenLog2

		// Delete the current node in the parent.
		parent := i.parent
		parent.mu.Lock()
		i.dead.Store(true)
		parent.children[(hash>>hashShift)&nChildrenMask].Store(nil)
		i.mu.Unlock()
		i = parent
	}
	i.mu.Unlock()
	return v, true
}

// Delete deletes the value for a key.
func (ht *HashTrieMap[K, V]) Delete(key K) {
	_, _ = ht.LoadAndDelete(key)
}

// CompareAndDelete deletes the entry for key if its value is equal to old.
// The value type must be comparable, otherwise this CompareAndDelete will panic.
//
// If there is no current value for key in the map, CompareAndDelete returns false
// (even if the old value is the nil interface value).
func (ht *HashTrieMap[K, V]) CompareAndDelete(key K, old V) (deleted bool) {
	ht.init()
	if ht.valEqual == nil {
		panic("called CompareAndDelete when value is not of comparable type")
	}
	hash := ht.keyHash(abi.NoEscape(unsafe.Pointer(&key)), ht.seed)

	// Find a node with the key. n != nil if we found the node.
	i, hashShift, slot, n := ht.find(key, hash, nil, *new(V))
	if n == nil {
		if i != nil {
			i.mu.Unlock()
		}
		return false
	}

	// Try to delete the entry.
	e, deleted := n.entry().compareAndDelete(key, old, ht.valEqual)
	if !deleted {
		// Nothing was actually deleted, which means the node is no longer there.
		i.mu.Unlock()
		return false
	}
	if e != nil {
		// We didn't actually delete the whole entry, just one entry in the chain.
		// Nothing else to do, since the parent is definitely not empty.
		slot.Store(&e.node)
		i.mu.Unlock()
		return true
	}
	// Delete the entry.
	slot.Store(nil)

	// Check if the node is now empty (and isn't the root), and delete it if able.
	for i.parent != nil && i.empty() {
		if hashShift == 8*goarch.PtrSize {
			panic("internal/concurrent.HashMapTrie: ran out of hash bits while iterating")
		}
		hashShift += nChildrenLog2

		// Delete the current node in the parent.
		parent := i.parent
		parent.mu.Lock()
		i.dead.Store(true)
		parent.children[(hash>>hashShift)&nChildrenMask].Store(nil)
		i.mu.Unlock()
		i = parent
	}
	i.mu.Unlock()
	return true
}

// find searches the tree for a node that contains key (hash must be the hash of key).
// If valEqual != nil, then it will also enforce that the values are equal as well.
//
// Returns a non-nil node, which will always be an entry, if found.
//
// If i != nil then i.mu is locked, and it is the caller's responsibility to unlock it.
func (ht *HashTrieMap[K, V]) find(key K, hash uintptr, valEqual equalFunc, value V) (i *indirect[K, V], hashShift uint, slot *atomic.Pointer[node[K, V]], n *node[K, V]) {
	for {
		// Find the key or return if it's not there.
		i = ht.root.Load()
		hashShift = 8 * goarch.PtrSize
		found := false
		for hashShift != 0 {
			hashShift -= nChildrenLog2

			slot = &i.children[(hash>>hashShift)&nChildrenMask]
			n = slot.Load()
			if n == nil {
				// Nothing to compare with. Give up.
				i = nil
				return
			}
			if n.isEntry {
				// We found an entry. Check if it matches.
				if _, ok := n.entry().lookupWithValue(key, value, valEqual); !ok {
					// No match, comparison failed.
					i = nil
					n = nil
					return
				}
				// We've got a match. Prepare to perform an operation on the key.
				found = true
				break
			}
			i = n.indirect()
		}
		if !found {
			panic("internal/concurrent.HashMapTrie: ran out of hash bits while iterating")
		}

		// Grab the lock and double-check what we saw.
		i.mu.Lock()
		n = slot.Load()
		if !i.dead.Load() && (n == nil || n.isEntry) {
			// Either we've got a valid node or the node is now nil under the lock.
			// In either case, we're done here.
			return
		}
		// We have to start over.
		i.mu.Unlock()
	}
}

// All returns an iterator over each key and value present in the map.
//
// The iterator does not necessarily correspond to any consistent snapshot of the
// HashTrieMap's contents: no key will be visited more than once, but if the value
// for any key is stored or deleted concurrently (including by yield), the iterator
// may reflect any mapping for that key from any point during iteration. The iterator
// does not block other methods on the receiver; even yield itself may call any
// method on the HashTrieMap.
func (ht *HashTrieMap[K, V]) All() func(yield func(K, V) bool) {
	ht.init()
	return func(yield func(key K, value V) bool) {
		ht.iter(ht.root.Load(), yield)
	}
}

// Range calls f sequentially for each key and value present in the map.
// If f returns false, range stops the iteration.
//
// This exists for compatibility with sync.Map; All should be preferred.
// It provides the same guarantees as sync.Map, and All.
func (ht *HashTrieMap[K, V]) Range(yield func(K, V) bool) {
	ht.init()
	ht.iter(ht.root.Load(), yield)
}

func (ht *HashTrieMap[K, V]) iter(i *indirect[K, V], yield func(key K, value V) bool) bool {
	for j := range i.children {
		n := i.children[j].Load()
		if n == nil {
			continue
		}
		if !n.isEntry {
			if !ht.iter(n.indirect(), yield) {
				return false
			}
			continue
		}
		e := n.entry()
		for e != nil {
			if !yield(e.key, *e.value.Load()) {
				return false
			}
			e = e.overflow.Load()
		}
	}
	return true
}

// Clear deletes all the entries, resulting in an empty HashTrieMap.
func (ht *HashTrieMap[K, V]) Clear() {
	ht.init()

	// It's sufficient to just drop the root on the floor, but the root
	// must always be non-nil.
	ht.root.Store(newIndirectNode[K, V](nil))
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
type indirect[K comparable, V any] struct {
	node[K, V]
	dead     atomic.Bool
	mu       Mutex // Protects mutation to children and any children that are entry nodes.
	parent   *indirect[K, V]
	children [nChildren]atomic.Pointer[node[K, V]]
}

func newIndirectNode[K comparable, V any](parent *indirect[K, V]) *indirect[K, V] {
	return &indirect[K, V]{node: node[K, V]{isEntry: false}, parent: parent}
}

func (i *indirect[K, V]) empty() bool {
	nc := 0
	for j := range i.children {
		if i.children[j].Load() != nil {
			nc++
		}
	}
	return nc == 0
}

// entry is a leaf node in the hash-trie.
type entry[K comparable, V any] struct {
	node[K, V]
	overflow atomic.Pointer[entry[K, V]] // Overflow for hash collisions.
	key      K
	value    atomic.Pointer[V]
}

func newEntryNode[K comparable, V any](key K, value V) *entry[K, V] {
	e := &entry[K, V]{
		node: node[K, V]{isEntry: true},
		key:  key,
	}
	e.value.Store(&value)
	return e
}

func (e *entry[K, V]) lookup(key K) (V, bool) {
	for e != nil {
		if e.key == key {
			return *e.value.Load(), true
		}
		e = e.overflow.Load()
	}
	return *new(V), false
}

func (e *entry[K, V]) lookupWithValue(key K, value V, valEqual equalFunc) (V, bool) {
	for e != nil {
		oldp := e.value.Load()
		if e.key == key && (valEqual == nil || valEqual(unsafe.Pointer(oldp), abi.NoEscape(unsafe.Pointer(&value)))) {
			return *oldp, true
		}
		e = e.overflow.Load()
	}
	return *new(V), false
}

// swap replaces a value in the overflow chain if keys compare equal.
// Returns the old value, and whether or not anything was swapped.
//
// swap must be called under the mutex of the indirect node which e is a child of.
func (head *entry[K, V]) swap(key K, newv V) (V, bool) {
	if head.key == key {
		vp := new(V)
		*vp = newv
		oldp := head.value.Swap(vp)
		return *oldp, true
	}
	i := &head.overflow
	e := i.Load()
	for e != nil {
		if e.key == key {
			vp := new(V)
			*vp = newv
			oldp := e.value.Swap(vp)
			return *oldp, true
		}
		i = &e.overflow
		e = e.overflow.Load()
	}
	var zero V
	return zero, false
}

// compareAndSwap replaces a value for a matching key and existing value in the overflow chain.
// Returns whether or not anything was swapped.
//
// compareAndSwap must be called under the mutex of the indirect node which e is a child of.
func (head *entry[K, V]) compareAndSwap(key K, oldv, newv V, valEqual equalFunc) bool {
	var vbox *V
outerLoop:
	for {
		oldvp := head.value.Load()
		if head.key == key && valEqual(unsafe.Pointer(oldvp), abi.NoEscape(unsafe.Pointer(&oldv))) {
			// Return the new head of the list.
			if vbox == nil {
				// Delay explicit creation of a new value to hold newv. If we just pass &newv
				// to CompareAndSwap, then newv will unconditionally escape, even if the CAS fails.
				vbox = new(V)
				*vbox = newv
			}
			if head.value.CompareAndSwap(oldvp, vbox) {
				return true
			}
			// We need to restart from the head of the overflow list in case, due to a removal, a node
			// is moved up the list and we miss it.
			continue outerLoop
		}
		i := &head.overflow
		e := i.Load()
		for e != nil {
			oldvp := e.value.Load()
			if e.key == key && valEqual(unsafe.Pointer(oldvp), abi.NoEscape(unsafe.Pointer(&oldv))) {
				if vbox == nil {
					// Delay explicit creation of a new value to hold newv. If we just pass &newv
					// to CompareAndSwap, then newv will unconditionally escape, even if the CAS fails.
					vbox = new(V)
					*vbox = newv
				}
				if e.value.CompareAndSwap(oldvp, vbox) {
					return true
				}
				continue outerLoop
			}
			i = &e.overflow
			e = e.overflow.Load()
		}
		return false
	}
}

// loadAndDelete deletes an entry in the overflow chain by key. Returns the value for the key, the new
// entry chain and whether or not anything was loaded (and deleted).
//
// loadAndDelete must be called under the mutex of the indirect node which e is a child of.
func (head *entry[K, V]) loadAndDelete(key K) (V, *entry[K, V], bool) {
	if head.key == key {
		// Drop the head of the list.
		return *head.value.Load(), head.overflow.Load(), true
	}
	i := &head.overflow
	e := i.Load()
	for e != nil {
		if e.key == key {
			i.Store(e.overflow.Load())
			return *e.value.Load(), head, true
		}
		i = &e.overflow
		e = e.overflow.Load()
	}
	return *new(V), head, false
}

// compareAndDelete deletes an entry in the overflow chain if both the key and value compare
// equal. Returns the new entry chain and whether or not anything was deleted.
//
// compareAndDelete must be called under the mutex of the indirect node which e is a child of.
func (head *entry[K, V]) compareAndDelete(key K, value V, valEqual equalFunc) (*entry[K, V], bool) {
	if head.key == key && valEqual(unsafe.Pointer(head.value.Load()), abi.NoEscape(unsafe.Pointer(&value))) {
		// Drop the head of the list.
		return head.overflow.Load(), true
	}
	i := &head.overflow
	e := i.Load()
	for e != nil {
		if e.key == key && valEqual(unsafe.Pointer(e.value.Load()), abi.NoEscape(unsafe.Pointer(&value))) {
			i.Store(e.overflow.Load())
			return head, true
		}
		i = &e.overflow
		e = e.overflow.Load()
	}
	return head, false
}

// node is the header for a node. It's polymorphic and
// is actually either an entry or an indirect.
type node[K comparable, V any] struct {
	isEntry bool
}

func (n *node[K, V]) entry() *entry[K, V] {
	if !n.isEntry {
		panic("called entry on non-entry node")
	}
	return (*entry[K, V])(unsafe.Pointer(n))
}

func (n *node[K, V]) indirect() *indirect[K, V] {
	if n.isEntry {
		panic("called indirect on entry node")
	}
	return (*indirect[K, V])(unsafe.Pointer(n))
}

// Pull in runtime.rand so that we don't need to take a dependency
// on math/rand/v2.
//
//go:linkname runtime_rand runtime.rand
func runtime_rand() uint64
