// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"internal/goarch"
	"internal/runtime/atomic"
	"unsafe"
)

// A spanSet is a set of *mspans.
//
// spanSet is safe for concurrent push and pop operations.
type spanSet struct {
	// A spanSet is a two-level data structure consisting of a
	// growable spine that points to fixed-sized blocks. The spine
	// can be accessed without locks, but adding a block or
	// growing it requires taking the spine lock.
	//
	// Because each mspan covers at least 8K of heap and takes at
	// most 8 bytes in the spanSet, the growth of the spine is
	// quite limited.
	//
	// The spine and all blocks are allocated off-heap, which
	// allows this to be used in the memory manager and avoids the
	// need for write barriers on all of these. spanSetBlocks are
	// managed in a pool, though never freed back to the operating
	// system. We never release spine memory because there could be
	// concurrent lock-free access and we're likely to reuse it
	// anyway. (In principle, we could do this during STW.)

	spineLock mutex
	spine     atomicSpanSetSpinePointer // *[N]atomic.Pointer[spanSetBlock]
	spineLen  atomic.Uintptr            // Spine array length
	spineCap  uintptr                   // Spine array cap, accessed under spineLock

	// index is the head and tail of the spanSet in a single field.
	// The head and the tail both represent an index into the logical
	// concatenation of all blocks, with the head always behind or
	// equal to the tail (indicating an empty set). This field is
	// always accessed atomically.
	//
	// The head and the tail are only 32 bits wide, which means we
	// can only support up to 2^32 pushes before a reset. If every
	// span in the heap were stored in this set, and each span were
	// the minimum size (1 runtime page, 8 KiB), then roughly the
	// smallest heap which would be unrepresentable is 32 TiB in size.
	index atomicHeadTailIndex
}

const (
	spanSetBlockEntries = 512 // 4KB on 64-bit
	spanSetInitSpineCap = 256 // Enough for 1GB heap on 64-bit
)

type spanSetBlock struct {
	// Free spanSetBlocks are managed via a lock-free stack.
	lfnode

	// popped is the number of pop operations that have occurred on
	// this block. This number is used to help determine when a block
	// may be safely recycled.
	popped atomic.Uint32

	// spans is the set of spans in this block.
	spans [spanSetBlockEntries]atomicMSpanPointer
}

// push adds span s to buffer b. push is safe to call concurrently
// with other push and pop operations.
func (b *spanSet) push(s *mspan) {
	// Obtain our slot.
	cursor := uintptr(b.index.incTail().tail() - 1)
	top, bottom := cursor/spanSetBlockEntries, cursor%spanSetBlockEntries

	// Do we need to add a block?
	spineLen := b.spineLen.Load()
	var block *spanSetBlock
retry:
	if top < spineLen {
		block = b.spine.Load().lookup(top).Load()
	} else {
		// Add a new block to the spine, potentially growing
		// the spine.
		lock(&b.spineLock)
		// spineLen cannot change until we release the lock,
		// but may have changed while we were waiting.
		spineLen = b.spineLen.Load()
		if top < spineLen {
			unlock(&b.spineLock)
			goto retry
		}

		spine := b.spine.Load()
		if spineLen == b.spineCap {
			// Grow the spine.
			newCap := b.spineCap * 2
			if newCap == 0 {
				newCap = spanSetInitSpineCap
			}
			newSpine := persistentalloc(newCap*goarch.PtrSize, cpu.CacheLineSize, &memstats.gcMiscSys)
			if b.spineCap != 0 {
				// Blocks are allocated off-heap, so
				// no write barriers.
				memmove(newSpine, spine.p, b.spineCap*goarch.PtrSize)
			}
			spine = spanSetSpinePointer{newSpine}

			// Spine is allocated off-heap, so no write barrier.
			b.spine.StoreNoWB(spine)
			b.spineCap = newCap
			// We can't immediately free the old spine
			// since a concurrent push with a lower index
			// could still be reading from it. We let it
			// leak because even a 1TB heap would waste
			// less than 2MB of memory on old spines. If
			// this is a problem, we could free old spines
			// during STW.
		}

		// Allocate a new block from the pool.
		block = spanSetBlockPool.alloc()

		// Add it to the spine.
		// Blocks are allocated off-heap, so no write barrier.
		spine.lookup(top).StoreNoWB(block)
		b.spineLen.Store(spineLen + 1)
		unlock(&b.spineLock)
	}

	// We have a block. Insert the span atomically, since there may be
	// concurrent readers via the block API.
	block.spans[bottom].StoreNoWB(s)
}

// pop removes and returns a span from buffer b, or nil if b is empty.
// pop is safe to call concurrently with other pop and push operations.
func (b *spanSet) pop() *mspan {
	var head, tail uint32
claimLoop:
	for {
		headtail := b.index.load()
		head, tail = headtail.split()
		if head >= tail {
			// The buf is empty, as far as we can tell.
			return nil
		}
		// Check if the head position we want to claim is actually
		// backed by a block.
		spineLen := b.spineLen.Load()
		if spineLen <= uintptr(head)/spanSetBlockEntries {
			// We're racing with a spine growth and the allocation of
			// a new block (and maybe a new spine!), and trying to grab
			// the span at the index which is currently being pushed.
			// Instead of spinning, let's just notify the caller that
			// there's nothing currently here. Spinning on this is
			// almost definitely not worth it.
			return nil
		}
		// Try to claim the current head by CASing in an updated head.
		// This may fail transiently due to a push which modifies the
		// tail, so keep trying while the head isn't changing.
		want := head
		for want == head {
			if b.index.cas(headtail, makeHeadTailIndex(want+1, tail)) {
				break claimLoop
			}
			headtail = b.index.load()
			head, tail = headtail.split()
		}
		// We failed to claim the spot we were after and the head changed,
		// meaning a popper got ahead of us. Try again from the top because
		// the buf may not be empty.
	}
	top, bottom := head/spanSetBlockEntries, head%spanSetBlockEntries

	// We may be reading a stale spine pointer, but because the length
	// grows monotonically and we've already verified it, we'll definitely
	// be reading from a valid block.
	blockp := b.spine.Load().lookup(uintptr(top))

	// Given that the spine length is correct, we know we will never
	// see a nil block here, since the length is always updated after
	// the block is set.
	block := blockp.Load()
	s := block.spans[bottom].Load()
	for s == nil {
		// We raced with the span actually being set, but given that we
		// know a block for this span exists, the race window here is
		// extremely small. Try again.
		s = block.spans[bottom].Load()
	}
	// Clear the pointer. This isn't strictly necessary, but defensively
	// avoids accidentally re-using blocks which could lead to memory
	// corruption. This way, we'll get a nil pointer access instead.
	block.spans[bottom].StoreNoWB(nil)

	// Increase the popped count. If we are the last possible popper
	// in the block (note that bottom need not equal spanSetBlockEntries-1
	// due to races) then it's our responsibility to free the block.
	//
	// If we increment popped to spanSetBlockEntries, we can be sure that
	// we're the last popper for this block, and it's thus safe to free it.
	// Every other popper must have crossed this barrier (and thus finished
	// popping its corresponding mspan) by the time we get here. Because
	// we're the last popper, we also don't have to worry about concurrent
	// pushers (there can't be any). Note that we may not be the popper
	// which claimed the last slot in the block, we're just the last one
	// to finish popping.
	if block.popped.Add(1) == spanSetBlockEntries {
		// Clear the block's pointer.
		blockp.StoreNoWB(nil)

		// Return the block to the block pool.
		spanSetBlockPool.free(block)
	}
	return s
}

// reset resets a spanSet which is empty. It will also clean up
// any left over blocks.
//
// Throws if the buf is not empty.
//
// reset may not be called concurrently with any other operations
// on the span set.
func (b *spanSet) reset() {
	head, tail := b.index.load().split()
	if head < tail {
		print("head = ", head, ", tail = ", tail, "\n")
		throw("attempt to clear non-empty span set")
	}
	top := head / spanSetBlockEntries
	if uintptr(top) < b.spineLen.Load() {
		// If the head catches up to the tail and the set is empty,
		// we may not clean up the block containing the head and tail
		// since it may be pushed into again. In order to avoid leaking
		// memory since we're going to reset the head and tail, clean
		// up such a block now, if it exists.
		blockp := b.spine.Load().lookup(uintptr(top))
		block := blockp.Load()
		if block != nil {
			// Check the popped value.
			if block.popped.Load() == 0 {
				// popped should never be zero because that means we have
				// pushed at least one value but not yet popped if this
				// block pointer is not nil.
				throw("span set block with unpopped elements found in reset")
			}
			if block.popped.Load() == spanSetBlockEntries {
				// popped should also never be equal to spanSetBlockEntries
				// because the last popper should have made the block pointer
				// in this slot nil.
				throw("fully empty unfreed span set block found in reset")
			}

			// Clear the pointer to the block.
			blockp.StoreNoWB(nil)

			// Return the block to the block pool.
			spanSetBlockPool.free(block)
		}
	}
	b.index.reset()
	b.spineLen.Store(0)
}

// atomicSpanSetSpinePointer is an atomically-accessed spanSetSpinePointer.
//
// It has the same semantics as atomic.UnsafePointer.
type atomicSpanSetSpinePointer struct {
	a atomic.UnsafePointer
}

// Loads the spanSetSpinePointer and returns it.
//
// It has the same semantics as atomic.UnsafePointer.
func (s *atomicSpanSetSpinePointer) Load() spanSetSpinePointer {
	return spanSetSpinePointer{s.a.Load()}
}

// Stores the spanSetSpinePointer.
//
// It has the same semantics as [atomic.UnsafePointer].
func (s *atomicSpanSetSpinePointer) StoreNoWB(p spanSetSpinePointer) {
	s.a.StoreNoWB(p.p)
}

// spanSetSpinePointer represents a pointer to a contiguous block of atomic.Pointer[spanSetBlock].
type spanSetSpinePointer struct {
	p unsafe.Pointer
}

// lookup returns &s[idx].
func (s spanSetSpinePointer) lookup(idx uintptr) *atomic.Pointer[spanSetBlock] {
	return (*atomic.Pointer[spanSetBlock])(add(s.p, goarch.PtrSize*idx))
}

// spanSetBlockPool is a global pool of spanSetBlocks.
var spanSetBlockPool spanSetBlockAlloc

// spanSetBlockAlloc represents a concurrent pool of spanSetBlocks.
type spanSetBlockAlloc struct {
	stack lfstack
}

// alloc tries to grab a spanSetBlock out of the pool, and if it fails
// persistentallocs a new one and returns it.
func (p *spanSetBlockAlloc) alloc() *spanSetBlock {
	if s := (*spanSetBlock)(p.stack.pop()); s != nil {
		return s
	}
	return (*spanSetBlock)(persistentalloc(unsafe.Sizeof(spanSetBlock{}), cpu.CacheLineSize, &memstats.gcMiscSys))
}

// free returns a spanSetBlock back to the pool.
func (p *spanSetBlockAlloc) free(block *spanSetBlock) {
	block.popped.Store(0)
	p.stack.push(&block.lfnode)
}

// headTailIndex represents a combined 32-bit head and 32-bit tail
// of a queue into a single 64-bit value.
type headTailIndex uint64

// makeHeadTailIndex creates a headTailIndex value from a separate
// head and tail.
func makeHeadTailIndex(head, tail uint32) headTailIndex {
	return headTailIndex(uint64(head)<<32 | uint64(tail))
}

// head returns the head of a headTailIndex value.
func (h headTailIndex) head() uint32 {
	return uint32(h >> 32)
}

// tail returns the tail of a headTailIndex value.
func (h headTailIndex) tail() uint32 {
	return uint32(h)
}

// split splits the headTailIndex value into its parts.
func (h headTailIndex) split() (head uint32, tail uint32) {
	return h.head(), h.tail()
}

// atomicHeadTailIndex is an atomically-accessed headTailIndex.
type atomicHeadTailIndex struct {
	u atomic.Uint64
}

// load atomically reads a headTailIndex value.
func (h *atomicHeadTailIndex) load() headTailIndex {
	return headTailIndex(h.u.Load())
}

// cas atomically compares-and-swaps a headTailIndex value.
func (h *atomicHeadTailIndex) cas(old, new headTailIndex) bool {
	return h.u.CompareAndSwap(uint64(old), uint64(new))
}

// incHead atomically increments the head of a headTailIndex.
func (h *atomicHeadTailIndex) incHead() headTailIndex {
	return headTailIndex(h.u.Add(1 << 32))
}

// decHead atomically decrements the head of a headTailIndex.
func (h *atomicHeadTailIndex) decHead() headTailIndex {
	return headTailIndex(h.u.Add(-(1 << 32)))
}

// incTail atomically increments the tail of a headTailIndex.
func (h *atomicHeadTailIndex) incTail() headTailIndex {
	ht := headTailIndex(h.u.Add(1))
	// Check for overflow.
	if ht.tail() == 0 {
		print("runtime: head = ", ht.head(), ", tail = ", ht.tail(), "\n")
		throw("headTailIndex overflow")
	}
	return ht
}

// reset clears the headTailIndex to (0, 0).
func (h *atomicHeadTailIndex) reset() {
	h.u.Store(0)
}

// atomicMSpanPointer is an atomic.Pointer[mspan]. Can't use generics because it's NotInHeap.
type atomicMSpanPointer struct {
	p atomic.UnsafePointer
}

// Load returns the *mspan.
func (p *atomicMSpanPointer) Load() *mspan {
	return (*mspan)(p.p.Load())
}

// Store stores an *mspan.
func (p *atomicMSpanPointer) StoreNoWB(s *mspan) {
	p.p.StoreNoWB(unsafe.Pointer(s))
}
