// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/goexperiment"
	"internal/runtime/atomic"
	"unsafe"
)

// userFrameRegion describes a region of code that is not managed by the
// Go compiler (e.g., JIT-compiled code, WASM engines, embedded VMs).
//
// The runtime consults user frame regions when it encounters a PC that
// does not belong to any Go module. This allows the unwinder, panic
// handler, and GC to handle these frames gracefully instead of crashing.
type userFrameRegion struct {
	start uintptr // inclusive
	end   uintptr // exclusive

	unwindMode userFrameUnwindMode

	// describe returns human-readable frame info for a PC in this region.
	// Must not allocate. May be nil.
	describe func(pc uintptr) (name string, file string, line int, ok bool)

	stackMaps *userFrameStackMapTable

	handle uintptr
}

// userFrameStackMapInput must have the same memory layout as
// runtime/jit.StackMap. It is copied into runtime-owned metadata by Register.
type userFrameStackMapInput struct {
	pcOffset              uintptr
	frameOffset           uintptr
	frameWords            uintptr
	pointerMask           []byte
	hasUnwind             bool
	unwindBaseOffset      uintptr
	unwindBaseDeltaOffset uintptr
	unwindBaseUsesDelta   bool
	callerPCOffset        uintptr
	callerSPOffset        uintptr
	callerBPOffset        uintptr
}

type userFrameStackMap struct {
	pcOffset              uintptr
	frameOffset           uintptr
	frameWords            uintptr
	pointerMask           *byte
	hasUnwind             bool
	unwindBaseOffset      uintptr
	unwindBaseDeltaOffset uintptr
	unwindBaseUsesDelta   bool
	callerPCOffset        uintptr
	callerSPOffset        uintptr
	callerBPOffset        uintptr
}

// Each leaf holds a stable block of entries. Append never relocates entries or
// masks. The radix index grows by publishing new child pointers and, rarely, a
// new root descriptor; old roots remain valid Go objects for concurrent readers.
const userFrameMapBlockShift = 6
const userFrameMapIndexShift = 4

type userFrameStackMapBlock struct {
	entries [1 << userFrameMapBlockShift]userFrameStackMap
}

type userFrameStackMapNode struct {
	children [1 << userFrameMapIndexShift]unsafe.Pointer
}

type userFrameStackMapIndex struct {
	root   unsafe.Pointer
	levels uint8
}

type userFrameStackMapTable struct {
	index *userFrameStackMapIndex
	count uintptr
}

//go:nosplit
func userFrameMapAt(index *userFrameStackMapIndex, i uintptr) *userFrameStackMap {
	block := i >> userFrameMapBlockShift
	p := index.root
	for level := index.levels; level != 0; level-- {
		n := (*userFrameStackMapNode)(p)
		slot := (block >> ((level - 1) * userFrameMapIndexShift)) & ((1 << userFrameMapIndexShift) - 1)
		p = atomic.Loadp(unsafe.Pointer(&n.children[slot]))
	}
	return &(*userFrameStackMapBlock)(p).entries[i&((1<<userFrameMapBlockShift)-1)]
}

// userFrameMapAppendSlot is writer-only. Publish count after initializing every
// new entry. Readers with an older count can safely use the expanded index.
func userFrameMapAppendSlot(table *userFrameStackMapTable, i uintptr) *userFrameStackMap {
	index := table.index
	if index == nil {
		index = &userFrameStackMapIndex{root: unsafe.Pointer(new(userFrameStackMapBlock))}
		atomicstorep(unsafe.Pointer(&table.index), unsafe.Pointer(index))
	}
	block := i >> userFrameMapBlockShift
	for block>>(index.levels*userFrameMapIndexShift) != 0 {
		root := new(userFrameStackMapNode)
		root.children[0] = index.root
		index = &userFrameStackMapIndex{root: unsafe.Pointer(root), levels: index.levels + 1}
		atomicstorep(unsafe.Pointer(&table.index), unsafe.Pointer(index))
	}
	p := index.root
	for level := index.levels; level != 0; level-- {
		n := (*userFrameStackMapNode)(p)
		slot := (block >> ((level - 1) * userFrameMapIndexShift)) & ((1 << userFrameMapIndexShift) - 1)
		child := n.children[slot]
		if child == nil {
			if level == 1 {
				child = unsafe.Pointer(new(userFrameStackMapBlock))
			} else {
				child = unsafe.Pointer(new(userFrameStackMapNode))
			}
			atomicstorep(unsafe.Pointer(&n.children[slot]), child)
		}
		p = child
	}
	return &(*userFrameStackMapBlock)(p).entries[i&((1<<userFrameMapBlockShift)-1)]
}

type userFrameUnwindMode uint8

const (
	userFrameUnwindStop    userFrameUnwindMode = iota // traceback ends at boundary
	userFrameUnwindSkip                               // frames are skipped, traceback continues
	userFrameUnwindDeclare                            // frames are described via callbacks
)

// User frame registry. Writers are serialized by runtime/jit's registryMu.
// Nodes are immutable after publication. An update copies O(log N) nodes;
// ordinary typed Go references retain the old paths for concurrent readers.
// The handle index belongs only to writers and never enters stack walks.
type userFrameNode struct {
	region *userFrameRegion
	left   *userFrameNode
	right  *userFrameNode
	height int
}

var (
	userFrameRoot     *userFrameNode
	userFrameByHandle map[uintptr]*userFrameRegion
	userFrameHandle   uintptr
)

func userFrameHeight(n *userFrameNode) int {
	if n == nil {
		return 0
	}
	return n.height
}

func userFrameNodeFor(r *userFrameRegion, left, right *userFrameNode) *userFrameNode {
	return &userFrameNode{region: r, left: left, right: right, height: 1 + max(userFrameHeight(left), userFrameHeight(right))}
}

func userFrameRotateLeft(n *userFrameNode) *userFrameNode {
	r := n.right
	return userFrameNodeFor(r.region, userFrameNodeFor(n.region, n.left, r.left), r.right)
}

func userFrameRotateRight(n *userFrameNode) *userFrameNode {
	l := n.left
	return userFrameNodeFor(l.region, l.left, userFrameNodeFor(n.region, l.right, n.right))
}

func userFrameBalance(n *userFrameNode) *userFrameNode {
	balance := userFrameHeight(n.left) - userFrameHeight(n.right)
	if balance > 1 {
		if userFrameHeight(n.left.left) < userFrameHeight(n.left.right) {
			n = userFrameNodeFor(n.region, userFrameRotateLeft(n.left), n.right)
		}
		return userFrameRotateRight(n)
	}
	if balance < -1 {
		if userFrameHeight(n.right.right) < userFrameHeight(n.right.left) {
			n = userFrameNodeFor(n.region, n.left, userFrameRotateRight(n.right))
		}
		return userFrameRotateLeft(n)
	}
	return n
}

func userFrameInsert(n *userFrameNode, r *userFrameRegion) *userFrameNode {
	if n == nil {
		return userFrameNodeFor(r, nil, nil)
	}
	if r.start < n.region.start {
		return userFrameBalance(userFrameNodeFor(n.region, userFrameInsert(n.left, r), n.right))
	}
	return userFrameBalance(userFrameNodeFor(n.region, n.left, userFrameInsert(n.right, r)))
}

func userFrameRemove(n *userFrameNode, start uintptr) *userFrameNode {
	if start < n.region.start {
		return userFrameBalance(userFrameNodeFor(n.region, userFrameRemove(n.left, start), n.right))
	}
	if start > n.region.start {
		return userFrameBalance(userFrameNodeFor(n.region, n.left, userFrameRemove(n.right, start)))
	}
	if n.left == nil {
		return n.right
	}
	if n.right == nil {
		return n.left
	}
	successor := n.right
	for successor.left != nil {
		successor = successor.left
	}
	return userFrameBalance(userFrameNodeFor(successor.region, n.left, userFrameRemove(n.right, successor.region.start)))
}

func registerUserFrameRegion(r userFrameRegion) uintptr {
	if r.start >= r.end {
		throw("registerUserFrameRegion: invalid address range")
	}
	root := (*userFrameNode)(atomic.Loadp(unsafe.Pointer(&userFrameRoot)))
	for n := root; n != nil; {
		if r.end <= n.region.start {
			n = n.left
		} else if r.start >= n.region.end {
			n = n.right
		} else {
			throw("registerUserFrameRegion: overlapping region")
		}
	}
	userFrameHandle++
	r.handle = userFrameHandle
	if userFrameByHandle == nil {
		userFrameByHandle = make(map[uintptr]*userFrameRegion)
	}
	region := new(userFrameRegion)
	*region = r
	userFrameByHandle[r.handle] = region
	root = userFrameInsert(root, region)
	atomicstorep(unsafe.Pointer(&userFrameRoot), unsafe.Pointer(root))
	return r.handle
}

func unregisterUserFrameRegion(handle uintptr) {
	r := userFrameByHandle[handle]
	if r == nil {
		throw("unregisterUserFrameRegion: handle not found")
	}
	root := (*userFrameNode)(atomic.Loadp(unsafe.Pointer(&userFrameRoot)))
	root = userFrameRemove(root, r.start)
	atomicstorep(unsafe.Pointer(&userFrameRoot), unsafe.Pointer(root))
	delete(userFrameByHandle, handle)
	if len(userFrameByHandle) == 0 {
		userFrameByHandle = nil
	}
}

// findUserFrameRegion performs a lock-free lookup without copying metadata.
// The returned Go pointer keeps the region and its metadata reachable.
//
//go:nosplit
func findUserFrameRegion(pc uintptr) *userFrameRegion {
	if !goexperiment.JIT {
		return nil
	}
	n := (*userFrameNode)(atomic.Loadp(unsafe.Pointer(&userFrameRoot)))
	for n != nil {
		if pc < n.region.start {
			n = n.left
		} else if pc >= n.region.end {
			n = n.right
		} else {
			return n.region
		}
	}
	return nil
}

func userFrameNext(r *userFrameRegion, pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool) {
	if m := findUserFrameStackMap(r, pc); m != nil && m.hasUnwind {
		base, ok := userFrameUnwindBase(m, sp)
		if !ok {
			return 0, 0, 0, false
		}
		callerPCAddr := base + m.callerPCOffset
		callerSP = base + m.callerSPOffset
		callerBPAddr := base + m.callerBPOffset
		if callerPCAddr < base || callerSP < base || callerBPAddr < base {
			return 0, 0, 0, false
		}
		return *(*uintptr)(unsafe.Pointer(callerPCAddr)), callerSP, *(*uintptr)(unsafe.Pointer(callerBPAddr)), true
	}
	return 0, 0, 0, false
}

func userFrameUnwindBase(m *userFrameStackMap, sp uintptr) (uintptr, bool) {
	base := sp + m.unwindBaseOffset
	if base < sp {
		return 0, false
	}
	if m.unwindBaseUsesDelta {
		deltaAddr := sp + m.unwindBaseDeltaOffset
		if deltaAddr < sp {
			return 0, false
		}
		delta := *(*uintptr)(unsafe.Pointer(deltaAddr))
		if base+delta < base {
			return 0, false
		}
		base += delta
	}
	return base, true
}

// userFrameCallerBPSlot returns the address of the saved caller frame pointer.
// Stack copying must relocate this value even when the user frame has no Go
// pointers of its own. On amd64, Go function epilogues may use LEAVE, so a
// stale frame pointer would also restore SP to the old, freed stack.
//
//go:nosplit
func userFrameCallerBPSlot(pc, sp uintptr) (uintptr, bool) {
	r := findUserFrameRegion(pc)
	m := findUserFrameStackMap(r, pc)
	if m == nil || !m.hasUnwind {
		return 0, false
	}
	base, ok := userFrameUnwindBase(m, sp)
	if !ok || base+m.callerBPOffset < base {
		return 0, false
	}
	return base + m.callerBPOffset, true
}

// userFrameNextGo unwinds a contiguous chain of user frames and returns the
// first Go caller. Consecutive user frames arise when generated functions call
// one another directly rather than returning through an intermediate Go
// trampoline.
//
//go:nosplit
func userFrameNextGo(pc, sp uintptr) (callerPC, callerSP uintptr, ok bool) {
	for {
		r := findUserFrameRegion(pc)
		if r == nil {
			return 0, 0, false
		}
		callerPC, callerSP, _, ok = userFrameNext(r, pc, sp)
		if !ok || callerPC == 0 || callerSP <= sp {
			return 0, 0, false
		}
		if findfunc(callerPC).valid() {
			return callerPC, callerSP, true
		}
		if findUserFrameRegion(callerPC) == nil {
			return 0, 0, false
		}
		pc, sp = callerPC, callerSP
	}
}

// userFrameNextUser returns the immediately enclosing user frame, if any.
// Callers use this to scan, relocate, or describe every frame in a contiguous
// chain after the main unwinder has validated that the chain reaches Go.
//
//go:nosplit
func userFrameNextUser(pc, sp uintptr) (callerPC, callerSP uintptr, ok bool) {
	r := findUserFrameRegion(pc)
	if r == nil {
		return 0, 0, false
	}
	callerPC, callerSP, _, ok = userFrameNext(r, pc, sp)
	if !ok || callerPC == 0 || callerSP <= sp || findUserFrameRegion(callerPC) == nil {
		return 0, 0, false
	}
	return callerPC, callerSP, true
}

// userFramePointerMap resolves one immutable map without calling user code.
// Looking the region up again keeps unwinder pointer-free.
//
//go:nosplit
func userFramePointerMap(pc, sp uintptr) (base, words uintptr, pointerMask *byte, ok bool) {
	r := findUserFrameRegion(pc)
	m := findUserFrameStackMap(r, pc)
	if m == nil || m.frameWords == 0 || sp+m.frameOffset < sp {
		return 0, 0, nil, false
	}
	return sp + m.frameOffset, m.frameWords, m.pointerMask, true
}

//go:nosplit
func findUserFrameStackMap(r *userFrameRegion, pc uintptr) *userFrameStackMap {
	if r == nil || r.stackMaps == nil {
		return nil
	}
	count := atomic.Loaduintptr(&r.stackMaps.count)
	index := (*userFrameStackMapIndex)(atomic.Loadp(unsafe.Pointer(&r.stackMaps.index)))
	if count == 0 || index == nil {
		return nil
	}
	offset := pc - r.start
	lo, hi := uintptr(0), count
	for lo < hi {
		mid := (lo + hi) >> 1
		m := userFrameMapAt(index, mid)
		if m.pcOffset < offset {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo == count {
		return nil
	}
	m := userFrameMapAt(index, lo)
	if m.pcOffset != offset {
		return nil
	}
	return m
}

func appendUserFrameStackMaps(table *userFrameStackMapTable, start, end uintptr, unwindMode userFrameUnwindMode, input []userFrameStackMapInput) {
	if len(input) == 0 {
		return
	}
	oldCount := atomic.Loaduintptr(&table.count)
	newCount := oldCount + uintptr(len(input))
	if newCount < oldCount {
		throw("registerUserFrameRegion: too many stack maps")
	}
	var previousPC uintptr
	if oldCount != 0 {
		previous := userFrameMapAt(table.index, oldCount-1)
		previousPC = previous.pcOffset
	}
	for i := range input {
		in := &input[i]
		if in.pcOffset >= end-start || ((oldCount != 0 || i > 0) && in.pcOffset <= previousPC) {
			throw("registerUserFrameRegion: unordered or invalid stack map PC")
		}
		if unwindMode != userFrameUnwindStop && !in.hasUnwind {
			throw("registerUserFrameRegion: declarative unwind required at every safepoint")
		}
		if in.frameWords > uintptr(^uint32(0)>>1) {
			throw("registerUserFrameRegion: stack map too large")
		}
		frameBytes := in.frameWords * goarch.PtrSize
		if frameBytes/goarch.PtrSize != in.frameWords || in.frameOffset+frameBytes < in.frameOffset {
			throw("registerUserFrameRegion: invalid stack map range")
		}
		maskBytes := (in.frameWords + 7) / 8
		if maskBytes < in.frameWords/8 || uintptr(len(in.pointerMask)) < maskBytes {
			throw("registerUserFrameRegion: short stack map bitmap")
		}
		var mask *byte
		if maskBytes != 0 {
			mask = &make([]byte, maskBytes)[0]
			copy(unsafe.Slice(mask, maskBytes), in.pointerMask[:maskBytes])
		}
		out := userFrameMapAppendSlot(table, oldCount+uintptr(i))
		*out = userFrameStackMap{
			pcOffset:              in.pcOffset,
			frameOffset:           in.frameOffset,
			frameWords:            in.frameWords,
			pointerMask:           mask,
			hasUnwind:             in.hasUnwind,
			unwindBaseOffset:      in.unwindBaseOffset,
			unwindBaseDeltaOffset: in.unwindBaseDeltaOffset,
			unwindBaseUsesDelta:   in.unwindBaseUsesDelta,
			callerPCOffset:        in.callerPCOffset,
			callerSPOffset:        in.callerSPOffset,
			callerBPOffset:        in.callerBPOffset,
		}
		previousPC = in.pcOffset
	}
	// Publish only after every entry and bitmap is fully initialized.
	atomic.Storeuintptr(&table.count, newCount)
}

// userFramePreempt reports whether the current goroutine has been
// asked to yield by the runtime (GC, scheduler). JIT code calls this
// at safepoints via jit.Preempt().
//
//go:nosplit
func userFramePreempt() bool {
	gp := getg()
	if gp == nil {
		return false
	}
	// If called from g0, check the user goroutine.
	if gp.m != nil && gp.m.curg != nil {
		gp = gp.m.curg
	}
	return gp.preempt
}

// Linknames for the runtime/jit package.

//go:linkname jit_registerUserFrameRegion runtime/jit.registerUserFrameRegion
func jit_registerUserFrameRegion(start, end uintptr, unwindMode uint8,
	describe func(pc uintptr) (string, string, int, bool),
	stackMaps []userFrameStackMapInput,
) uintptr {
	if start >= end {
		throw("registerUserFrameRegion: invalid address range")
	}
	if userFrameUnwindMode(unwindMode) > userFrameUnwindDeclare {
		throw("registerUserFrameRegion: invalid unwind mode")
	}
	table := new(userFrameStackMapTable)
	appendUserFrameStackMaps(table, start, end, userFrameUnwindMode(unwindMode), stackMaps)
	r := userFrameRegion{
		start:      start,
		end:        end,
		unwindMode: userFrameUnwindMode(unwindMode),
		describe:   describe,
		stackMaps:  table,
	}
	return registerUserFrameRegion(r)
}

//go:linkname jit_addUserFrameStackMaps runtime/jit.addUserFrameStackMaps
func jit_addUserFrameStackMaps(handle uintptr, stackMaps []userFrameStackMapInput) {
	r := userFrameByHandle[handle]
	if r == nil {
		throw("addUserFrameStackMaps: handle not found")
	}
	appendUserFrameStackMaps(r.stackMaps, r.start, r.end, r.unwindMode, stackMaps)
}

//go:linkname jit_unregisterUserFrameRegion runtime/jit.unregisterUserFrameRegion
func jit_unregisterUserFrameRegion(handle uintptr) {
	unregisterUserFrameRegion(handle)
}

//go:linkname jit_userFramePreempt runtime/jit.userFramePreempt
//go:nosplit
func jit_userFramePreempt() bool {
	return userFramePreempt()
}

//go:linkname jit_userFrameStackCheck runtime/jit.userFrameStackCheck
//go:nosplit
func jit_userFrameStackCheck() (guardOffset, stackSmall, moreStackPC uintptr) {
	return unsafe.Offsetof(g{}.stackguard0), abi.StackSmall, abi.FuncPCABI0(morestack)
}

//go:linkname jit_userFrameWriteBarrier runtime/jit.userFrameWriteBarrier
func jit_userFrameWriteBarrier() uintptr { return uintptr(unsafe.Pointer(&writeBarrier)) }

//go:linkname jit_userFramePreemptOffset runtime/jit.userFramePreemptOffset
func jit_userFramePreemptOffset() uintptr { return unsafe.Offsetof(g{}.preempt) }

// badUserFrame identifies a malformed active map without allocating or calling
// JIT code from the collector or stack copier.
func badUserFrame(reason string, pc, sp, base, words uintptr) {
	printlock()
	print("runtime: JIT ", reason, " pc=", hex(pc), " sp=", hex(sp), " map-base=", hex(base), " words=", words)
	if r := findUserFrameRegion(pc); r != nil {
		print(" region=[", hex(r.start), ",", hex(r.end), ") pc-offset=", hex(pc-r.start))
	}
	println()
	printunlock()
	throw("invalid JIT frame metadata")
}
