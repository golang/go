// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

// Package jit provides support for registering user code frames with
// the Go runtime, enabling JIT compilers, WASM engines, and other code
// generators to interoperate safely with Go's stack unwinder, panic/recover
// mechanism, and garbage collector.
//
// A user frame region is a range of executable memory containing code that
// was not produced by the Go compiler. Without registration, any Go operation
// that inspects the call stack (panic, recover, runtime.Callers, GC stack
// scanning) will crash if it encounters a PC from such a region.
//
// Registering a region makes the runtime aware of the user code and tells
// it how to handle frames in that code during stack unwinding and GC.
//
// # Unwind modes
//
// When the runtime encounters a frame whose PC falls in a registered
// region, it consults the region's unwind mode:
//
//   - [UnwindStop]: The traceback ends at the user frame boundary.
//   - [UnwindSkip]: The user frame is skipped and the traceback continues
//     past it using the registered safepoint's unwind recipe.
//   - [UnwindDeclare]: Like UnwindSkip, but every user frame is also exposed
//     as a regular logical frame by runtime.Stack and runtime.CallersFrames,
//     using the provided Describe callback for symbols and source positions.
//
// # GC integration
//
// If user code holds Go pointers in an active frame, StackMaps must describe
// those slots precisely. The runtime uses the same map both
// while marking the goroutine stack and while relocating a growing stack.
// Failure to describe a live pointer may cause the GC to collect its target
// or leave a stale pointer after stack growth.
package jit

import (
	"internal/abi"
	"sync"
	"unsafe"
)

// TailType is a prepared GC allocation layout for a struct whose final field
// is a zero-length array. It describes the struct prefix once followed by a
// caller-selected number of array elements.
type TailType struct {
	runtimeType *abi.Type
}

type tailTypeKey struct {
	base  *abi.Type
	count uintptr
}

var tailTypes sync.Map

// TailTypeFor prepares the allocation layout T with count repetitions of the
// element type of T's final [0]E field. Preparation is intended to happen when
// JIT code is compiled; allocating the prepared type is an O(1) operation.
func TailTypeFor[T any](count uintptr) TailType {
	base := abi.TypeFor[T]()
	key := tailTypeKey{base: base, count: count}
	if cached, ok := tailTypes.Load(key); ok {
		return TailType{runtimeType: cached.(*abi.Type)}
	}
	if base.Kind() != abi.Struct {
		panic("runtime/jit: tail allocation base is not a struct")
	}
	baseStruct := base.StructType()
	if len(baseStruct.Fields) == 0 {
		panic("runtime/jit: tail allocation struct has no fields")
	}
	last := baseStruct.Fields[len(baseStruct.Fields)-1]
	if last.Typ.Kind() != abi.Array || last.Typ.Len() != 0 {
		panic("runtime/jit: tail allocation struct must end in [0]E")
	}
	element := last.Typ.Elem()
	if element.Size_ != 0 && count > ^uintptr(0)/element.Size_ {
		panic("runtime/jit: tail allocation size overflow")
	}
	tailSize := element.Size_ * count
	if last.Offset > ^uintptr(0)-tailSize {
		panic("runtime/jit: tail allocation size overflow")
	}

	array := new(abi.ArrayType)
	array.Type.Size_ = tailSize
	array.Type.Align_ = element.Align_
	array.Type.FieldAlign_ = element.FieldAlign_
	array.Type.Kind_ = abi.Array
	array.Elem = element
	array.Len = count
	if count != 0 && element.PtrBytes != 0 {
		array.Type.PtrBytes = (count-1)*element.Size_ + element.PtrBytes
		array.Type.TFlag = abi.TFlagGCMaskOnDemand
		mask := new(*byte)
		array.Type.GCData = (*byte)(unsafe.Pointer(mask))
	}

	exactSize := last.Offset + tailSize
	if count == 0 {
		// Go gives a final zero-sized field a distinct address by retaining
		// trailing padding in the base struct. Preserve that layout when the
		// repeated tail is empty.
		exactSize = base.Size_
	} else if align := uintptr(base.Align_); exactSize%align != 0 {
		padding := align - exactSize%align
		if exactSize > ^uintptr(0)-padding {
			panic("runtime/jit: tail allocation size overflow")
		}
		exactSize += padding
	}
	exact := new(abi.StructType)
	exact.Type.Size_ = exactSize
	exact.Type.Align_ = base.Align_
	exact.Type.FieldAlign_ = base.FieldAlign_
	exact.Type.Kind_ = abi.Struct
	exact.Fields = append([]abi.StructField(nil), baseStruct.Fields...)
	exact.Fields[len(exact.Fields)-1].Typ = &array.Type
	exact.Type.PtrBytes = base.PtrBytes
	if array.PtrBytes != 0 {
		exact.Type.PtrBytes = last.Offset + array.PtrBytes
	}
	if exact.Type.PtrBytes != 0 {
		exact.Type.TFlag = abi.TFlagGCMaskOnDemand
		mask := new(*byte)
		exact.Type.GCData = (*byte)(unsafe.Pointer(mask))
	}
	actual, _ := tailTypes.LoadOrStore(key, &exact.Type)
	return TailType{runtimeType: actual.(*abi.Type)}
}

// Alloc returns one zeroed, precisely typed Go heap object.
func (t TailType) Alloc() unsafe.Pointer {
	if t.runtimeType == nil {
		panic("runtime/jit: uninitialized TailType")
	}
	return mallocgc(t.runtimeType.Size_, t.runtimeType, true)
}

// UnwindMode controls how the Go runtime handles user frames during
// stack unwinding (tracebacks, panics, GC).
type UnwindMode uint8

const (
	// UnwindStop ends the traceback at the user frame boundary.
	UnwindStop UnwindMode = iota

	// UnwindSkip skips user frames and continues unwinding using the
	// safepoint metadata. The user frames do not appear in tracebacks.
	UnwindSkip

	// UnwindDeclare is like UnwindSkip but also exposes user frames as regular
	// logical stack entries in runtime.Stack and runtime.CallersFrames, using
	// the Describe callback for symbols and source positions.
	UnwindDeclare
)

// StackMap describes pointer-bearing words in a user frame at one safepoint.
// PointerMask has one bit per frame word in least-significant-bit-first order;
// a set bit denotes a Go pointer. The runtime copies maps passed to Register
// or AddStackMaps, so callers may release or reuse the input afterward.
type StackMap struct {
	// PCOffset is the return-PC offset from Region.Start. Entries in a Region
	// must be strictly ordered by PCOffset.
	PCOffset uintptr

	// FrameOffset is added to the user frame SP to locate the first word
	// described by PointerMask.
	FrameOffset uintptr

	// FrameWords is the number of pointer-sized words described by the map.
	FrameWords uintptr

	PointerMask []byte

	// HasUnwind enables the declarative unwind recipe below. base starts at
	// sp+UnwindBaseOffset. If UnwindBaseUsesDelta is set, the uintptr stored at
	// sp+UnwindBaseDeltaOffset is added to base. CallerPC and CallerBP are read
	// from base plus their offsets; CallerSP is base+CallerSPOffset.
	HasUnwind             bool
	UnwindBaseOffset      uintptr
	UnwindBaseDeltaOffset uintptr
	UnwindBaseUsesDelta   bool
	CallerPCOffset        uintptr
	CallerSPOffset        uintptr
	CallerBPOffset        uintptr
}

// Region describes a range of executable memory containing user code.
type Region struct {
	// Start is the first byte of the code region (inclusive).
	Start uintptr

	// End is one past the last byte of the code region (exclusive).
	End uintptr

	// Unwind controls the behavior when the runtime encounters a
	// frame in this region during stack unwinding.
	Unwind UnwindMode

	// Describe returns human-readable information about a PC in this
	// region for use in tracebacks. Only called when Unwind is
	// UnwindDeclare. Must not allocate Go memory. May be nil.
	Describe func(pc uintptr) (name, file string, line int, ok bool)

	// StackMaps contains immutable GC and optional unwind metadata for
	// safepoints in this region. The runtime performs lookup itself without
	// calling user code. Declarative unwind recipes avoid executing instrumented
	// Go callbacks from the runtime system stack.
	// It may initially be nil when maps are published with Handle.AddStackMaps
	// before the corresponding code becomes reachable.
	StackMaps []StackMap
}

// Handle represents a registered user frame region.
type Handle struct {
	handle uintptr
}

// StackCheck describes the runtime values needed to emit the same split-stack
// prologue as a Go function. The instruction sequence and the register used to
// preserve closure context across MoreStackPC are architecture-specific.
type StackCheck struct {
	StackGuardOffset uintptr
	StackSmall       uintptr
	MoreStackPC      uintptr
}

// StackCheckConfig returns immutable process-wide metadata for emitting a
// split-stack check. MoreStackPC is runtime.morestack, not an ordinary
// returning function: generated code must follow the platform's Go prologue
// ABI and resume by retrying its stack check.
func StackCheckConfig() StackCheck {
	guardOffset, stackSmall, moreStackPC := userFrameStackCheck()
	return StackCheck{
		StackGuardOffset: guardOffset,
		StackSmall:       stackSmall,
		MoreStackPC:      moreStackPC,
	}
}

// AddStackMaps appends safepoints to a registered region. Entries must be
// strictly ordered after all previously published entries. The runtime copies
// maps and masks before publishing them to lock-free stack walkers.
//
// JIT compilers must call AddStackMaps after the corresponding machine code is
// complete and before making that code reachable by another goroutine.
func (h Handle) AddStackMaps(stackMaps ...StackMap) {
	addUserFrameStackMaps(h.handle, stackMaps)
}

// liveRegions keeps function pointers (closures) alive so the GC does
// not collect them. The runtime stores these pointers in persistentalloc
// memory that is invisible to the GC, so we must retain a reference here.
var (
	liveRegionsMu sync.Mutex
	liveRegions   map[uintptr]Region
)

// Register makes the runtime aware of a user code region.
func Register(r Region) Handle {
	h := registerUserFrameRegion(
		r.Start, r.End,
		uint8(r.Unwind),
		r.Describe,
		r.StackMaps,
	)

	// Keep closures alive for the GC.
	liveRegionsMu.Lock()
	if liveRegions == nil {
		liveRegions = make(map[uintptr]Region)
	}
	liveRegions[h] = r
	liveRegionsMu.Unlock()

	return Handle{handle: h}
}

// Unregister removes the user frame region from the runtime. The caller must
// ensure that no goroutine is executing code in the region, that no user frame
// from the region remains on a goroutine stack, and that the code cannot be
// entered again.
func (h Handle) Unregister() {
	unregisterUserFrameRegion(h.handle)

	liveRegionsMu.Lock()
	delete(liveRegions, h.handle)
	liveRegionsMu.Unlock()
}

// Preempt reports whether the Go runtime is requesting the current
// goroutine to yield (e.g., for garbage collection or scheduling).
//
// JIT-compiled code should call Preempt at regular safepoints such as
// loop back-edges and function entries. When Preempt returns true, the
// JIT code should return to its Go caller as soon as possible so the
// runtime can perform pending operations (GC stop-the-world, goroutine
// preemption, etc.).
//
// JIT code that frequently calls Go functions may not need explicit
// Preempt checks, because Go function prologues already check for
// preemption requests. Preempt is primarily useful for long-running
// JIT loops that do not call back into Go.
//
// Preempt is very lightweight: it reads a single flag from the current
// goroutine with no synchronization overhead.
func Preempt() bool {
	return userFramePreempt()
}

//go:linkname userFramePreempt runtime/jit.userFramePreempt
func userFramePreempt() bool

// mallocgc is private runtime ABI. runtime/jit is part of the same standard
// library revision, so this dependency stays inside the package shipped with
// that runtime instead of leaking into JIT clients.
//
//go:linkname mallocgc runtime.mallocgc
func mallocgc(size uintptr, typ *abi.Type, needzero bool) unsafe.Pointer

//go:linkname userFrameStackCheck runtime/jit.userFrameStackCheck
func userFrameStackCheck() (guardOffset, stackSmall, moreStackPC uintptr)

//go:linkname registerUserFrameRegion runtime/jit.registerUserFrameRegion
func registerUserFrameRegion(start, end uintptr, unwindMode uint8,
	describe func(pc uintptr) (string, string, int, bool),
	stackMaps []StackMap,
) uintptr

//go:linkname unregisterUserFrameRegion runtime/jit.unregisterUserFrameRegion
func unregisterUserFrameRegion(handle uintptr)

//go:linkname addUserFrameStackMaps runtime/jit.addUserFrameStackMaps
func addUserFrameStackMaps(handle uintptr, stackMaps []StackMap)
