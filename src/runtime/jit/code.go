// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit

import (
	"fmt"
	"reflect"
	"runtime"
	"sync/atomic"
	"unsafe"
)

// CodeBuffer writes unpublished code into a fixed, caller-owned address range.
// Offset counts bytes from the beginning of that range. Write either writes all
// bytes or panics with CodeBufferFull before changing bytes or Offset. Patch
// changes only previously written bytes and does not change Offset. Invalid
// patches are programming errors, not CodeBufferFull. Implementations must not
// silently grow, move, publish, or execute the code. The caller owns permissions,
// instruction-cache synchronization, registration and lifetime of the mapping.
type CodeBuffer interface {
	Offset() int
	Write(code []byte)
	Patch(offset int, code []byte)
}

// CodeBufferFull requests a larger contiguous write area. Recovering does not
// resume the failed emitter. Retry the compilation unit after rolling back its
// bytes, labels, roots and metadata. No code may be published before completion.
// A buffer which links separate mappings must additionally translate PCs and
// relocations and reserve its own inter-segment jump space.
type CodeBufferFull struct{ Offset, Required, Available int }

func (e CodeBufferFull) Error() string {
	return fmt.Sprintf("runtime/jit: code buffer full at byte %d: need %d, have %d", e.Offset, e.Required, e.Available)
}

// SliceBuffer adapts caller-owned writable memory without reallocating it.
type SliceBuffer struct {
	Bytes    []byte
	Position int
}

func (b *SliceBuffer) Offset() int { return b.Position }
func (b *SliceBuffer) Write(code []byte) {
	if b.Position < 0 || b.Position > len(b.Bytes) {
		panic("runtime/jit: invalid buffer position")
	}
	if len(code) > len(b.Bytes)-b.Position {
		panic(CodeBufferFull{b.Position, len(code), len(b.Bytes) - b.Position})
	}
	copy(b.Bytes[b.Position:], code)
	b.Position += len(code)
}
func (b *SliceBuffer) Patch(offset int, code []byte) {
	if offset < 0 || offset > b.Position || len(code) > b.Position-offset {
		panic("runtime/jit: patch outside written code")
	}
	copy(b.Bytes[offset:], code)
}

// FunctionHeader is the toolchain-owned prefix of a JIT closure context. Embed
// it first in an ordinary typed Go struct (possibly a prepared TailType). Keep
// its complete allocation reachable; neither executable bytes nor uintptr
// constants are GC roots. Initialize pointer fields through Go writes/barriers
// before publication. A stack context must not escape its owning invocation or
// goroutine and all its live pointer fields must be in that frame's stack map.
// The prefix layout is deliberately owned here, not re-created by clients.
type FunctionHeader struct{ entry uintptr }

// SetEntry publishes code after all its metadata and dependencies are ready.
// Concurrent readers must use Entry. Clearing an entry does not revoke copies
// of an already-created function value or authorize Unregister.
func (h *FunctionHeader) SetEntry(pc uintptr) { atomic.StoreUintptr(&h.entry, pc) }

// Entry returns the current code entry.
func (h *FunctionHeader) Entry() uintptr { return atomic.LoadUintptr(&h.entry) }

// Function creates a normally callable Go function value of type F whose
// closure context is h's complete allocation. The entry must implement F's
// ABIInternal signature. F is checked here, but machine-code conformance and
// the enclosing allocation's pointer layout cannot be checked. A zero entry
// is allowed while privately constructing a function before publication.
func Function[F any](h *FunctionHeader) F {
	if reflect.TypeFor[F]().Kind() != reflect.Func {
		panic("runtime/jit: Function requires a function type")
	}
	p := unsafe.Pointer(h)
	return *(*F)(unsafe.Pointer(&p))
}

// FunctionContext obtains a function value's closure context without exposing
// Go interface representation to its caller. It also works for ordinary Go
// closures. Callers must keep fn alive while using the returned pointer.
func FunctionContext[F any](fn F) unsafe.Pointer {
	if reflect.TypeFor[F]().Kind() != reflect.Func {
		panic("runtime/jit: FunctionContext requires a function type")
	}
	return *(*unsafe.Pointer)(unsafe.Pointer(&fn))
}

// FunctionEntryOffset is the code word offset used by closure-call emitters.
func FunctionEntryOffset() uintptr { return unsafe.Offsetof(FunctionHeader{}.entry) }

// SupportedEmitters reports whether this toolchain implements ABI code emission
// for its native architecture. Registration itself supports additional targets.
func SupportedEmitters() bool {
	return runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64" || runtime.GOARCH == "riscv64"
}
