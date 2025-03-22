// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build valgrind && linux && (arm64 || amd64)

package runtime

import "unsafe"

const valgrindenabled = true

// Valgrind provides a mechanism to allow programs under test to modify
// Valgrinds behavior in certain ways, referred to as client requests [0]. These
// requests are triggered putting the address of a series of uints in a specific
// register and emitting a very specific sequence of assembly instructions. The
// result of the request (if there is one) is then put in another register for
// the program to retrieve. Each request is identified by a unique uint, which
// is passed as the first "argument".
//
// Valgrind provides headers (valgrind/valgrind.h, valgrind/memcheck.h) with
// macros that emit the correct assembly for these requests. Instead of copying
// these headers into the tree and using cgo to call the macros, we implement
// the client request assembly ourselves. Since both the magic instruction
// sequences, and the request uint's are stable, it is safe for us to implement.
//
// The client requests we add are used to describe our memory allocator to
// Valgrind, per [1]. We describe the allocator using the two-level mempool
// structure a We also add annotations which allow Valgrind to track which
// memory we use for stacks, which seems necessary to let it properly function.
//
// We describe the memory model to Valgrind as follows: we treat heap arenas as
// "pools" created with VALGRIND_CREATE_MEMPOOL_EXT (so that we can use
// VALGRIND_MEMPOOL_METAPOOL and VALGRIND_MEMPOOL_AUTO_FREE). Within the pool we
// treat spans as "superblocks", annotated with VALGRIND_MEMPOOL_ALLOC. We then
// allocate individual objects within spans with VALGRIND_MALLOCLIKE_BLOCK.
//
// [0] https://valgrind.org/docs/manual/manual-core-adv.html#manual-core-adv.clientreq
// [1] https://valgrind.org/docs/manual/mc-manual.html#mc-manual.mempools

const (
	// Valgrind request IDs, copied from valgrind/valgrind.h.
	vg_userreq__malloclike_block = 0x1301
	vg_userreq__freelike_block   = 0x1302
	vg_userreq__create_mempool   = 0x1303
	vg_userreq__mempool_alloc    = 0x1305
	vg_userreq__mempool_free     = 0x1306
	vg_userreq__stack_register   = 0x1501
	vg_userreq__stack_deregister = 0x1502
	vg_userreq__stack_change     = 0x1503
)

const (
	// Memcheck request IDs are offset from ('M'&0xff) << 24 | ('C'&0xff) << 16, or 0x4d430000,
	// copied from valgrind/memcheck.h.
	vg_userreq__make_mem_noaccess = iota + ('M'&0xff)<<24 | ('C'&0xff)<<16
	vg_userreq__make_mem_undefined
	vg_userreq__make_mem_defined
)

const (
	// VALGRIND_CREATE_MEMPOOL_EXT flags, copied from valgrind/valgrind.h.
	valgrind_mempool_auto_free = 1
	valgrind_mempool_metapool  = 2
)

//

//go:noescape
func valgrindClientRequest(uintptr, uintptr, uintptr, uintptr, uintptr, uintptr) uintptr

//go:nosplit
func valgrindRegisterStack(start, end unsafe.Pointer) uintptr {
	// VALGRIND_STACK_REGISTER
	return valgrindClientRequest(vg_userreq__stack_register, uintptr(start), uintptr(end), 0, 0, 0)
}

//go:nosplit
func valgrindDeregisterStack(id uintptr) {
	// VALGRIND_STACK_DEREGISTER
	valgrindClientRequest(vg_userreq__stack_deregister, id, 0, 0, 0, 0)
}

//go:nosplit
func valgrindChangeStack(id uintptr, start, end unsafe.Pointer) {
	// VALGRIND_STACK_CHANGE
	valgrindClientRequest(vg_userreq__stack_change, id, uintptr(start), uintptr(end), 0, 0)
}

//go:nosplit
func valgrindMalloc(addr unsafe.Pointer, size uintptr) {
	// VALGRIND_MALLOCLIKE_BLOCK
	valgrindClientRequest(vg_userreq__malloclike_block, uintptr(addr), size, 0, 1, 0)
}

//go:nosplit
func valgrindFree(addr unsafe.Pointer) {
	// VALGRIND_FREELIKE_BLOCK
	valgrindClientRequest(vg_userreq__freelike_block, uintptr(addr), 0, 0, 0, 0)
}

//go:nosplit
func valgrindCreateMempool(addr unsafe.Pointer) {
	// VALGRIND_CREATE_MEMPOOL_EXT
	valgrindClientRequest(vg_userreq__create_mempool, uintptr(addr), 0, 1, valgrind_mempool_auto_free|valgrind_mempool_metapool, 0)
}

//go:nosplit
func valgrindMempoolMalloc(pool, addr unsafe.Pointer, size uintptr) {
	// VALGRIND_MEMPOOL_ALLOC
	valgrindClientRequest(vg_userreq__mempool_alloc, uintptr(pool), uintptr(addr), size, 0, 0)
}

//go:nosplit
func valgrindMempoolFree(pool, addr unsafe.Pointer) {
	// VALGRIND_MEMPOOL_FREE
	valgrindClientRequest(vg_userreq__mempool_free, uintptr(pool), uintptr(addr), 0, 0, 0)
}

// Memcheck client requests, copied from valgrind/memcheck.h

//go:nosplit
func valgrindMakeMemUndefined(addr unsafe.Pointer, size uintptr) {
	// VALGRIND_MAKE_MEM_UNDEFINED
	valgrindClientRequest(vg_userreq__make_mem_undefined, uintptr(addr), size, 0, 0, 0)
}

//go:nosplit
func valgrindMakeMemDefined(addr unsafe.Pointer, size uintptr) {
	// VALGRIND_MAKE_MEM_DEFINED
	valgrindClientRequest(vg_userreq__make_mem_defined, uintptr(addr), size, 0, 0, 0)
}

//go:nosplit
func valgrindMakeMemNoAccess(addr unsafe.Pointer, size uintptr) {
	// VALGRIND_MAKE_MEM_NOACCESS
	valgrindClientRequest(vg_userreq__make_mem_noaccess, uintptr(addr), size, 0, 0, 0)
}
