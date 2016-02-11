// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

var Fadd64 = fadd64
var Fsub64 = fsub64
var Fmul64 = fmul64
var Fdiv64 = fdiv64
var F64to32 = f64to32
var F32to64 = f32to64
var Fcmp64 = fcmp64
var Fintto64 = fintto64
var F64toint = f64toint
var Sqrt = sqrt

var Entersyscall = entersyscall
var Exitsyscall = exitsyscall
var LockedOSThread = lockedOSThread
var Xadduintptr = atomic.Xadduintptr

var FuncPC = funcPC

var Fastlog2 = fastlog2

type LFNode struct {
	Next    uint64
	Pushcnt uintptr
}

func LFStackPush(head *uint64, node *LFNode) {
	lfstackpush(head, (*lfnode)(unsafe.Pointer(node)))
}

func LFStackPop(head *uint64) *LFNode {
	return (*LFNode)(unsafe.Pointer(lfstackpop(head)))
}

type ParFor struct {
	body   func(*ParFor, uint32)
	done   uint32
	Nthr   uint32
	thrseq uint32
	Cnt    uint32
	wait   bool
}

func NewParFor(nthrmax uint32) *ParFor {
	var desc *ParFor
	systemstack(func() {
		desc = (*ParFor)(unsafe.Pointer(parforalloc(nthrmax)))
	})
	return desc
}

func ParForSetup(desc *ParFor, nthr, n uint32, wait bool, body func(*ParFor, uint32)) {
	systemstack(func() {
		parforsetup((*parfor)(unsafe.Pointer(desc)), nthr, n, wait,
			*(*func(*parfor, uint32))(unsafe.Pointer(&body)))
	})
}

func ParForDo(desc *ParFor) {
	systemstack(func() {
		parfordo((*parfor)(unsafe.Pointer(desc)))
	})
}

func ParForIters(desc *ParFor, tid uint32) (uint32, uint32) {
	desc1 := (*parfor)(unsafe.Pointer(desc))
	pos := desc1.thr[tid].pos
	return uint32(pos), uint32(pos >> 32)
}

func GCMask(x interface{}) (ret []byte) {
	systemstack(func() {
		ret = getgcmask(x)
	})
	return
}

func RunSchedLocalQueueTest() {
	testSchedLocalQueue()
}
func RunSchedLocalQueueStealTest() {
	testSchedLocalQueueSteal()
}

var StringHash = stringHash
var BytesHash = bytesHash
var Int32Hash = int32Hash
var Int64Hash = int64Hash
var EfaceHash = efaceHash
var IfaceHash = ifaceHash
var MemclrBytes = memclrBytes

var HashLoad = &hashLoad

// entry point for testing
func GostringW(w []uint16) (s string) {
	systemstack(func() {
		s = gostringw(&w[0])
	})
	return
}

var Gostringnocopy = gostringnocopy
var Maxstring = &maxstring

type Uintreg sys.Uintreg

var Open = open
var Close = closefd
var Read = read
var Write = write

func Envs() []string     { return envs }
func SetEnvs(e []string) { envs = e }

var BigEndian = sys.BigEndian

// For benchmarking.

func BenchSetType(n int, x interface{}) {
	e := *efaceOf(&x)
	t := e._type
	var size uintptr
	var p unsafe.Pointer
	switch t.kind & kindMask {
	case kindPtr:
		t = (*ptrtype)(unsafe.Pointer(t)).elem
		size = t.size
		p = e.data
	case kindSlice:
		slice := *(*struct {
			ptr      unsafe.Pointer
			len, cap uintptr
		})(e.data)
		t = (*slicetype)(unsafe.Pointer(t)).elem
		size = t.size * slice.len
		p = slice.ptr
	}
	allocSize := roundupsize(size)
	systemstack(func() {
		for i := 0; i < n; i++ {
			heapBitsSetType(uintptr(p), allocSize, size, t)
		}
	})
}

const PtrSize = sys.PtrSize

var TestingAssertE2I2GC = &testingAssertE2I2GC
var TestingAssertE2T2GC = &testingAssertE2T2GC

var ForceGCPeriod = &forcegcperiod

// SetTracebackEnv is like runtime/debug.SetTraceback, but it raises
// the "environment" traceback level, so later calls to
// debug.SetTraceback (e.g., from testing timeouts) can't lower it.
func SetTracebackEnv(level string) {
	setTraceback(level)
	traceback_env = traceback_cache
}
