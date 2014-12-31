// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

import "unsafe"

var Fadd64 = fadd64
var Fsub64 = fsub64
var Fmul64 = fmul64
var Fdiv64 = fdiv64
var F64to32 = f64to32
var F32to64 = f32to64
var Fcmp64 = fcmp64
var Fintto64 = fintto64
var F64toint = f64toint

var Entersyscall = entersyscall
var Exitsyscall = exitsyscall
var LockedOSThread = lockedOSThread

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
	body    *byte
	done    uint32
	Nthr    uint32
	nthrmax uint32
	thrseq  uint32
	Cnt     uint32
	Ctx     *byte
	wait    bool
}

func NewParFor(nthrmax uint32) *ParFor {
	var desc *ParFor
	systemstack(func() {
		desc = (*ParFor)(unsafe.Pointer(parforalloc(nthrmax)))
	})
	return desc
}

func ParForSetup(desc *ParFor, nthr, n uint32, ctx *byte, wait bool, body func(*ParFor, uint32)) {
	systemstack(func() {
		parforsetup((*parfor)(unsafe.Pointer(desc)), nthr, n, unsafe.Pointer(ctx), wait,
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
	pos := desc_thr_index(desc1, tid).pos
	return uint32(pos), uint32(pos >> 32)
}

func GCMask(x interface{}) (ret []byte) {
	e := (*eface)(unsafe.Pointer(&x))
	s := (*slice)(unsafe.Pointer(&ret))
	systemstack(func() {
		var len uintptr
		getgcmask(e.data, e._type, &s.array, &len)
		s.len = uint(len)
		s.cap = s.len
	})
	return
}

func RunSchedLocalQueueTest() {
	systemstack(testSchedLocalQueue)
}
func RunSchedLocalQueueStealTest() {
	systemstack(testSchedLocalQueueSteal)
}

var StringHash = stringHash
var BytesHash = bytesHash
var Int32Hash = int32Hash
var Int64Hash = int64Hash
var EfaceHash = efaceHash
var IfaceHash = ifaceHash
var MemclrBytes = memclrBytes

var HashLoad = &hashLoad

// For testing.
func GogoBytes() int32 {
	return _RuntimeGogoBytes
}

// entry point for testing
func GostringW(w []uint16) (s string) {
	systemstack(func() {
		s = gostringw(&w[0])
	})
	return
}

var Gostringnocopy = gostringnocopy
var Maxstring = &maxstring
