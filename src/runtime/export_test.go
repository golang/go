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

// in asm_*.s
func stackguard() (sp, limit uintptr)

var Entersyscall = entersyscall
var Exitsyscall = exitsyscall
var LockedOSThread = lockedOSThread

type LFNode struct {
	Next    *LFNode
	Pushcnt uintptr
}

func lfstackpush_m()
func lfstackpop_m()

func LFStackPush(head *uint64, node *LFNode) {
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(head)
	mp.ptrarg[1] = unsafe.Pointer(node)
	onM(lfstackpush_m)
	releasem(mp)
}

func LFStackPop(head *uint64) *LFNode {
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(head)
	onM(lfstackpop_m)
	node := (*LFNode)(unsafe.Pointer(mp.ptrarg[0]))
	mp.ptrarg[0] = nil
	releasem(mp)
	return node
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

func newparfor_m()
func parforsetup_m()
func parfordo_m()
func parforiters_m()

func NewParFor(nthrmax uint32) *ParFor {
	mp := acquirem()
	mp.scalararg[0] = uintptr(nthrmax)
	onM(newparfor_m)
	desc := (*ParFor)(mp.ptrarg[0])
	mp.ptrarg[0] = nil
	releasem(mp)
	return desc
}

func ParForSetup(desc *ParFor, nthr, n uint32, ctx *byte, wait bool, body func(*ParFor, uint32)) {
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(desc)
	mp.ptrarg[1] = unsafe.Pointer(ctx)
	mp.ptrarg[2] = unsafe.Pointer(funcPC(body)) // TODO(rsc): Should be a scalar.
	mp.scalararg[0] = uintptr(nthr)
	mp.scalararg[1] = uintptr(n)
	mp.scalararg[2] = 0
	if wait {
		mp.scalararg[2] = 1
	}
	onM(parforsetup_m)
	releasem(mp)
}

func ParForDo(desc *ParFor) {
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(desc)
	onM(parfordo_m)
	releasem(mp)
}

func ParForIters(desc *ParFor, tid uint32) (uint32, uint32) {
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(desc)
	mp.scalararg[0] = uintptr(tid)
	onM(parforiters_m)
	begin := uint32(mp.scalararg[0])
	end := uint32(mp.scalararg[1])
	releasem(mp)
	return begin, end
}

// in mgc0.c
//go:noescape
func getgcmask(data unsafe.Pointer, typ *_type, array **byte, len *uint)

func GCMask(x interface{}) (ret []byte) {
	e := (*eface)(unsafe.Pointer(&x))
	s := (*slice)(unsafe.Pointer(&ret))
	onM(func() {
		getgcmask(e.data, e._type, &s.array, &s.len)
		s.cap = s.len
	})
	return
}

func testSchedLocalQueue()
func testSchedLocalQueueSteal()
func RunSchedLocalQueueTest() {
	onM(testSchedLocalQueue)
}
func RunSchedLocalQueueStealTest() {
	onM(testSchedLocalQueueSteal)
}

var HaveGoodHash = haveGoodHash
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

// in string.c
//go:noescape
func gostringw(w *uint16) string

// entry point for testing
func GostringW(w []uint16) (s string) {
	onM(func() {
		s = gostringw(&w[0])
	})
	return
}

var Gostringnocopy = gostringnocopy
var Maxstring = &maxstring
