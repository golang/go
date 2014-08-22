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

func entersyscall()
func golockedOSThread() bool
func stackguard() (sp, limit uintptr)

var Entersyscall = entersyscall
var Exitsyscall = exitsyscall
var LockedOSThread = golockedOSThread
var Stackguard = stackguard

type LFNode struct {
	Next    *LFNode
	Pushcnt uintptr
}

var (
	lfstackpush_m,
	lfstackpop_m mFunction
)

func LFStackPush(head *uint64, node *LFNode) {
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(head)
	mp.ptrarg[1] = unsafe.Pointer(node)
	onM(&lfstackpush_m)
	releasem(mp)
}

func LFStackPop(head *uint64) *LFNode {
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(head)
	onM(&lfstackpop_m)
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

func newParFor(nthrmax uint32) *ParFor
func parForSetup(desc *ParFor, nthr, n uint32, ctx *byte, wait bool, body func(*ParFor, uint32))
func parForDo(desc *ParFor)
func parForIters(desc *ParFor, tid uintptr) (uintptr, uintptr)

var NewParFor = newParFor
var ParForSetup = parForSetup
var ParForDo = parForDo

func ParForIters(desc *ParFor, tid uint32) (uint32, uint32) {
	begin, end := parForIters(desc, uintptr(tid))
	return uint32(begin), uint32(end)
}

//go:noescape
func GCMask(x interface{}) []byte

func testSchedLocalQueue()
func testSchedLocalQueueSteal()

var TestSchedLocalQueue1 = testSchedLocalQueue
var TestSchedLocalQueueSteal1 = testSchedLocalQueueSteal

var HaveGoodHash = haveGoodHash
var StringHash = stringHash
var BytesHash = bytesHash
var Int32Hash = int32Hash
var Int64Hash = int64Hash
var EfaceHash = efaceHash
var IfaceHash = ifaceHash
var MemclrBytes = memclrBytes

var HashLoad = &hashLoad

func gogoBytes() int32

var GogoBytes = gogoBytes

func gostringW([]uint16) string

var GostringW = gostringW
