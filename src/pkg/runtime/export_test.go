// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

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
func exitsyscall()
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

func lfstackpush_go(head *uint64, node *LFNode)
func lfstackpop_go(head *uint64) *LFNode

var LFStackPush = lfstackpush_go
var LFStackPop = lfstackpop_go

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

func haveGoodHash() bool
func stringHash(s string, seed uintptr) uintptr
func bytesHash(b []byte, seed uintptr) uintptr
func int32Hash(i uint32, seed uintptr) uintptr
func int64Hash(i uint64, seed uintptr) uintptr

var HaveGoodHash = haveGoodHash
var StringHash = stringHash
var BytesHash = bytesHash
var Int32Hash = int32Hash
var Int64Hash = int64Hash

var HashLoad = &hashLoad

func memclrBytes(b []byte)

var MemclrBytes = memclrBytes

func gogoBytes() int32

var GogoBytes = gogoBytes

func gostringW([]uint16) string

var GostringW = gostringW
