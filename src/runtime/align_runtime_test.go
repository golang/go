// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file lives in the runtime package
// so we can get access to the runtime guts.
// The rest of the implementation of this test is in align_test.go.

package runtime

import "unsafe"

// AtomicFields is the set of fields on which we perform 64-bit atomic
// operations (all the *64 operations in runtime/internal/atomic).
var AtomicFields = []uintptr{
	unsafe.Offsetof(m{}.procid),
	unsafe.Offsetof(p{}.gcFractionalMarkTime),
	unsafe.Offsetof(profBuf{}.overflow),
	unsafe.Offsetof(profBuf{}.overflowTime),
	unsafe.Offsetof(heapStatsDelta{}.tinyAllocCount),
	unsafe.Offsetof(heapStatsDelta{}.smallAllocCount),
	unsafe.Offsetof(heapStatsDelta{}.smallFreeCount),
	unsafe.Offsetof(heapStatsDelta{}.largeAlloc),
	unsafe.Offsetof(heapStatsDelta{}.largeAllocCount),
	unsafe.Offsetof(heapStatsDelta{}.largeFree),
	unsafe.Offsetof(heapStatsDelta{}.largeFreeCount),
	unsafe.Offsetof(heapStatsDelta{}.committed),
	unsafe.Offsetof(heapStatsDelta{}.released),
	unsafe.Offsetof(heapStatsDelta{}.inHeap),
	unsafe.Offsetof(heapStatsDelta{}.inStacks),
	unsafe.Offsetof(heapStatsDelta{}.inPtrScalarBits),
	unsafe.Offsetof(heapStatsDelta{}.inWorkBufs),
	unsafe.Offsetof(lfnode{}.next),
	unsafe.Offsetof(mstats{}.last_gc_nanotime),
	unsafe.Offsetof(mstats{}.last_gc_unix),
	unsafe.Offsetof(workType{}.bytesMarked),
}

// AtomicVariables is the set of global variables on which we perform
// 64-bit atomic operations.
var AtomicVariables = []unsafe.Pointer{
	unsafe.Pointer(&ncgocall),
	unsafe.Pointer(&test_z64),
	unsafe.Pointer(&blockprofilerate),
	unsafe.Pointer(&mutexprofilerate),
	unsafe.Pointer(&gcController),
	unsafe.Pointer(&memstats),
	unsafe.Pointer(&sched),
	unsafe.Pointer(&ticks),
	unsafe.Pointer(&work),
}
