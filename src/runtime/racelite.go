// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: //go:build racelite

package runtime

import (
	"internal/goarch"
	"internal/runtime/gc"
	"unsafe"
)

// TODO(thepudds): we currently have a ~3 level scheme of lighter checks before doing more work.
// The intent is the first check could be emitted by the compiler.
// And in general, we certainly can improve from this very first cut.

var (
	// raceliteCheckAddrMask controls how often we check addresses for racelite.
	// Currently, for proof-of-concept, we set to check 1 out of 16 addresses to make
	// results come back very quickly, but a "real" system could be much higher.
	// We would pick a higher value to keep overhead manageable.
	raceliteCheckAddrMask = (1 << 4) - 1

	// raceliteCheckAddrRand allows us to randomize which addresses we check for racelite,
	// but to do so in a way we get a consistent answer per address across concurrent readers
	// and writers.
	//
	// TODO(thepudds): for now, for convenience we only check this in runtime, but we
	// could have the compiler emit the check -- probably important to do or at least
	// try if were were to pursue this general approach.
	raceliteCheckAddrRand uint32 = 0

	// raceliteCheckWordRand allows us to randomize which offset we check for racelite within
	// a given object, which helps prevent us from having different words from the same object
	// interfere with each other.
	raceliteCheckWordRand uint32 = 0
)

// const raceliteCheckAddrMask = (1 << 10) - 1 // check 1 out of 1024 addresses

// checkinstack checks if addr is in the current goroutine's stack.
func checkinstack(addr uintptr) bool {
	// TODO(thepudds): probably want to make sure we are not preempted.
	mp := acquirem()
	inStack := mp.curg.stack.lo <= addr && addr < mp.curg.stack.hi
	releasem(mp)
	return inStack
}

// racelitewrite instruments a store operation with lightweight data race detection.
// The logic follows the following structure:
//
//	Let A be the stored address
//	Let state(A) ∈ {0, 1} be the state of address A in the memory,
//		where 0 means the address is not being written to, and 1 means the address is being written to.
//
//	if state(A) ≠ 0 {
//		The address is already being written to!
//		Report write-write race
//	}
//	state(A) = 1
//	micropause()
//	if state(A) = 0 {
//		The address state was updated by another writer!
//		Report write-write race
//	}
//	state(A) = 0
func racelitewrite(addr uintptr) {
	randWordIndex, check := raceliteCheckAddr(addr)
	switch {
	case !check:
		// Not checking this addr. This is the common case.
		if diag() {
			print("racelitewrite: not checking addr=", hex(addr), "\n")
		}
		return
	case checkinstack(addr):
		if diag() {
			print("racelitewrite: skip stack address ", hex(addr), "\n")
		}
		return
	}

	base, span, _ := findObject(uintptr(addr), 0, 0)
	switch {
	case span == nil && base == 0:
		return
	case span.elemsize > gc.RaceliteFooterSize*8:
		if diag() {
			println("racelite: not tracking heap objects > 64 bytes yet")
		}
		return
	case span.elemsize <= 16:
		// 	// TODO(thepudds): sidestep tiny allocator for now.
		// 	if diag() {
		// 		println("racelite: not tracking heap objects <= 16 bytes yet")
		// 	}
		// 	return
	}

	// Determine the index of the address in the heap object.
	// Index 0 is base, index 1 is the next word, etc.
	//
	// Indexing is determined by the pointer size for the architecture.
	index := (uintptr(addr) - base) / goarch.PtrSize
	switch {
	case index >= 64:
		throw("racelite: index too large")
	case index != uintptr(randWordIndex)%(span.elemsize/goarch.PtrSize):
		// Check if we are monitoring this word index in general for this heap object.
		// This prevents false positives, e.g., 'p.a = 1' and 'p.b = 2' for some 'p struct{ a, b int }'
		// won't trigger races.
		if diag() {
			print("racelitewrite: skip undesired word ",
				"addr=", hex(addr), " index=", index, " randWordIndex=", randWordIndex,
				"\n")
		}
		return
	}

	// Now, we finally check and record our write in the footer.
	// First, get the footer.
	//
	// TODO(vsaioc): We want to dereference this as a value and compare the difference
	// instead of extracting a footer.
	footerPtr := (*uint64)(unsafe.Pointer(base + span.elemsize - gc.RaceliteFooterSize))

	if *footerPtr != 0 {
		// Already being written to.
		// Another writer intercepted the write.
		print("racelite: write-write race (entry) at ",
			"addr (base[index]) = ", hex(addr), " (", hex(base), "[", index, "])",
			"\n")
		return
	}

	// Now record our write, followed by a small delay to let someone see it.
	// We either do a ~nano delay or a ~1 in 100 chance of a ~micro delay.
	// The ~nano delay is from just checking cheaprand.
	*footerPtr ^= 1
	// good with 100k samples!
	if cheaprandn(10_000) == 0 {
		// TODO(thepudds): multiple ways to delay here. For now, do something simple that hopefully
		// let's us see it work for the first time. ;)
		usleep(1)
	}

	if *footerPtr == 0 {
		// Another writer intercepted the write.
		// TODO(vsaioc): This might be unnecessary, since we already check at the entry.
		print("racelite: write-write race (exit) at ",
			"addr (base[index]) = ", hex(addr), " (", hex(base), "[", index, "])",
			"\n")
	}

	// Now clear that bit, again via XOR.
	*footerPtr ^= 1
}

// raceliteread instruments a load operation with lightweight data race detection.
// The logic follows the following structure:
//
//	Let A be the loaded address
//	Let state(A) ∈ {0, 1} be the state of address A in the memory,
//		where 0 means the address is not being written to, and 1 means the address is being written to.
//
//	if state(A) ≠ 0 {
//		The address is already being written to.
//		Report write-read race
//	}
//	micropause()
//	if state(A) ≠ 0 {
//		The address state was updated by a writer.
//		Report read-write race
//	}
func raceliteread(addr uintptr) {
	randWordIndex, check := raceliteCheckAddr(addr)
	if !check {
		// Not checking this addr. This is the common case.
		if diag() {
			print("raceliteread: skip stack address ", hex(addr), "\n")
		}
		return
	}

	// Do a cheap-ish check to see if this points into our stack.
	// This check is likely cheaper than findObject or findSpan.
	// TODO(thepudds): probably want to make sure we are not preempted.
	if checkinstack(addr) {
		// This points into our stack, so ignore.
		if diag() {
			print("raceliteread: skip stack address ", hex(addr), "\n")
		}
		return
	}

	base, span, _ := findObject(uintptr(addr), 0, 0)

	switch {
	case span == nil && base == 0:
		return
	case span.elemsize > gc.RaceliteFooterSize*8:
		if diag() {
			print("raceliteread: not tracking objects > 64 bytes yet; ",
				"object at addr=", hex(addr), " is ", span.elemsize, " bytes",
				"\n")
		}
		return
	case span.elemsize <= 16:
		// 	// TODO(thepudds): sidestep tiny allocator for now.
		// 	if diag() {
		// 		println("racelite: not tracking heap objects <= 16 bytes yet")
		// 	}
		// 	return
	}

	// Determine the index of the address in the heap object.
	// Index 0 is base, index 1 is the next word, etc.
	//
	// Indexing is determined by the pointer size for the architecture.
	index := (uintptr(addr) - base) / goarch.PtrSize

	switch {
	case index >= 64:
		// TODO(vsaioc): should this value be hardcoded?
		throw("raceliteread: index too large")
	case index != uintptr(randWordIndex)%(span.elemsize/goarch.PtrSize):
		// Check if we are monitoring this word index in general for this heap object.
		// This prevents false positives, e.g., 'p.a = 1' and '_ = p.b' for some 'p struct{ a, b int }'
		// won't trigger races.
		if diag() {
			print("raceliteread: skip undesired word ",
				"addr=", hex(addr), " index=", index, " randWordIndex=", randWordIndex,
				"\n")
		}
		return
	}

	// Now, we finally check and record our write in the footer.
	// First, get the footer.
	footerPtr := (*uint64)(unsafe.Pointer(base + span.elemsize - gc.RaceliteFooterSize))

	if *footerPtr != 0 {
		// Already written to!
		print("racelite: write-read race at ",
			"addr (base[index]) = ", hex(addr), " (", hex(base), "[", index, "])",
			"\n")
		return
	}

	// TODO(vsaioc): We are currently randomly sampling the delay
	// at in 1 in 10,000. This should be configurable.
	if cheaprandn(10_000) == 0 {
		// TODO(thepudds): multiple ways to delay here. For now, do something simple that hopefully
		// let's us see it work for the first time. ;)
		usleep(1)
	}

	if *footerPtr != 0 {
		// Another writer intercepted the read.
		print("racelite: read-write race at ",
			"addr (base[index]) = ", hex(addr), " (", hex(base), "[", index, "])",
			"\n")
	}
}

// raceliteCheckAddr reports if we should check addr for racelite,
// and if so, which word index within the object to check.
func raceliteCheckAddr(addr uintptr) (index uint64, ok bool) {
	// TODO(thepudds): multiple ways to do this, improve later.

	index = uint64(raceliteCheckWordRand & 63)

	// Most addrs are 8 or 16 byte aligned.
	// TODO(thepudds): probably don't need the xor, maybe not shift. maybe just do a mask and == check.
	// Also debatable if we want monitored addresses to be strided (while shifting over time),
	// or maybe better not strided.
	blended := uint32(addr>>3) ^ raceliteCheckAddrRand
	_ = blended
	return index, true
}

// diag reports true is we should print extra info.
func diag() bool {
	return debug.racelite >= 2
}

/*

Sample program we can currently detect:

$ GOEXPERIMENT=nosizespecializedmalloc GODEBUG=racelite=1 go run -gcflags='-racelite' . |& grep 'racelite: multiple concurrent writes'

or via stress, which fails maybe 15% of the time:

$ GOEXPERIMENT=nosizespecializedmalloc GODEBUG=racelite=1 stress -p 2 go run -gcflags='-racelite' .

$ cat t.go

package main

import (
	"fmt"
	"sync"
	"time"
)

type Foo struct{ a, b, c int } // 24 bytes; avoids tiny allocator.

func main() {
	start := time.Now()
	v := Foo{}
	p := &v

	// Do two racy writes.
	var wg sync.WaitGroup
	// var mu sync.Mutex
	for range 2 {
		wg.Go(func() {
			for range 10000 {
				// mu.Lock()
				// Without the mutex, both goroutines have a data race on writing to p.c here.
				// This triggers racelitewrite, and we detect the race (some percent of time)!
				// If we uncomment the mutex, then no race is reported.
				p.c = 2

				// mu.Unlock()
			}
		})
	}
	wg.Wait()
	sink = &p
	println(v.c)
	fmt.Println("done in", time.Since(start))
}

var sink any

*/
