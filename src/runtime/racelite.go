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

// raceliteCheckAddrMask controls how often we check addresses for racelite.
// Currently, for proof-of-concept, we set to check 1 out of 16 addresses to make
// results come back very quickly, but a "real" system could be much higher.
// We would pick a higher value to keep overhead manageable.
const raceliteCheckAddrMask = (1 << 4) - 1

// const raceliteCheckAddrMask = (1 << 10) - 1 // check 1 out of 1024 addresses

func racelitewrite(addr uintptr) {
	if diag() {
		println("racelitewrite called with addr:    ", hex(addr),
			"raceliteCheckAddrRand:", hex(raceliteCheckAddrRand), "raceliteCheckWordRand:", hex(raceliteCheckWordRand))
	}

	randWordIndex, check := raceliteCheckAddr(addr)
	if !check {
		// Not checking this addr. This is the common case.
		if diag() {
			println("racelitewrite not checking addr:    ", hex(addr))
		}
		return
	}

	// Do a cheap-ish check to see if this points into our stack.
	// This check is likely cheaper than findObject or findSpan.
	// TODO(thepudds): probably want to make sure we are not preempted.
	mp := acquirem()
	if mp.curg.stack.lo <= uintptr(addr) && uintptr(addr) < mp.curg.stack.hi {
		// This points into our stack, so ignore.
		if diag() {
			println("racelitewrite skip checking pointer into stack:    ", hex(addr))
		}
		releasem(mp)
		return
	}
	releasem(mp)

	base, span, _ := findObject(uintptr(addr), 0, 0)
	if diag() {
		println("racelitewrite called for base addr:", hex(base))
	}

	if span != nil && base != 0 {
		if span.elemsize > gc.RaceliteFooterSize*8 {
			if diag() {
				println("racelite: not tracking heap objects > 64 bytes yet")
			}
			return
		}
		if span.elemsize <= 16 {
			// TODO(thepudds): sidestep tiny allocator for now.
			if diag() {
				println("racelite: not tracking heap objects <= 16 bytes yet")
			}
			return
		}

		// Determine how far addr is into this heap allocation object.
		// For example, does it point to the base (wordIndex 0), or
		// point into the next work (wordIndex 1), etc.
		wordIndex := (uintptr(addr) - base) / goarch.PtrSize
		if diag() {
			println("racelite: word index being accessed:", wordIndex)
		}
		if wordIndex >= 64 {
			throw("racelite: wordIndex too large")
		}

		// TEMP: for sanity checking, we can force it to only track word index 0.
		// if wordIndex != 0 {
		// 	// TEMP: for sanity checking:
		// 	if diag() { println("racelite: TEMP: only tracking writes to word index 0 of object for now")}
		// 	return
		// }

		// Check if we are monitoring this word index in general for this heap object.
		// This intent is so that any given heap object is only being monitored for writes
		// on a single word at any given moment, which means something like 'p.a = 1' in one goroutine
		// and 'p.b = 2' in another goroutine for 'var p struct{ a, b int }' won't both attempt
		// to trigger writes on the single footer for the same heap object for both a and b fields
		// at the same time.
		if wordIndex != uintptr(randWordIndex)%(span.elemsize/goarch.PtrSize) {
			if diag() {
				println("racelite: skipping write check, addr is not for a desired word of object. addr:", hex(addr), "wordIndex:", wordIndex, "randWordIndex:", randWordIndex)
			}
			return
		}

		// Now, we finally check and record our write in the footer.
		// First, get the footer.
		footerPtr := (*uint64)(unsafe.Pointer(base + span.elemsize - gc.RaceliteFooterSize))

		if *footerPtr != 0 {
			// Already written to!
			println("racelite: multiple concurrent writes to same object detected")
			println("racelite: write flag already set for addr", hex(addr), "at word", wordIndex, "of object at base", hex(base))
			println("racelite: existing footer value:      ", hex(*footerPtr))
			fatal("racelite: multiple concurrent writes to same object detected")
		}
		// Now record our write, followed by a small delay to let someone see it.
		// We either do a ~nano delay or a ~1 in 100 chance of a ~micro delay.
		// The ~nano delay is from just checking cheaprand.
		println("racelitewrite writing bit in footer at:  ", footerPtr, "word index:", wordIndex)
		*footerPtr ^= 1
		// good with 100k samples!
		// if cheaprandn(50_000) == 0 {
		if cheaprandn(1_000_000) == 0 {
			// TODO(thepudds): multiple ways to delay here. For now, do something simple that hopefully
			// let's us see it work for the first time. ;)
			usleep(10)
		}

		if diag() {
			println("racelitewrite clearing bit in footer at:  ", footerPtr, "word index:", wordIndex)
		}

		if *footerPtr == 0 {
			// Someone else changed it! It should be 1 still if we are the only writer.
			println("racelite: multiple concurrent writes to same object detected")
			println("racelite: write flag was changed for addr", hex(addr), "at word", wordIndex, "of object at base", hex(base))
			println("racelite: existing footer value:      ", *footerPtr)
			panic("racelite: multiple concurrent writes to same object detected")
		}

		// Now clear that bit, again via XOR.
		*footerPtr ^= 1

		// Done!
	}
}

func raceliteread(addr uintptr) {
	// TODO(thepudds): not yet implemented. we'll first prototype with write/write races.
	// if diag() {
	// 	println("raceliteread called with addr:     ", hex(addr))
	// }
}

// raceliteCheckAddr reports if we should check addr for racelite,
// and if so, which word index within the object to check.
func raceliteCheckAddr(addr uintptr) (wordIndex uint64, ok bool) {
	// TODO(thepudds): multiple ways to do this, improve later.

	wordIndex = uint64(raceliteCheckWordRand & 63)

	// Most addrs are 8 or 16 byte aligned.
	// TODO(thepudds): probably don't need the xor, maybe not shift. maybe just do a mask and == check.
	// Also debatable if we want monitored addresses to be strided (while shifting over time),
	// or maybe better not strided.
	blended := uint32(addr>>3) ^ raceliteCheckAddrRand
	return wordIndex, blended&raceliteCheckAddrMask == 0
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
