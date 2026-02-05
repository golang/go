// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: //go:build racelite

package runtime

import (
	"internal/goarch"
	"internal/runtime/atomic"
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

	// raceliteCheckAddrMask = (1 << 10) - 1 // check 1 out of 1024 addresses
)

// raceliteVirtualRegister is a virtual register that can be used to store an address.
type raceliteVirtualRegister struct {
	state   atomic.Uint32  // 0 means not being written to, 1 means being written to
	addr    atomic.Uintptr // the address currently being monitored in the virtual register
	watcher atomic.Uint32  // the number of watchers on the virtual register
}

var raceliteReg *raceliteVirtualRegister = new(raceliteVirtualRegister)

func (r *raceliteVirtualRegister) init() {
	r.state.Store(0)
	r.addr.Store(0)
}

func (r *raceliteVirtualRegister) AddWatcher() {
	r.watcher.Add(1)
}

func (r *raceliteVirtualRegister) RemoveWatcher() {
	if r.watcher.Add(-1) == 0 {
		// There are no more watchers. Disarm the virtual register.
		r.state.Store(0)
		r.addr.Store(0)
	}
}

// diag reports true is we should print extra info.
func diag() bool {
	return debug.racelite >= 2
}

// checkinstack checks if addr is in the current goroutine's stack.
func checkinstack(addr uintptr) bool {
	// TODO(thepudds): probably want to make sure we are not preempted.
	// TODO(vsaioc): is the check through mp.preemptoff or mp.curg.preempt?
	mp := acquirem()
	inStack := mp.curg.stack.lo <= addr && addr < mp.curg.stack.hi
	releasem(mp)
	return inStack
}

// racelitewrite instruments a store operation with lightweight data race detection.
// The logic follows the following structure:
//
//	Let A be the stored address
//	Let state(A) ∈ {0, 1} be the status of address A:
//		- 0 denotes that the address is not being written to
//		- 1 denotes that the address is being written to
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
			print("RACELITE write: not checking addr=", hex(addr), "\n")
		}
		return
	case checkinstack(addr):
		if diag() {
			print("RACELITE write: skip stack address ", hex(addr), "\n")
		}
		return
	}

	// TODO(vsaioc): remove indexing (not necessary without footer)
	base, span, _ := findObject(uintptr(addr), 0, 0)
	switch {
	case span == nil && base == 0:
		// We got some bad pointer.
		return
	case span.elemsize > 64:
		if diag() {
			println("RACELITE: not tracking heap objects > 64 bytes yet")
		}
		return
	case span.elemsize <= 16:
		// 	// TODO(thepudds): sidestep tiny allocator for now.
		// 	if diag() {
		// 		println("RACELITE: not tracking heap objects <= 16 bytes yet")
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
		throw("RACELITE: index too large")
	case index != uintptr(randWordIndex)%(span.elemsize/goarch.PtrSize):
		// Check if we are monitoring this word index in general for this heap object.
		// This prevents false positives, e.g., 'p.a = 1' and 'p.b = 2' for some 'p struct{ a, b int }'
		// won't trigger races.
		if diag() {
			print("RACELITE write: skip undesired word ",
				"addr=", hex(addr), " index=", index, " randWordIndex=", randWordIndex,
				"\n")
		}
		return
	}

	// Check the status of the virtual register.
	switch registerAddr := raceliteReg.addr.Load(); registerAddr {
	case 0:
		// We now have a watcher on the virtual register.
		raceliteReg.AddWatcher()
		// The virtual register is not occupied. Try to snatch it.
		if !raceliteReg.addr.CompareAndSwap(0, addr) {
			// Someone else beat us to it.
			//
			// TODO(vsaioc): There is a narrow window where the CAS
			// may fail and we swap with the same address.
			// This could lead to a false negative.
			if diag() {
				print("RACELITE write: virtual register already occupied by addr=", hex(addr), "\n")
			}
			raceliteReg.RemoveWatcher()
			return
		}
		if diag() {
			print("RACELITE write: snatched virtual register\n",
				"Virtual register:",
				" addr=", hex(addr),
				" reg.addr=", hex(raceliteReg.addr.Load()),
				" state=", raceliteReg.state.Load(),
				" watcher=", raceliteReg.watcher.Load(),
				" goid=", getg().goid,
				"\n")
		}
	case addr:
		// We now have a watcher on the virtual register.
		raceliteReg.AddWatcher()
		// The virtual register is already occupied by the same address.
		// Proceed as normal.
		if diag() {
			print("RACELITE write: virtual register occupied by same address\n",
				"Virtual register:",
				" addr=", hex(addr),
				" reg.addr=", hex(raceliteReg.addr.Load()),
				" state=", raceliteReg.state.Load(),
				" watcher=", raceliteReg.watcher.Load(),
				" goid=", getg().goid,
				"\n")
		}
	default:
		// If the virtual register is already occupied by another address,
		// then bail out.
		if diag() {
			print("RACELITE write: virtual register already occupied by other address ", hex(registerAddr), "\n")
		}
		return
	}

	// Try to mark the state as being written to.
	if !raceliteReg.state.CompareAndSwap(0, 1) {
		if diag() {
			print("RACELITE write: already writing\n",
				"Virtual register:",
				" addr=", addr,
				" reg.addr=", hex(raceliteReg.addr.Load()),
				" state=", raceliteReg.state.Load(),
				" watcher=", raceliteReg.watcher.Load(),
				" goid=", getg().goid,
				"\n")
		}
		// Another writer intercepted the write.
		//
		// Write the current goroutine's stack on the system stack.
		stw := stopTheWorld(stwRacelite)
		gp := getg()
		systemstack(func() {
			print("RACELITE TRIGGERED: write-write race (entry) at ",
				"addr (base[index]) = ", hex(addr), " (", hex(base), "[", index, "])",
				" goroutine=", gp.goid,
				"\n")
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
			println("RACELITE END")
		})
		startTheWorld(stw)
		// We can disengage the watcher here, because we have already observed the data race.
		raceliteReg.RemoveWatcher()
		if diag() {
			print("RACELITE write: removed watcher\n",
				"Virtual register:",
				" addr=", hex(addr),
				" reg.addr=", hex(raceliteReg.addr.Load()),
				" state=", raceliteReg.state.Load(),
				" watcher=", raceliteReg.watcher.Load(),
				" goid=", getg().goid,
				"\n")
		}
		return
	}
	if diag() {
		print("RACELITE write: marked state as being written to\n",
			"Virtual register:",
			" addr=", hex(addr),
			" reg.addr=", hex(raceliteReg.addr.Load()),
			" state=", raceliteReg.state.Load(),
			" watcher=", raceliteReg.watcher.Load(),
			" goid=", getg().goid,
			"\n")
	}

	// Now record our write, followed by a small delay to let someone see it.
	// We either do a ~nano delay or a ~1 in 100 chance of a ~micro delay.
	// The ~nano delay is from just checking cheaprand.
	// good with 100k samples!
	if diag() || cheaprandn(10_000) == 0 {
		// TODO(thepudds): multiple ways to delay here. For now, do something simple that hopefully
		// let's us see it work for the first time. ;)
		usleep(1)
	}
	if diag() {
		Gosched() // FIXME: Remove this after testing.
	}

	// Reset the state to "not-writing".
	if !raceliteReg.state.CompareAndSwap(1, 0) {
		// Another writer intercepted this write.
		// TODO(vsaioc): This might be unnecessary, since we already check at the entry.
		// We are already sampling, so missing the narrow window where the two write
		// checks might race overlap is (probably) not a big deal.
		//
		// Write the current goroutine's stack on the system stack.
		stw := stopTheWorld(stwRacelite)
		gp := getg()
		systemstack(func() {
			print("RACELITE TRIGGERED: write-write race (exit) at ",
				"addr (base[index]) = ", hex(addr), " (", hex(base), "[", index, "])",
				" goroutine=", gp.goid,
				"\n")
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
			println("RACELITE END")
		})
		startTheWorld(stw)
	}
	raceliteReg.RemoveWatcher()
	if diag() {
		print("RACELITE write: removed watcher and reset state\n",
			"Virtual register:",
			" addr=", hex(addr),
			" reg.addr=", hex(raceliteReg.addr.Load()),
			" state=", raceliteReg.state.Load(),
			" watcher=", raceliteReg.watcher.Load(),
			" goid=", getg().goid,
			"\n")
	}
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
			print("RACELITE read: skip stack address ", hex(addr), "\n")
		}
		return
	}

	// Do a cheap-ish check to see if this points into our stack.
	// This check is likely cheaper than findObject or findSpan.
	// TODO(thepudds): probably want to make sure we are not preempted.
	if checkinstack(addr) {
		// This points into our stack, so ignore.
		if diag() {
			print("RACELITE read: skip stack address ", hex(addr), "\n")
		}
		return
	}

	base, span, _ := findObject(uintptr(addr), 0, 0)

	switch {
	case span == nil && base == 0:
		return
	case span.elemsize > 64:
		if diag() {
			print("RACELITE read: not tracking objects > 64 bytes yet; ",
				"object at addr=", hex(addr), " is ", span.elemsize, " bytes",
				"\n")
		}
		return
	case span.elemsize <= 16:
		// 	// TODO(thepudds): sidestep tiny allocator for now.
		// 	if diag() {
		// 		println("RACELITE: not tracking heap objects <= 16 bytes yet")
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
		throw("RACELITE read: index too large")
	case index != uintptr(randWordIndex)%(span.elemsize/goarch.PtrSize):
		// Check if we are monitoring this word index in general for this heap object.
		// This prevents false positives, e.g., 'p.a = 1' and '_ = p.b' for some 'p struct{ a, b int }'
		// from triggering races.
		if diag() {
			print("RACELITE read: skip undesired word ",
				"addr=", hex(addr), " index=", index, " randWordIndex=", randWordIndex,
				"\n")
		}
		return
	}

	// Add a watcher here to prevent swapping out the virtual register.
	raceliteReg.AddWatcher()
	if diag() {
		print("RACELITE read: added watcher\n",
			"Virtual register:",
			" addr=", hex(addr),
			" reg.addr=", hex(raceliteReg.addr.Load()),
			" state=", raceliteReg.state.Load(),
			" watcher=", raceliteReg.watcher.Load(),
			" goid=", getg().goid,
			"\n")
	}

	// Check the status of the virtual register.
	switch registerAddr := raceliteReg.addr.Load(); registerAddr {
	case 0:
		if diag() {
			print("RACELITE read: virtual register not occupied\n",
				"Virtual register:",
				" addr=", hex(raceliteReg.addr.Load()),
				" state=", raceliteReg.state.Load(),
				" watcher=", raceliteReg.watcher.Load(),
				" goid=", getg().goid,
				"\n")
		}
		// The virtual register is not occupied. Try to snatch it.
		if !raceliteReg.addr.CompareAndSwap(0, addr) {
			// Someone else beat us to it.
			//
			// TODO(vsaioc): There is a narrow window where the CAS
			// may fail and we swap with the same address.
			// This could lead to a false negative.
			if diag() {
				print("RACELITE write: virtual register already occupied by addr=", hex(addr), "\n")
			}
			return
		}
	case addr:
		if diag() {
			print("RACELITE read: virtual register occupied by same address\n",
				"Virtual register:",
				" addr=", addr,
				" state=", raceliteReg.state.Load(),
				" watcher=", raceliteReg.watcher.Load(),
				" goid=", getg().goid,
				"\n")
		}
		// The virtual register is already occupied by the same address.
		// Check whether we are being written to.
		if raceliteReg.state.Load() != 0 {
			// Already being written to.
			// Another writer intercepted the write.
			//
			// Write the current goroutine's stack on the system stack.
			stw := stopTheWorld(stwRacelite)
			gp := getg()
			systemstack(func() {
				print("RACELITE TRIGGERED: write-read race at ",
					"addr (base[index]) = ", hex(addr), " (", hex(base), "[", index, "])",
					" goroutine=", gp.goid,
					"\n")
				traceback(^uintptr(0), ^uintptr(0), 0, gp)
				println("RACELITE END")
			})
			startTheWorld(stw)
			// We can disengage here the watcher here, because we have already observed the data race.
			raceliteReg.RemoveWatcher()
			return
		}
	}

	// TODO(vsaioc): We are currently randomly sampling the delay
	// at in 1 in 10,000. This should be configurable.
	if cheaprandn(10_000) == 0 {
		// TODO(thepudds): multiple ways to delay here. For now, do something simple that hopefully
		// let's us see it work for the first time. ;)
		usleep(1)
	}
	Gosched() // FIXME: Remove this after testing.

	// Check again whether we are being written to,
	// after the delay.
	if raceliteReg.state.Load() != 0 {
		// Another writer intercepted the read.
		//
		// Write the current goroutine's stack on the system stack.
		stw := stopTheWorld(stwRacelite)
		gp := getg()
		systemstack(func() {
			print("RACELITE TRIGGERED: read-write race at ",
				"addr (base[index]) = ", hex(addr), " (", hex(base), "[", index, "])",
				" goroutine=", gp.goid,
				"\n")
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
			println("RACELITE END")
		})
		startTheWorld(stw)
	}
	raceliteReg.RemoveWatcher()
	if diag() {
		print("RACELITE read: removed watcher\n",
			"Virtual register:",
			" addr=", hex(raceliteReg.addr.Load()),
			" state=", raceliteReg.state.Load(),
			" watcher=", raceliteReg.watcher.Load(),
			" goid=", getg().goid,
			"\n")
	}
}

// raceliteCheckAddr reports if we should check the given address.
// It also retrieves the word index within the object.
func raceliteCheckAddr(addr uintptr) (index uint64, ok bool) {
	// TODO(thepudds): multiple ways to do this, improve later.

	// Select last 6 bits of the address.
	index = uint64(raceliteCheckWordRand & 63)

	if diag() {
		// TOOO(vsaioc): Decouple this check from diagnostics.
		// For debugging, always sample the address.
		return index, true
	}
	// Most addresses are 8 or 16 byte aligned.
	// TODO(thepudds): probably don't need the xor, maybe not shift. maybe just do a mask and == check.
	// Also debatable if we want monitored addresses to be strided (while shifting over time),
	// or maybe better not strided.
	return index, uint32(addr>>3)^raceliteCheckAddrRand == 0
}

/*

Sample program we can currently detect:

$ GOEXPERIMENT=nosizespecializedmalloc GODEBUG=racelite=1 go run -gcflags='-racelite' . |& grep 'RACELITE: multiple concurrent writes'

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
