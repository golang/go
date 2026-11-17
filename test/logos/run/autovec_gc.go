// run -goexperiment simd,preemptibleloops

//go:build amd64 || arm64

package main

import (
	"fmt"
	"runtime"
	"sync/atomic"
)

func inputValue(i int, seed uint64) uint64 {
	return uint64(i)*0x9e3779b97f4a7c15 ^ seed
}

//go:noinline
func newInput(n int, seed uint64) []uint64 {
	backing := make([]uint64, n+2)
	for i := range backing {
		backing[i] = inputValue(i, seed)
	}
	return backing[1 : n+1]
}

//go:noinline
func scalarGCStep(x, y, k uint64) uint64 {
	s := x + k
	return (s ^ (y - k)) + (s & (x | y))
}

//go:noinline
func gcMap(n int, sequence uint64, request, completed *atomic.Uint64) ([]uint64, bool) {
	// These interior source pointers have no owner in the caller's frame.
	a := newInput(n, 0x0123456789abcdef)
	b := newInput(n, 0xfedcba9876543210)
	backing := make([]uint64, n+2)
	backing[0], backing[n+1] = 0x13579bdf2468ace0, 0xfdb97531eca86420
	dst := backing[1 : n+1]
	if len(dst) > len(a) || len(dst) > len(b) {
		panic("fixture source length")
	}
	k := sequence * 0xd1b54a32d192ed03
	request.Store(sequence)
	for i := range dst {
		x, y := a[i], b[i]
		s := x + k
		dst[i] = (s ^ (y - k)) + (s & (x | y))
	}
	return backing, completed.Load() == sequence
}

func main() {
	oldProcs := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(oldProcs)
	var request, completed atomic.Uint64
	stop := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		var handled uint64
		for {
			select {
			case <-stop:
				return
			default:
			}
			sequence := request.Load()
			if sequence == handled {
				runtime.Gosched()
				continue
			}
			runtime.GC()
			completed.Store(sequence)
			handled = sequence
		}
	}()

	const n = 1 << 22
	completedInside := 0
	for sequence := uint64(1); sequence <= 8; sequence++ {
		backing, during := gcMap(n, sequence, &request, &completed)
		if during {
			completedInside++
		}
		for completed.Load() != sequence {
			runtime.Gosched()
		}
		if backing[0] != 0x13579bdf2468ace0 || backing[n+1] != 0xfdb97531eca86420 {
			panic("GC vector loop changed a canary")
		}
		k := sequence * 0xd1b54a32d192ed03
		for i, got := range backing[1 : n+1] {
			want := scalarGCStep(inputValue(i+1, 0x0123456789abcdef), inputValue(i+1, 0xfedcba9876543210), k)
			if got != want {
				panic(fmt.Sprintf("GC round=%d index=%d: got %x want %x", sequence, i, got, want))
			}
		}
	}
	close(stop)
	<-stopped
	// Each counted collection was requested after allocations and completed
	// before the kernel returned. With asyncpreemptoff=1, the vector backedge
	// is the only reachable safepoint in that window; n has no scalar tail.
	if completedInside == 0 {
		panic("no requested GC completed inside the vector kernel")
	}
}
