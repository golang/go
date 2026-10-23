// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

import "runtime"

// sink makes mix a function that is not a leaf. mix never calls it.
//
//go:noinline
func sink(x uint64) { println(x) }

// mix returns a value that depends on all n iterations. It keeps 28
// values live in a loop without calls, which is more than arm64 has
// registers other than R30. The compiler uses R30 for data in a function
// that is not a leaf, and the call to sink makes mix one. Only
// asynchronous preemption can interrupt the loop.
//
//go:noinline
func mix(n int) uint64 {
	k := uint64(n) | 0x9e3779b97f4a7c15
	a0 := k*1 ^ 0x6a09e667f3bcc908
	a1 := k*3 ^ 0x6a09e667f3bcc908
	a2 := k*5 ^ 0x6a09e667f3bcc908
	a3 := k*7 ^ 0x6a09e667f3bcc908
	a4 := k*9 ^ 0x6a09e667f3bcc908
	a5 := k*11 ^ 0x6a09e667f3bcc908
	a6 := k*13 ^ 0x6a09e667f3bcc908
	a7 := k*15 ^ 0x6a09e667f3bcc908
	a8 := k*17 ^ 0x6a09e667f3bcc908
	a9 := k*19 ^ 0x6a09e667f3bcc908
	a10 := k*21 ^ 0x6a09e667f3bcc908
	a11 := k*23 ^ 0x6a09e667f3bcc908
	a12 := k*25 ^ 0x6a09e667f3bcc908
	a13 := k*27 ^ 0x6a09e667f3bcc908
	a14 := k*29 ^ 0x6a09e667f3bcc908
	a15 := k*31 ^ 0x6a09e667f3bcc908
	a16 := k*33 ^ 0x6a09e667f3bcc908
	a17 := k*35 ^ 0x6a09e667f3bcc908
	a18 := k*37 ^ 0x6a09e667f3bcc908
	a19 := k*39 ^ 0x6a09e667f3bcc908
	a20 := k*41 ^ 0x6a09e667f3bcc908
	a21 := k*43 ^ 0x6a09e667f3bcc908
	a22 := k*45 ^ 0x6a09e667f3bcc908
	a23 := k*47 ^ 0x6a09e667f3bcc908
	a24 := k*49 ^ 0x6a09e667f3bcc908
	a25 := k*51 ^ 0x6a09e667f3bcc908
	a26 := k*53 ^ 0x6a09e667f3bcc908
	a27 := k*55 ^ 0x6a09e667f3bcc908
	for i := 0; i < n; i++ {
		a0 = (a0 ^ a1) + (a27 >> 3) + uint64(i)
		a1 = (a1 ^ a2) + (a0 >> 3) + uint64(i)
		a2 = (a2 ^ a3) + (a1 >> 3) + uint64(i)
		a3 = (a3 ^ a4) + (a2 >> 3) + uint64(i)
		a4 = (a4 ^ a5) + (a3 >> 3) + uint64(i)
		a5 = (a5 ^ a6) + (a4 >> 3) + uint64(i)
		a6 = (a6 ^ a7) + (a5 >> 3) + uint64(i)
		a7 = (a7 ^ a8) + (a6 >> 3) + uint64(i)
		a8 = (a8 ^ a9) + (a7 >> 3) + uint64(i)
		a9 = (a9 ^ a10) + (a8 >> 3) + uint64(i)
		a10 = (a10 ^ a11) + (a9 >> 3) + uint64(i)
		a11 = (a11 ^ a12) + (a10 >> 3) + uint64(i)
		a12 = (a12 ^ a13) + (a11 >> 3) + uint64(i)
		a13 = (a13 ^ a14) + (a12 >> 3) + uint64(i)
		a14 = (a14 ^ a15) + (a13 >> 3) + uint64(i)
		a15 = (a15 ^ a16) + (a14 >> 3) + uint64(i)
		a16 = (a16 ^ a17) + (a15 >> 3) + uint64(i)
		a17 = (a17 ^ a18) + (a16 >> 3) + uint64(i)
		a18 = (a18 ^ a19) + (a17 >> 3) + uint64(i)
		a19 = (a19 ^ a20) + (a18 >> 3) + uint64(i)
		a20 = (a20 ^ a21) + (a19 >> 3) + uint64(i)
		a21 = (a21 ^ a22) + (a20 >> 3) + uint64(i)
		a22 = (a22 ^ a23) + (a21 >> 3) + uint64(i)
		a23 = (a23 ^ a24) + (a22 >> 3) + uint64(i)
		a24 = (a24 ^ a25) + (a23 >> 3) + uint64(i)
		a25 = (a25 ^ a26) + (a24 >> 3) + uint64(i)
		a26 = (a26 ^ a27) + (a25 >> 3) + uint64(i)
		a27 = (a27 ^ a0) + (a26 >> 3) + uint64(i)
	}
	if n < 0 {
		sink(a0)
	}
	return a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7 ^ a8 ^ a9 ^ a10 ^ a11 ^ a12 ^ a13 ^ a14 ^ a15 ^ a16 ^ a17 ^ a18 ^ a19 ^ a20 ^ a21 ^ a22 ^ a23 ^ a24 ^ a25 ^ a26 ^ a27
}

// PreemptedMix returns mix(n), computed while another goroutine collects
// garbage in a loop. Each collection stops the world, which preempts the
// goroutine that runs mix, and scans its stack, which preempts it again.
//
//export PreemptedMix
func PreemptedMix(n int) uint64 {
	stop := make(chan struct{})
	collected := make(chan struct{})
	go func() {
		defer close(collected)
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()

	result := make(chan uint64)
	go func() { result <- mix(n) }()
	r := <-result

	close(stop)
	<-collected
	return r
}

func main() {}
