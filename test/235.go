// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Solve the 2,3,5 problem (print all numbers with 2, 3, or 5 as factor) using channels.
// Test the solution, silently.

package main

type T chan uint64

func M(f uint64) (in, out T) {
	in = make(T, 100)
	out = make(T, 100)
	go func(in, out T, f uint64) {
		for {
			out <- f*<-in
		}
	}(in, out, f)
	return in, out
}


func min(xs []uint64) uint64 {
	m := xs[0]
	for i := 1; i < len(xs); i++ {
		if xs[i] < m {
			m = xs[i]
		}
	}
	return m
}


func main() {
	F := []uint64{2, 3, 5}
	var n = len(F)
	OUT := []uint64{
		2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36,
		40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125,
		128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250,
		256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480,
		486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729, 750, 768,
		800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 1215,
		1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600}

	x := uint64(1)
	ins := make([]T, n)
	outs := make([]T, n)
	xs := make([]uint64, n)
	for i := 0; i < n; i++ {
		ins[i], outs[i] = M(F[i])
		xs[i] = x
	}

	for i := 0; i < len(OUT); i++ {
		for i := 0; i < n; i++ {
			ins[i] <- x
		}

		for i := 0; i < n; i++ {
			if xs[i] == x {
				xs[i] = <-outs[i]
			}
		}

		x = min(xs)
		if x != OUT[i] {
			println("bad: ", x, " should be ", OUT[i])
			panic("235")
		}
	}
}
