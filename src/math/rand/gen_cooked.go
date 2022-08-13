// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// This program computes the value of rngCooked in rng.go,
// which is used for seeding all instances of rand.Source.
// a 64bit and a 63bit version of the array is printed to
// the standard output.

package main

import "fmt"

const (
	length = 607
	tap    = 273
	mask   = (1 << 63) - 1
	a      = 48271
	m      = (1 << 31) - 1
	q      = 44488
	r      = 3399
)

var (
	rngVec          [length]int64
	rngTap, rngFeed int
)

func seedrand(x int32) int32 {
	hi := x / q
	lo := x % q
	x = a*lo - r*hi
	if x < 0 {
		x += m
	}
	return x
}

func srand(seed int32) {
	rngTap = 0
	rngFeed = length - tap
	seed %= m
	if seed < 0 {
		seed += m
	} else if seed == 0 {
		seed = 89482311
	}
	x := seed
	for i := -20; i < length; i++ {
		x = seedrand(x)
		if i >= 0 {
			var u int64
			u = int64(x) << 20
			x = seedrand(x)
			u ^= int64(x) << 10
			x = seedrand(x)
			u ^= int64(x)
			rngVec[i] = u
		}
	}
}

func vrand() int64 {
	rngTap--
	if rngTap < 0 {
		rngTap += length
	}
	rngFeed--
	if rngFeed < 0 {
		rngFeed += length
	}
	x := (rngVec[rngFeed] + rngVec[rngTap])
	rngVec[rngFeed] = x
	return x
}

func main() {
	srand(1)
	for i := uint64(0); i < 7.8e12; i++ {
		vrand()
	}
	fmt.Printf("rngVec after 7.8e12 calls to vrand:\n%#v\n", rngVec)
	for i := range rngVec {
		rngVec[i] &= mask
	}
	fmt.Printf("lower 63bit of rngVec after 7.8e12 calls to vrand:\n%#v\n", rngVec)
}
