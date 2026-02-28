// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// This type and the following one will share the same GC shape and size.
type Pointery struct {
	p *Pointery
	x [1024]int
}

type Pointery2 struct {
	p *Pointery2
	x [1024]int
}

// This type and the following one will have the same size.
type Vanilla struct {
	np uintptr
	x  [1024]int
}

type Vanilla2 struct {
	np uintptr
	x  [1023]int
	y  int
}

type Single struct {
	np uintptr
	x  [1023]int
}

var G int

//go:noinline
func clobber() {
	G++
}

func ABC(i, j int) int {
	r := 0

	// here v2 and v3 can be overlapped.
	clobber()
	if i < 101 {
		var v2 Vanilla
		v2.x[i] = j
		r += v2.x[j]
	}
	if j != 303 {
		var v3 Vanilla2
		v3.x[i] = j
		r += v3.x[j]
	}
	clobber()

	// not an overlap candidate (only one var of this size).
	var s Single
	s.x[i] = j
	r += s.x[j]

	// Here p1 and p2 interfere, but p1 could be overlapped with xp3 + xp4.
	var p1, p2 Pointery
	p1.x[i] = j
	r += p1.x[j]
	p2.x[i] = j
	r += p2.x[j]
	if j != 505 {
		var xp3 Pointery2
		xp3.x[i] = j
		r += xp3.x[j]
	}

	if i == j*2 {
		// p2 live on this path
		p2.x[i] += j
		r += p2.x[j]
	} else {
		// p2 not live on this path
		var xp4 Pointery2
		xp4.x[i] = j
		r += xp4.x[j]
	}

	return r + G
}
