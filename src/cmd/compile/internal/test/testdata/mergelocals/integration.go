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

func ABC(i, j int) int {
	r := 0

	// here v1 interferes with v2 but could be overlapped with v3.
	// we can also overlap v1 with v3.
	var v1 Vanilla
	if i < 101 {
		var v2 Vanilla
		v1.x[i] = j
		r += v1.x[j]
		v2.x[i] = j
		r += v2.x[j]
	}

	{
		var v3 Vanilla2
		v3.x[i] = j
		r += v3.x[j]
	}

	var s Single
	s.x[i] = j
	r += s.x[j]

	// Here p1 and p2 interfere, but p1 could be overlapped with xp3.
	var p1, p2 Pointery
	p1.x[i] = j
	r += p1.x[j]
	p2.x[i] = j
	r += p2.x[j]
	{
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

	return r
}
