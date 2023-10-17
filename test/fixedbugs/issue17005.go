// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This tickles (a version of) the PPC64 back end to
// emit a BVS instruction.

package foo

type Flag int

const (
	Identity  Flag = iota - 2 // H is the identity matrix; no rotation is needed.
	Rescaling                 // H specifies rescaling.
)

type DrotmParams struct {
	Flag
}

func Drotmg(d1, d2, x1, y1 float64) (p DrotmParams, rd1, rd2, rx1 float64) {

	const (
		gam    = 4.0
		gamsq  = 16.0
		rgamsq = 5e-8
	)

	if d1 < 0 {
		p.Flag = Rescaling
		return
	}

	for rd1 <= rgamsq || rd1 >= gamsq {
		if rd1 <= rgamsq {
			rd1 *= gam * gam
			rx1 /= gam
		} else {
			rd1 /= gam * gam
			rx1 *= gam
		}
	}
	return
}
