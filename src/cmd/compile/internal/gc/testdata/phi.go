// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Test to make sure spills of cast-shortened values
// don't end up spilling the pre-shortened size instead
// of the post-shortened size.

import (
	"fmt"
	"runtime"
)

// unfoldable true
var true_ = true

var data1 [26]int32
var data2 [26]int64

func init() {
	for i := 0; i < 26; i++ {
		// If we spill all 8 bytes of this datum, the 1 in the high-order 4 bytes
		// will overwrite some other variable in the stack frame.
		data2[i] = 0x100000000
	}
}

func foo() int32 {
	var a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z int32
	if true_ {
		a = data1[0]
		b = data1[1]
		c = data1[2]
		d = data1[3]
		e = data1[4]
		f = data1[5]
		g = data1[6]
		h = data1[7]
		i = data1[8]
		j = data1[9]
		k = data1[10]
		l = data1[11]
		m = data1[12]
		n = data1[13]
		o = data1[14]
		p = data1[15]
		q = data1[16]
		r = data1[17]
		s = data1[18]
		t = data1[19]
		u = data1[20]
		v = data1[21]
		w = data1[22]
		x = data1[23]
		y = data1[24]
		z = data1[25]
	} else {
		a = int32(data2[0])
		b = int32(data2[1])
		c = int32(data2[2])
		d = int32(data2[3])
		e = int32(data2[4])
		f = int32(data2[5])
		g = int32(data2[6])
		h = int32(data2[7])
		i = int32(data2[8])
		j = int32(data2[9])
		k = int32(data2[10])
		l = int32(data2[11])
		m = int32(data2[12])
		n = int32(data2[13])
		o = int32(data2[14])
		p = int32(data2[15])
		q = int32(data2[16])
		r = int32(data2[17])
		s = int32(data2[18])
		t = int32(data2[19])
		u = int32(data2[20])
		v = int32(data2[21])
		w = int32(data2[22])
		x = int32(data2[23])
		y = int32(data2[24])
		z = int32(data2[25])
	}
	// Lots of phis of the form phi(int32,int64) of type int32 happen here.
	// Some will be stack phis. For those stack phis, make sure the spill
	// of the second argument uses the phi's width (4 bytes), not its width
	// (8 bytes).  Otherwise, a random stack slot gets clobbered.

	runtime.Gosched()
	return a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v + w + x + y + z
}

func main() {
	want := int32(0)
	got := foo()
	if got != want {
		fmt.Printf("want %d, got %d\n", want, got)
		panic("bad")
	}
}
