// $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var (
	r0 = 'a'
	r1 = 'a'+1
	r2 = 1+'a'
	r3 = 'a'*2
	r4 = 'a'/2
	r5 = 'a'<<1
	r6 = 'b'<<2
	r7 int32

	r = []rune{r0, r1, r2, r3, r4, r5, r6, r7}
)

var (
	f0 = 1.2
	f1 = 1.2/'a'

	f = []float64{f0, f1}
)

var (
	i0 = 1
	i1 = 1<<'\x01'
	
	i = []int{i0, i1}
)

const (
	maxRune = '\U0010FFFF'
)

var (
	b0 = maxRune < r0
	
	b = []bool{b0}
)
