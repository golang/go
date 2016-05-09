// skip

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that many initializations can be done at link time and
// generate no executable init functions.
// This test is run by sinit_run.go.

package p

import "unsafe"

// Should be no init func in the assembly.
// All these initializations should be done at link time.

type S struct{ a, b, c int }
type SS struct{ aa, bb, cc S }
type SA struct{ a, b, c [3]int }
type SC struct{ a, b, c []int }

var (
	zero                      = 2
	one                       = 1
	pi                        = 3.14
	slice                     = []byte{1, 2, 3}
	sliceInt                  = []int{1, 2, 3}
	hello                     = "hello, world"
	bytes                     = []byte("hello, world")
	four, five                = 4, 5
	x, y                      = 0.1, "hello"
	nilslice   []byte         = nil
	nilmap     map[string]int = nil
	nilfunc    func()         = nil
	nilchan    chan int       = nil
	nilptr     *byte          = nil
)

var a = [3]int{1001, 1002, 1003}
var s = S{1101, 1102, 1103}
var c = []int{1201, 1202, 1203}

var aa = [3][3]int{[3]int{2001, 2002, 2003}, [3]int{2004, 2005, 2006}, [3]int{2007, 2008, 2009}}
var as = [3]S{S{2101, 2102, 2103}, S{2104, 2105, 2106}, S{2107, 2108, 2109}}
var ac = [3][]int{[]int{2201, 2202, 2203}, []int{2204, 2205, 2206}, []int{2207, 2208, 2209}}

var sa = SA{[3]int{3001, 3002, 3003}, [3]int{3004, 3005, 3006}, [3]int{3007, 3008, 3009}}
var ss = SS{S{3101, 3102, 3103}, S{3104, 3105, 3106}, S{3107, 3108, 3109}}
var sc = SC{[]int{3201, 3202, 3203}, []int{3204, 3205, 3206}, []int{3207, 3208, 3209}}

var ca = [][3]int{[3]int{4001, 4002, 4003}, [3]int{4004, 4005, 4006}, [3]int{4007, 4008, 4009}}
var cs = []S{S{4101, 4102, 4103}, S{4104, 4105, 4106}, S{4107, 4108, 4109}}
var cc = [][]int{[]int{4201, 4202, 4203}, []int{4204, 4205, 4206}, []int{4207, 4208, 4209}}

var answers = [...]int{
	// s
	1101, 1102, 1103,

	// ss
	3101, 3102, 3103,
	3104, 3105, 3106,
	3107, 3108, 3109,

	// [0]
	1001, 1201, 1301,
	2101, 2102, 2103,
	4101, 4102, 4103,
	5101, 5102, 5103,
	3001, 3004, 3007,
	3201, 3204, 3207,
	3301, 3304, 3307,

	// [0][j]
	2001, 2201, 2301, 4001, 4201, 4301, 5001, 5201, 5301,
	2002, 2202, 2302, 4002, 4202, 4302, 5002, 5202, 5302,
	2003, 2203, 2303, 4003, 4203, 4303, 5003, 5203, 5303,

	// [1]
	1002, 1202, 1302,
	2104, 2105, 2106,
	4104, 4105, 4106,
	5104, 5105, 5106,
	3002, 3005, 3008,
	3202, 3205, 3208,
	3302, 3305, 3308,

	// [1][j]
	2004, 2204, 2304, 4004, 4204, 4304, 5004, 5204, 5304,
	2005, 2205, 2305, 4005, 4205, 4305, 5005, 5205, 5305,
	2006, 2206, 2306, 4006, 4206, 4306, 5006, 5206, 5306,

	// [2]
	1003, 1203, 1303,
	2107, 2108, 2109,
	4107, 4108, 4109,
	5107, 5108, 5109,
	3003, 3006, 3009,
	3203, 3206, 3209,
	3303, 3306, 3309,

	// [2][j]
	2007, 2207, 2307, 4007, 4207, 4307, 5007, 5207, 5307,
	2008, 2208, 2308, 4008, 4208, 4308, 5008, 5208, 5308,
	2009, 2209, 2309, 4009, 4209, 4309, 5009, 5209, 5309,
}

var (
	copy_zero     = zero
	copy_one      = one
	copy_pi       = pi
	copy_slice    = slice
	copy_sliceInt = sliceInt
	copy_hello    = hello

	// Could be handled without an initialization function, but
	// requires special handling for "a = []byte("..."); b = a"
	// which is not a likely case.
	// copy_bytes = bytes
	// https://codereview.appspot.com/171840043 is one approach to
	// make this special case work.

	copy_four, copy_five = four, five
	copy_x, copy_y       = x, y
	copy_nilslice        = nilslice
	copy_nilmap          = nilmap
	copy_nilfunc         = nilfunc
	copy_nilchan         = nilchan
	copy_nilptr          = nilptr
)

var copy_a = a
var copy_s = s
var copy_c = c

var copy_aa = aa
var copy_as = as
var copy_ac = ac

var copy_sa = sa
var copy_ss = ss
var copy_sc = sc

var copy_ca = ca
var copy_cs = cs
var copy_cc = cc

var copy_answers = answers

var bx bool
var b0 = false
var b1 = true

var fx float32
var f0 = float32(0)
var f1 = float32(1)

var gx float64
var g0 = float64(0)
var g1 = float64(1)

var ix int
var i0 = 0
var i1 = 1

var jx uint
var j0 = uint(0)
var j1 = uint(1)

var cx complex64
var c0 = complex64(0)
var c1 = complex64(1)

var dx complex128
var d0 = complex128(0)
var d1 = complex128(1)

var sx []int
var s0 = []int{0, 0, 0}
var s1 = []int{1, 2, 3}

func fi() int { return 1 }

var ax [10]int
var a0 = [10]int{0, 0, 0}
var a1 = [10]int{1, 2, 3, 4}

type T struct{ X, Y int }

var tx T
var t0 = T{}
var t0a = T{0, 0}
var t0b = T{X: 0}
var t1 = T{X: 1, Y: 2}
var t1a = T{3, 4}

var psx *[]int
var ps0 = &[]int{0, 0, 0}
var ps1 = &[]int{1, 2, 3}

var pax *[10]int
var pa0 = &[10]int{0, 0, 0}
var pa1 = &[10]int{1, 2, 3}

var ptx *T
var pt0 = &T{}
var pt0a = &T{0, 0}
var pt0b = &T{X: 0}
var pt1 = &T{X: 1, Y: 2}
var pt1a = &T{3, 4}

// The checks similar to
// var copy_bx = bx
// are commented out.  The  compiler no longer statically initializes them.
// See issue 7665 and https://codereview.appspot.com/93200044.
// If https://codereview.appspot.com/169040043 is submitted, and this
// test is changed to pass -complete to the compiler, then we can
// uncomment the copy lines again.

// var copy_bx = bx
var copy_b0 = b0
var copy_b1 = b1

// var copy_fx = fx
var copy_f0 = f0
var copy_f1 = f1

// var copy_gx = gx
var copy_g0 = g0
var copy_g1 = g1

// var copy_ix = ix
var copy_i0 = i0
var copy_i1 = i1

// var copy_jx = jx
var copy_j0 = j0
var copy_j1 = j1

// var copy_cx = cx
var copy_c0 = c0
var copy_c1 = c1

// var copy_dx = dx
var copy_d0 = d0
var copy_d1 = d1

// var copy_sx = sx
var copy_s0 = s0
var copy_s1 = s1

// var copy_ax = ax
var copy_a0 = a0
var copy_a1 = a1

// var copy_tx = tx
var copy_t0 = t0
var copy_t0a = t0a
var copy_t0b = t0b
var copy_t1 = t1
var copy_t1a = t1a

// var copy_psx = psx
var copy_ps0 = ps0
var copy_ps1 = ps1

// var copy_pax = pax
var copy_pa0 = pa0
var copy_pa1 = pa1

// var copy_ptx = ptx
var copy_pt0 = pt0
var copy_pt0a = pt0a
var copy_pt0b = pt0b
var copy_pt1 = pt1
var copy_pt1a = pt1a

var _ interface{} = 1

type T1 int

func (t *T1) M() {}

type Mer interface {
	M()
}

var _ Mer = (*T1)(nil)

var Byte byte
var PtrByte unsafe.Pointer = unsafe.Pointer(&Byte)
