// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test conversion from non-interface types to the empty interface.

package main

var (
	z    = struct{}{}
	p    = &z
	pp   = &p
	u16  = uint16(1)
	u32  = uint32(2)
	u64  = uint64(3)
	u128 = [2]uint64{4, 5}
	f32  = float32(6)
	f64  = float64(7)
	c128 = complex128(8 + 9i)
	s    = "10"
	b    = []byte("11")
	m    = map[int]int{12: 13}
	c    = make(chan int, 14)
)

var (
	iz    interface{} = z
	ip    interface{} = p
	ipp   interface{} = pp
	iu16  interface{} = u16
	iu32  interface{} = u32
	iu64  interface{} = u64
	iu128 interface{} = u128
	if32  interface{} = f32
	if64  interface{} = f64
	ic128 interface{} = c128
	is    interface{} = s
	ib    interface{} = b
	im    interface{} = m
	ic    interface{} = c
)

func second(a ...interface{}) interface{} {
	return a[1]
}

func main() {
	// Test equality. There are no tests for b and m, as slices and
	// maps are not comparable by ==.
	if z != iz {
		panic("z != iz")
	}
	if p != ip {
		panic("p != ip")
	}
	if pp != ipp {
		panic("pp != ipp")
	}
	if u16 != iu16 {
		panic("u16 != iu16")
	}
	if u32 != iu32 {
		panic("u32 != iu32")
	}
	if u64 != iu64 {
		panic("u64 != iu64")
	}
	if u128 != iu128 {
		panic("u128 != iu128")
	}
	if f32 != if32 {
		panic("f32 != if32")
	}
	if f64 != if64 {
		panic("f64 != if64")
	}
	if c128 != ic128 {
		panic("c128 != ic128")
	}
	if s != is {
		panic("s != is")
	}
	if c != ic {
		panic("c != ic")
	}

	// Test that non-interface types can be used as ...interface{} arguments.
	if got := second(z, p, pp, u16, u32, u64, u128, f32, f64, c128, s, b, m, c); got != ip {
		println("second: got", got, "want", ip)
		panic("fail")
	}

	// Test that non-interface types can be sent on a chan interface{}.
	const n = 100
	uc := make(chan interface{})
	go func() {
		for i := 0; i < n; i++ {
			select {
			case uc <- nil:
			case uc <- u32:
			case uc <- u64:
			case uc <- u128:
			}
		}
	}()
	for i := 0; i < n; i++ {
		if got := <-uc; got != nil && got != u32 && got != u64 && got != u128 {
			println("recv: i", i, "got", got)
			panic("fail")
		}
	}
}
