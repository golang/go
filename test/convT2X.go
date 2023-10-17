// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test conversion from non-interface types to the empty interface.

package main

type J interface {
	Method()
}

type (
	U16  uint16
	U32  uint32
	U64  uint64
	U128 [2]uint64
	F32  float32
	F64  float64
	C128 complex128
	S    string
	B    []byte
	M    map[int]int
	C    chan int
	Z    struct{}
)

func (U16) Method()  {}
func (U32) Method()  {}
func (U64) Method()  {}
func (U128) Method() {}
func (F32) Method()  {}
func (F64) Method()  {}
func (C128) Method() {}
func (S) Method()    {}
func (B) Method()    {}
func (M) Method()    {}
func (C) Method()    {}
func (Z) Method()    {}

var (
	u16  = U16(1)
	u32  = U32(2)
	u64  = U64(3)
	u128 = U128{4, 5}
	f32  = F32(6)
	f64  = F64(7)
	c128 = C128(8 + 9i)
	s    = S("10")
	b    = B("11")
	m    = M{12: 13}
	c    = make(C, 14)
	z    = Z{}
	p    = &z
	pp   = &p
)

var (
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
	iz    interface{} = z
	ip    interface{} = p
	ipp   interface{} = pp

	ju16  J = u16
	ju32  J = u32
	ju64  J = u64
	ju128 J = u128
	jf32  J = f32
	jf64  J = f64
	jc128 J = c128
	js    J = s
	jb    J = b
	jm    J = m
	jc    J = c
	jz J = z
	jp J = p // The method set for *T contains the methods for T.
	// pp does not implement error.
)

func second(a ...interface{}) interface{} {
	return a[1]
}

func main() {
	// Test equality.
	if u16 != iu16 {
		panic("u16 != iu16")
	}
	if u16 != ju16 {
		panic("u16 != ju16")
	}
	if u32 != iu32 {
		panic("u32 != iu32")
	}
	if u32 != ju32 {
		panic("u32 != ju32")
	}
	if u64 != iu64 {
		panic("u64 != iu64")
	}
	if u64 != ju64 {
		panic("u64 != ju64")
	}
	if u128 != iu128 {
		panic("u128 != iu128")
	}
	if u128 != ju128 {
		panic("u128 != ju128")
	}
	if f32 != if32 {
		panic("f32 != if32")
	}
	if f32 != jf32 {
		panic("f32 != jf32")
	}
	if f64 != if64 {
		panic("f64 != if64")
	}
	if f64 != jf64 {
		panic("f64 != jf64")
	}
	if c128 != ic128 {
		panic("c128 != ic128")
	}
	if c128 != jc128 {
		panic("c128 != jc128")
	}
	if s != is {
		panic("s != is")
	}
	if s != js {
		panic("s != js")
	}
	if c != ic {
		panic("c != ic")
	}
	if c != jc {
		panic("c != jc")
	}
	// There are no tests for b and m, as slices and maps are not comparable by ==.
	if z != iz {
		panic("z != iz")
	}
	if z != jz {
		panic("z != jz")
	}
	if p != ip {
		panic("p != ip")
	}
	if p != jp {
		panic("p != jp")
	}
	if pp != ipp {
		panic("pp != ipp")
	}
	// pp does not implement J.

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
