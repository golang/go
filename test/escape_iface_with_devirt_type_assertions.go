// errorcheck -0 -m

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"crypto/sha256"
	"encoding"
	"hash"
	"io"
)

type M interface{ M() }

type A interface{ A() }

type C interface{ C() }

type Impl struct{}

func (*Impl) M() {} // ERROR "can inline"

func (*Impl) A() {} // ERROR "can inline"

type CImpl struct{}

func (CImpl) C() {} // ERROR "can inline"

func t() {
	var a M = &Impl{} // ERROR "&Impl{} does not escape"

	a.(M).M()     // ERROR "devirtualizing a.\(M\).M" "inlining call"
	a.(A).A()     // ERROR "devirtualizing a.\(A\).A" "inlining call"
	a.(*Impl).M() // ERROR "inlining call"
	a.(*Impl).A() // ERROR "inlining call"

	v := a.(M)
	v.M()         // ERROR "devirtualizing v.M" "inlining call"
	v.(A).A()     // ERROR "devirtualizing v.\(A\).A" "inlining call"
	v.(*Impl).A() // ERROR "inlining call"
	v.(*Impl).M() // ERROR "inlining call"

	v2 := a.(A)
	v2.A()         // ERROR "devirtualizing v2.A" "inlining call"
	v2.(M).M()     // ERROR "devirtualizing v2.\(M\).M" "inlining call"
	v2.(*Impl).A() // ERROR "inlining call"
	v2.(*Impl).M() // ERROR "inlining call"

	a.(M).(A).A() // ERROR "devirtualizing a.\(M\).\(A\).A" "inlining call"
	a.(A).(M).M() // ERROR "devirtualizing a.\(A\).\(M\).M" "inlining call"

	a.(M).(A).(*Impl).A() // ERROR "inlining call"
	a.(A).(M).(*Impl).M() // ERROR "inlining call"

	{
		var a C = &CImpl{}   // ERROR "does not escape"
		a.(any).(C).C()      // ERROR "devirtualizing" "inlining"
		a.(any).(*CImpl).C() // ERROR "inlining"
	}
}

// TODO: these type assertions could also be devirtualized.
func t2() {
	{
		var a M = &Impl{} // ERROR "&Impl{} escapes to heap"
		if v, ok := a.(M); ok {
			v.M()
		}
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} escapes to heap"
		if v, ok := a.(A); ok {
			v.A()
		}
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} escapes to heap"
		v, ok := a.(M)
		if ok {
			v.M()
		}
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} escapes to heap"
		v, ok := a.(A)
		if ok {
			v.A()
		}
	}
	{
		var a M = &Impl{} // ERROR "does not escape"
		v, ok := a.(*Impl)
		if ok {
			v.A() // ERROR "inlining"
			v.M() // ERROR "inlining"
		}
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} escapes to heap"
		v, _ := a.(M)
		v.M()
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} escapes to heap"
		v, _ := a.(A)
		v.A()
	}
	{
		var a M = &Impl{} // ERROR "does not escape"
		v, _ := a.(*Impl)
		v.A() // ERROR "inlining"
		v.M() // ERROR "inlining"
	}
}

//go:noinline
func testInvalidAsserts() {
	{
		var a M = &Impl{} // ERROR "escapes"
		a.(C).C()         // this will panic
		a.(any).(C).C()   // this will panic
	}
	{
		var a C = &CImpl{} // ERROR "escapes"
		a.(M).M()          // this will panic
		a.(any).(M).M()    // this will panic
	}
	{
		var a C = &CImpl{} // ERROR "does not escape"

		// this will panic
		a.(M).(*Impl).M() // ERROR "inlining"

		// this will panic
		a.(any).(M).(*Impl).M() // ERROR "inlining"
	}
}

func testSha256() {
	h := sha256.New()                                   // ERROR "inlining call" "does not escape"
	h.Write(nil)                                        // ERROR "devirtualizing"
	h.(io.Writer).Write(nil)                            // ERROR "devirtualizing"
	h.(hash.Hash).Write(nil)                            // ERROR "devirtualizing"
	h.(encoding.BinaryUnmarshaler).UnmarshalBinary(nil) // ERROR "devirtualizing"

	h2 := sha256.New() // ERROR "escapes" "inlining call"
	h2.(M).M()         // this will panic
}
