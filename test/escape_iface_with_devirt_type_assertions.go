// errorcheck -0 -m

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

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

	any(a).(M).M()           // ERROR "devirtualizing" "inlining call"
	any(a).(A).A()           // ERROR "devirtualizing" "inlining call"
	any(a).(M).(any).(A).A() // ERROR "devirtualizing" "inlining call"

	c := any(a)
	c.(A).A() // ERROR "devirtualizing" "inlining call"
	c.(M).M() // ERROR "devirtualizing" "inlining call"

	{
		var a C = &CImpl{}   // ERROR "does not escape"
		a.(any).(C).C()      // ERROR "devirtualizing" "inlining"
		a.(any).(*CImpl).C() // ERROR "inlining"
	}
}

func t2() {
	{
		var a M = &Impl{} // ERROR "does not escape"
		if v, ok := a.(M); ok {
			v.M() // ERROR "devirtualizing" "inlining call"
		}
	}
	{
		var a M = &Impl{} // ERROR "does not escape"
		if v, ok := a.(A); ok {
			v.A() // ERROR "devirtualizing" "inlining call"
		}
	}
	{
		var a M = &Impl{} // ERROR "does not escape"
		v, ok := a.(M)
		if ok {
			v.M() // ERROR "devirtualizing" "inlining call"
		}
	}
	{
		var a M = &Impl{} // ERROR "does not escape"
		v, ok := a.(A)
		if ok {
			v.A() // ERROR "devirtualizing" "inlining call"
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
		var a M = &Impl{} // ERROR "does not escape"
		v, _ := a.(M)
		v.M() // ERROR "devirtualizing" "inlining call"
	}
	{
		var a M = &Impl{} // ERROR "does not escape"
		v, _ := a.(A)
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var a M = &Impl{} // ERROR "does not escape"
		v, _ := a.(*Impl)
		v.A() // ERROR "inlining"
		v.M() // ERROR "inlining"
	}
	{
		a := newM() // ERROR "does not escape" "inlining call"
		callA(a)    // ERROR "devirtualizing" "inlining call"
		callIfA(a)  // ERROR "devirtualizing" "inlining call"
	}

	{
		var a M = &Impl{} // ERROR "does not escape"
		// Note the !ok condition, devirtualizing here is fine.
		if v, ok := a.(M); !ok {
			v.M() // ERROR "devirtualizing" "inlining call"
		}
	}
}

func newM() M { // ERROR "can inline"
	return &Impl{} // ERROR "escapes"
}

func callA(m M) { // ERROR "can inline" "leaking param"
	m.(A).A()
}

func callIfA(m M) { // ERROR "can inline" "leaking param"
	if v, ok := m.(A); ok {
		v.A()
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
