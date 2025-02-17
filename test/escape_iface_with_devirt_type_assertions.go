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

type Impl2 struct{}

func (*Impl2) M() {} // ERROR "can inline"

func (*Impl2) A() {} // ERROR "can inline"

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
func newImplNoInline() *Impl {
	return &Impl{} // ERROR "escapes"
}

func t3() {
	{
		var a A = newImplNoInline()
		if v, ok := a.(M); ok {
			v.M() // ERROR "devirtualizing" "inlining call"
		}
	}
	{
		m := make(map[*Impl]struct{}) // ERROR "does not escape"
		for v := range m {
			var v A = v
			v.A() // ERROR "devirtualizing" "inlining call"
			if v, ok := v.(M); ok {
				v.M() // ERROR "devirtualizing" "inlining call"
			}
		}
	}
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		for _, v := range m {
			var v A = v
			v.A() // ERROR "devirtualizing" "inlining call"
			if v, ok := v.(M); ok {
				v.M() // ERROR "devirtualizing" "inlining call"
			}
		}
	}
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		var v A = m[0]
		v.A() // ERROR "devirtualizing" "inlining call"
		if v, ok := v.(M); ok {
			v.M() // ERROR "devirtualizing" "inlining call"
		}
	}
	{
		m := make(chan *Impl)
		var v A = <-m
		v.A() // ERROR "devirtualizing" "inlining call"
		if v, ok := v.(M); ok {
			v.M() // ERROR "devirtualizing" "inlining call"
		}
	}
	{
		m := make(chan *Impl)
		var v A
		var ok bool
		if v, ok = <-m; ok {
			v.A()     // ERROR "devirtualizing" "inlining call"
			v.(M).M() // ERROR "devirtualizing" "inlining call"
		}
		select {
		case <-m:
			v.A()     // ERROR "devirtualizing" "inlining call"
			v.(M).M() // ERROR "devirtualizing" "inlining call"
		case v = <-m:
			v.A()     // ERROR "devirtualizing" "inlining call"
			v.(M).M() // ERROR "devirtualizing" "inlining call"
		case v, ok = <-m:
			v.A()     // ERROR "devirtualizing" "inlining call"
			v.(M).M() // ERROR "devirtualizing" "inlining call"
		}
	}
}

//go:noinline
func newImpl2ret2() (string, *Impl2) {
	return "str", &Impl2{} // ERROR "escapes"
}

//go:noinline
func newImpl2() *Impl2 {
	return &Impl2{} // ERROR "escapes"
}

func t5() {
	{
		var a A
		a = &Impl{}  // ERROR "escapes"
		a = &Impl2{} // ERROR "escapes"
		a.A()
	}
	{
		a := A(&Impl{}) // ERROR "escapes"
		a = &Impl2{}    // ERROR "escapes"
		a.A()
	}
	{
		a := A(&Impl{}) // ERROR "escapes"
		a.A()
		a = &Impl2{} // ERROR "escapes"
	}
	{
		a := A(&Impl{}) // ERROR "escapes"
		a = &Impl2{}    // ERROR "escapes"
		var asAny any = a
		asAny.(A).A()
	}
	{
		a := A(&Impl{}) // ERROR "escapes"
		var asAny any = a
		asAny = &Impl2{} // ERROR "escapes"
		asAny.(A).A()
	}
	{
		a := A(&Impl{}) // ERROR "escapes"
		var asAny any = a
		asAny.(A).A()
		asAny = &Impl2{} // ERROR "escapes"
		a.A()            // ERROR "devirtualizing" "inlining call"
	}
	{
		var a A
		a = &Impl{} // ERROR "escapes"
		a = newImpl2()
		a.A()
	}
	{
		var a A
		a = &Impl{} // ERROR "escapes"
		_, a = newImpl2ret2()
		a.A()
	}
	{
		var a A
		a = &Impl{}               // ERROR "escapes"
		m := make(map[int]*Impl2) // ERROR "does not escape"
		a = m[0]
		a.A()
	}
	{
		var a A
		a = &Impl{} // ERROR "escapes"
		m := make(chan *Impl2)
		a = <-m
		a.A()
	}
}

func t6() {
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		var a A
		a, _ = m[0]
		if v, ok := a.(M); ok {
			v.M() // ERROR "devirtualizing" "inlining call"
		}
	}
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		var a A
		var ok bool
		if a, ok = m[0]; ok {
			if v, ok := a.(M); ok {
				v.M() // ERROR "devirtualizing" "inlining call"
			}
		}
	}
	{
		m := make(chan *Impl)
		var a A
		a, _ = <-m
		if v, ok := a.(M); ok {
			v.M() // ERROR "devirtualizing" "inlining call"
		}
	}
	{
		m := make(chan *Impl)
		var a A
		var ok bool
		if a, ok = <-m; ok {
			if v, ok := a.(M); ok {
				v.M() // ERROR "devirtualizing" "inlining call"
			}
		}
	}
}

var (
	globalImpl    = &Impl{}
	globalImpl2   = &Impl2{}
	globalA     A = &Impl{}
	globalM     M = &Impl{}
)

func t7() {
	{
		var a A = &Impl{} // ERROR "does not escape"
		a = globalImpl
		a.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var a A = &Impl{} // ERROR "does not escape"
		a = A(globalImpl)
		a.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var a A = &Impl{} // ERROR "does not escape"
		a = M(globalImpl).(A)
		a.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var a A = &Impl{} // ERROR "escapes"
		a = globalImpl2
		a.A()
	}
	{
		var a A = &Impl{} // ERROR "escapes"
		a = globalA
		a.A()
	}
	{
		var a A = &Impl{} // ERROR "escapes"
		a = globalM.(A)
		a.A()
	}
	{
		var a A = &Impl{}                    // ERROR "does not escape"
		for _, v := range []*Impl{&Impl{}} { // ERROR "does not escape"
			a = v
		}

		k, v := &Impl{}, &Impl{}                  // ERROR "escapes"
		for k, v := range map[*Impl]*Impl{k: v} { // ERROR "does not escape"
			a = k
			a = v
		}

		a.A()     // ERROR "devirtualizing" "inlining call"
		a.(A).A() // ERROR "devirtualizing""inlining call"
		a.(M).M() // ERROR "devirtualizing""inlining call"

		var m M = a.(M)
		m.M()     // ERROR "devirtualizing""inlining call"
		m.(A).A() // ERROR "devirtualizing""inlining call"
	}
	{
		var a A = &Impl{}                   // ERROR "escapes"
		var impl2 = &Impl2{}                // ERROR "escapes"
		for _, v := range []*Impl2{impl2} { // ERROR "does not escape"
			a = v
		}
		a.A()
	}
	{
		var a A = &Impl{}                           // ERROR "escapes"
		k, v := &Impl2{}, &Impl2{}                  // ERROR "escapes"
		for k, _ := range map[*Impl2]*Impl2{k: v} { // ERROR "does not escape"
			a = k
		}
		a.A()
	}
	{
		var a A = &Impl{}                           // ERROR "escapes"
		k, v := &Impl2{}, &Impl2{}                  // ERROR "escapes"
		for _, v := range map[*Impl2]*Impl2{k: v} { // ERROR "does not escape"
			a = v
		}
		a.A()
	}
}

func t8() {
	{
		var a A = &Impl{} // ERROR "escapes"
		a = a
		a.A()
	}
	{
		var a A = &Impl{} // ERROR "escapes"
		var asAny any = a
		asAny = asAny
		asAny.(A).A()
	}
	{
		var a A = &Impl{} // ERROR "escapes"
		var asAny any = a
		asAny = asAny
		a = asAny.(A)
		asAny = a
		asAny.(A).A()
		asAny.(M).M()
	}
	{
		var a A = &Impl{} // ERROR "escapes"
		var asAny A = a
		a = asAny.(A)
		a.A()
	}
}

func t9() {
	var a interface {
		M
		A
	} = &Impl{} // ERROR "does not escape"

	{
		var b A = a
		b.A()     // ERROR "devirtualizing" "inlining call"
		b.(M).M() // ERROR "devirtualizing" "inlining call"
	}
	{
		var b M = a
		b.M()     // ERROR "devirtualizing" "inlining call"
		b.(A).A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var b A = a.(M).(A)
		b.A()     // ERROR "devirtualizing" "inlining call"
		b.(M).M() // ERROR "devirtualizing" "inlining call"
	}
	{
		var b M = a.(A).(M)
		b.M()     // ERROR "devirtualizing" "inlining call"
		b.(A).A() // ERROR "devirtualizing" "inlining call"
	}

	if v, ok := a.(A); ok {
		v.A() // ERROR "devirtualizing" "inlining call"
	}

	if v, ok := a.(M); ok {
		v.M() // ERROR "devirtualizing" "inlining call"
	}

	{
		var c A = a

		if v, ok := c.(A); ok {
			v.A() // ERROR "devirtualizing" "inlining call"
		}

		c = &Impl{} // ERROR "does not escape"

		if v, ok := c.(M); ok {
			v.M() // ERROR "devirtualizing" "inlining call"
		}

		if v, ok := c.(interface {
			A
			M
		}); ok {
			v.M() // ERROR "devirtualizing" "inlining call"
			v.A() // ERROR "devirtualizing" "inlining call"
		}
	}
}

func t10() {
	var a A
	defer func() { // ERROR "func literal does not escape" "can inline"
		a = &Impl{} // ERROR "escapes"
	}()
	a = &Impl{} // ERROR "does not escape"
	a.A()       // ERROR "devirtualizing" "inlining call"
}

func t11() {
	var a A
	defer func() { // ERROR "func literal does not escape" "can inline"
		a = &Impl2{} // ERROR "escapes"
	}()
	a = &Impl{} // ERROR "escapes"
	a.A()
}

func t12() {
	var a A
	func() { // ERROR "func literal does not escape"
		// defer so that it does not lnline.
		defer func() {}() // ERROR "can inline" "func literal does not escape"
		a = &Impl{}       // ERROR "escapes"
	}()
	a = &Impl{} // ERROR "does not escape"
	a.A()       // ERROR "devirtualizing" "inlining call"
}

func t13() {
	var a A
	func() { // ERROR "func literal does not escape"
		// defer so that it does not lnline.
		defer func() {}() // ERROR "can inline" "func literal does not escape"
		a = &Impl2{}      // ERROR "escapes"
	}()
	a = &Impl{} // ERROR "escapes"
	a.A()
}

var global = "1"

func t14() {
	var a A
	a = &Impl{}   // ERROR "does not escape"
	c := func() { // ERROR "can inline" "func literal does not escape"
		a = &Impl{} // ERROR "escapes"
	}
	if global == "1" {
		c = func() { // ERROR "can inline" "func literal does not escape"
			a = &Impl{} // ERROR "escapes"
		}
	}
	a.A() // ERROR "devirtualizing" "inlining call"
	c()
}

func t15() {
	var a A
	a = &Impl{}   // ERROR "escapes"
	c := func() { // ERROR "can inline" "func literal does not escape"
		a = &Impl2{} // ERROR "escapes"
	}
	if global == "1" {
		c = func() { // ERROR "can inline" "func literal does not escape"
			a = &Impl{} // ERROR "escapes"
		}
	}
	a.A()
	c()
}

type implWrapper Impl

func (implWrapper) A() {} // ERROR "can inline"

//go:noinline
func t16() {
	i := &Impl{} // ERROR "does not escape"
	var a A = (*implWrapper)(i)
	a.A() // ERROR "devirtualizing a.A to \*implWrapper" "inlining call"
}

func testInvalidAsserts() {
	any(0).(interface{ A() }).A() // ERROR "escapes"
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
