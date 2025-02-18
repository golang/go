// errorcheck -0 -m -d=testing=2

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

func typeAsserts() {
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

func typeAssertsWithOkReturn() {
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
	{
		var a A = newImplNoInline()
		if v, ok := a.(M); ok {
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

//go:noinline
func newImpl2ret2() (string, *Impl2) {
	return "str", &Impl2{} // ERROR "escapes"
}

//go:noinline
func newImpl2() *Impl2 {
	return &Impl2{} // ERROR "escapes"
}

func differentTypeAssign() {
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
}

func assignWithTypeAssert() {
	var i1 A = &Impl{}  // ERROR "does not escape"
	var i2 A = &Impl2{} // ERROR "does not escape"
	i1 = i2.(*Impl)     // this will panic
	i1.A()              // ERROR "devirtualizing i1\.A to \*Impl" "inlining call"
	i2.A()              // ERROR "devirtualizing i2\.A to \*Impl2" "inlining call"
}

func longDevirtTest() {
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

func deferDevirt() {
	var a A
	defer func() { // ERROR "func literal does not escape" "can inline"
		a = &Impl{} // ERROR "escapes"
	}()
	a = &Impl{} // ERROR "does not escape"
	a.A()       // ERROR "devirtualizing" "inlining call"
}

func deferNoDevirt() {
	var a A
	defer func() { // ERROR "func literal does not escape" "can inline"
		a = &Impl2{} // ERROR "escapes"
	}()
	a = &Impl{} // ERROR "escapes"
	a.A()
}

func closureDevirt() {
	var a A
	func() { // ERROR "func literal does not escape"
		// defer so that it does not lnline.
		defer func() {}() // ERROR "can inline" "func literal does not escape"
		a = &Impl{}       // ERROR "escapes"
	}()
	a = &Impl{} // ERROR "does not escape"
	a.A()       // ERROR "devirtualizing" "inlining call"
}

func closureNoDevirt() {
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

func closureDevirt2() {
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

func closureNoDevirt2() {
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

var (
	globalImpl    = &Impl{}
	globalImpl2   = &Impl2{}
	globalA     A = &Impl{}
	globalM     M = &Impl{}
)

func globals() {
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
		var a A = &Impl{} // ERROR "does not escape"
		a = globalA.(*Impl)
		a.A() // ERROR "devirtualizing" "inlining call"
		a = globalM.(*Impl)
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
}

func mapsDevirt() {
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		var v A = m[0]
		v.A()     // ERROR "devirtualizing" "inlining call"
		v.(M).M() // ERROR "devirtualizing" "inlining call"
	}
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		var v A
		var ok bool
		if v, ok = m[0]; ok {
			v.A() // ERROR "devirtualizing" "inlining call"
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		var v A
		v, _ = m[0]
		v.A() // ERROR "devirtualizing" "inlining call"
	}
}

func mapsNoDevirt() {
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		var v A = m[0]
		v.A()
		v = &Impl2{} // ERROR "escapes"
		v.(M).M()
	}
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		var v A
		var ok bool
		if v, ok = m[0]; ok {
			v.A()
		}
		v = &Impl2{} // ERROR "escapes"
		v.A()
	}
	{
		m := make(map[int]*Impl) // ERROR "does not escape"
		var v A
		v, _ = m[0]
		v.A()
		v = &Impl2{} // ERROR "escapes"
		v.A()
	}
}

func chanDevirt() {
	{
		m := make(chan *Impl)
		var v A = <-m
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		m := make(chan *Impl)
		var v A
		v = <-m
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		m := make(chan *Impl)
		var v A
		v, _ = <-m
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		m := make(chan *Impl)
		var v A
		var ok bool
		if v, ok = <-m; ok {
			v.A() // ERROR "devirtualizing" "inlining call"
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		m := make(chan *Impl)
		var v A
		var ok bool
		if v, ok = <-m; ok {
			v.A() // ERROR "devirtualizing" "inlining call"
		}
		select {
		case <-m:
			v.A() // ERROR "devirtualizing" "inlining call"
		case v = <-m:
			v.A() // ERROR "devirtualizing" "inlining call"
		case v, ok = <-m:
			v.A() // ERROR "devirtualizing" "inlining call"
		}
	}
}

func chanNoDevirt() {
	{
		m := make(chan *Impl)
		var v A = <-m
		v = &Impl2{} // ERROR "escapes"
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A
		v = <-m
		v = &Impl2{} // ERROR "escapes"
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A
		v, _ = <-m
		v = &Impl2{} // ERROR "escapes"
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A
		var ok bool
		if v, ok = <-m; ok {
			v.A()
		}
		v = &Impl2{} // ERROR "escapes"
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A = &Impl2{} // ERROR "escapes"
		var ok bool
		if v, ok = <-m; ok {
			v.A()
		}
	}
	{
		m := make(chan *Impl)
		var v A = &Impl2{} // ERROR "escapes"
		select {
		case v = <-m:
			v.A()
		}
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A = &Impl2{} // ERROR "escapes"
		select {
		case v, _ = <-m:
			v.A()
		}
		v.A()
	}
}

func rangeDevirt() {
	{
		var v A
		m := make(map[*Impl]struct{}) // ERROR "does not escape"
		v = &Impl{}                   // ERROR "does not escape"
		for v = range m {
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var v A
		m := make(map[*Impl]*Impl) // ERROR "does not escape"
		v = &Impl{}                // ERROR "does not escape"
		for v = range m {
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var v A
		m := make(map[*Impl]*Impl) // ERROR "does not escape"
		v = &Impl{}                // ERROR "does not escape"
		for _, v = range m {
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var v A
		m := make(chan *Impl)
		v = &Impl{} // ERROR "does not escape"
		for v = range m {
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var v A
		m := []*Impl{} // ERROR "does not escape"
		v = &Impl{}    // ERROR "does not escape"
		for _, v = range m {
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var v A
		v = &Impl{}     // ERROR "does not escape"
		impl := &Impl{} // ERROR "does not escape"
		i := 0
		for v = impl; i < 10; i++ {
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var v A
		v = &Impl{}     // ERROR "does not escape"
		impl := &Impl{} // ERROR "does not escape"
		i := 0
		for v = impl; i < 10; i++ {
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var v A
		m := [1]*Impl{&Impl{}} // ERROR "does not escape"
		v = &Impl{}            // ERROR "does not escape"
		for _, v = range m {
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
	{
		var v A
		m := [1]*Impl{&Impl{}} // ERROR "does not escape"
		v = &Impl{}            // ERROR "does not escape"
		for _, v = range &m {
		}
		v.A() // ERROR "devirtualizing" "inlining call"
	}
}

func rangeNoDevirt() {
	{
		var v A = &Impl2{}            // ERROR "escapes"
		m := make(map[*Impl]struct{}) // ERROR "does not escape"
		for v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl2{}         // ERROR "escapes"
		m := make(map[*Impl]*Impl) // ERROR "does not escape"
		for v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl2{}         // ERROR "escapes"
		m := make(map[*Impl]*Impl) // ERROR "does not escape"
		for _, v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl2{} // ERROR "escapes"
		m := make(chan *Impl)
		for v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl2{} // ERROR "escapes"
		m := []*Impl{}     // ERROR "does not escape"
		for _, v = range m {
		}
		v.A()
	}
	{
		var v A
		v = &Impl2{}    // ERROR "escapes"
		impl := &Impl{} // ERROR "escapes"
		i := 0
		for v = impl; i < 10; i++ {
		}
		v.A()
	}
	{
		var v A
		v = &Impl2{}    // ERROR "escapes"
		impl := &Impl{} // ERROR "escapes"
		i := 0
		for v = impl; i < 10; i++ {
		}
		v.A()
	}
	{
		var v A
		m := [1]*Impl{&Impl{}} // ERROR "escapes"
		v = &Impl2{}           // ERROR "escapes"
		for _, v = range m {
		}
		v.A()
	}
	{
		var v A
		m := [1]*Impl{&Impl{}} // ERROR "escapes"
		v = &Impl2{}           // ERROR "escapes"
		for _, v = range &m {
		}
		v.A()
	}
}

type implWrapper Impl

func (implWrapper) A() {} // ERROR "can inline"

//go:noinline
func devirtWrapperType() {
	{
		i := &Impl{} // ERROR "does not escape"
		// This is an OCONVNOP, so we have to be carefull, not to devirtualize it to Impl.A.
		var a A = (*implWrapper)(i)
		a.A() // ERROR "devirtualizing a.A to \*implWrapper" "inlining call"
	}
	{
		i := Impl{}
		// This is an OCONVNOP, so we have to be carefull, not to devirtualize it to Impl.A.
		var a A = (implWrapper)(i) // ERROR "does not escape"
		a.A()                      // ERROR "devirtualizing a.A to implWrapper" "inlining call"
	}
}

func selfAssigns() {
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

func addrTaken() {
	{
		var a A = &Impl{} // ERROR "escapes"
		var ptrA = &a
		a.A()
		_ = ptrA
	}
	{
		var a A = &Impl{} // ERROR "escapes"
		var ptrA = &a
		*ptrA = &Impl{} // ERROR "escapes"
		a.A()
	}
	{
		var a A = &Impl{} // ERROR "escapes"
		var ptrA = &a
		*ptrA = &Impl2{} // ERROR "escapes"
		a.A()
	}
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
