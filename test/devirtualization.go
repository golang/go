// errorcheck -0 -m

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

type M interface{ M() }

type A interface{ A() }

type C interface{ C() }

type Impl struct{}

func (*Impl) M() {} // ERROR "can inline \(\*Impl\).M$"

func (*Impl) A() {} // ERROR "can inline \(\*Impl\).A$"

type Impl2 struct{}

func (*Impl2) M() {} // ERROR "can inline \(\*Impl2\).M$"

func (*Impl2) A() {} // ERROR "can inline \(\*Impl2\).A$"

type CImpl struct{}

func (CImpl) C() {} // ERROR "can inline CImpl.C$"

func typeAsserts() {
	var a M = &Impl{} // ERROR "&Impl{} does not escape$"

	a.(M).M()     // ERROR "devirtualizing a.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	a.(A).A()     // ERROR "devirtualizing a.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	a.(*Impl).M() // ERROR "inlining call to \(\*Impl\).M"
	a.(*Impl).A() // ERROR "inlining call to \(\*Impl\).A"

	v := a.(M)
	v.M()         // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
	v.(A).A()     // ERROR "devirtualizing v.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	v.(*Impl).A() // ERROR "inlining call to \(\*Impl\).A"
	v.(*Impl).M() // ERROR "inlining call to \(\*Impl\).M"

	v2 := a.(A)
	v2.A()         // ERROR "devirtualizing v2.A to \*Impl$" "inlining call to \(\*Impl\).A"
	v2.(M).M()     // ERROR "devirtualizing v2.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	v2.(*Impl).A() // ERROR "inlining call to \(\*Impl\).A"
	v2.(*Impl).M() // ERROR "inlining call to \(\*Impl\).M"

	a.(M).(A).A() // ERROR "devirtualizing a.\(M\).\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	a.(A).(M).M() // ERROR "devirtualizing a.\(A\).\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"

	a.(M).(A).(*Impl).A() // ERROR "inlining call to \(\*Impl\).A"
	a.(A).(M).(*Impl).M() // ERROR "inlining call to \(\*Impl\).M"

	any(a).(M).M()           // ERROR "devirtualizing any\(a\).\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	any(a).(A).A()           // ERROR "devirtualizing any\(a\).\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	any(a).(M).(any).(A).A() // ERROR "devirtualizing any\(a\).\(M\).\(any\).\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"

	c := any(a)
	c.(A).A() // ERROR "devirtualizing c.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	c.(M).M() // ERROR "devirtualizing c.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"

	M(a).M()    // ERROR "devirtualizing M\(a\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	M(M(a)).M() // ERROR "devirtualizing M\(M\(a\)\).M to \*Impl$" "inlining call to \(\*Impl\).M"

	a2 := a.(A)
	A(a2).A()    // ERROR "devirtualizing A\(a2\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	A(A(a2)).A() // ERROR "devirtualizing A\(A\(a2\)\).A to \*Impl$" "inlining call to \(\*Impl\).A"

	{
		var a C = &CImpl{}   // ERROR "&CImpl{} does not escape$"
		a.(any).(C).C()      // ERROR "devirtualizing a.\(any\).\(C\).C to \*CImpl$" "inlining call to CImpl.C"
		a.(any).(*CImpl).C() // ERROR "inlining call to CImpl.C"
	}
}

func typeAssertsWithOkReturn() {
	{
		var a M = &Impl{} // ERROR "&Impl{} does not escape$"
		if v, ok := a.(M); ok {
			v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
		}
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} does not escape$"
		if v, ok := a.(A); ok {
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} does not escape$"
		v, ok := a.(M)
		if ok {
			v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
		}
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} does not escape$"
		v, ok := a.(A)
		if ok {
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} does not escape$"
		v, ok := a.(*Impl)
		if ok {
			v.A() // ERROR "inlining call to \(\*Impl\).A"
			v.M() // ERROR "inlining call to \(\*Impl\).M"
		}
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} does not escape$"
		v, _ := a.(M)
		v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} does not escape$"
		v, _ := a.(A)
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} does not escape$"
		v, _ := a.(*Impl)
		v.A() // ERROR "inlining call to \(\*Impl\).A"
		v.M() // ERROR "inlining call to \(\*Impl\).M"
	}
	{
		a := newM() // ERROR "&Impl{} does not escape$" "inlining call to newM"
		callA(a)    // ERROR "devirtualizing m.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A" "inlining call to callA"
		callIfA(a)  // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A" "inlining call to callIfA"
	}
	{
		_, a := newM2ret() // ERROR "&Impl{} does not escape$" "inlining call to newM2ret"
		callA(a)           // ERROR "devirtualizing m.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A" "inlining call to callA"
		callIfA(a)         // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A" "inlining call to callIfA"
	}
	{
		var a M = &Impl{} // ERROR "&Impl{} does not escape$"
		// Note the !ok condition, devirtualizing here is fine.
		if v, ok := a.(M); !ok {
			v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
		}
	}
	{
		var a A = newImplNoInline()
		if v, ok := a.(M); ok {
			v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
		}
	}
	{
		var impl2InA A = &Impl2{} // ERROR "&Impl2{} does not escape$"
		var a A
		a, _ = impl2InA.(*Impl)
		// a now contains the zero value of *Impl
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		a := newANoInline()
		a.A()
	}
	{
		_, a := newANoInlineRet2()
		a.A()
	}
}

func newM() M { // ERROR "can inline newM$"
	return &Impl{} // ERROR "&Impl{} escapes to heap$"
}

func newM2ret() (int, M) { // ERROR "can inline newM2ret$"
	return -1, &Impl{} // ERROR "&Impl{} escapes to heap$"
}

func callA(m M) { // ERROR "can inline callA$" "leaking param: m$"
	m.(A).A()
}

func callIfA(m M) { // ERROR "can inline callIfA$" "leaking param: m$"
	if v, ok := m.(A); ok {
		v.A()
	}
}

//go:noinline
func newImplNoInline() *Impl {
	return &Impl{} // ERROR "&Impl{} escapes to heap$"
}

//go:noinline
func newImpl2ret2() (string, *Impl2) {
	return "str", &Impl2{} // ERROR "&Impl2{} escapes to heap$"
}

//go:noinline
func newImpl2() *Impl2 {
	return &Impl2{} // ERROR "&Impl2{} escapes to heap$"
}

//go:noinline
func newANoInline() A {
	return &Impl{} // ERROR "&Impl{} escapes to heap$"
}

//go:noinline
func newANoInlineRet2() (string, A) {
	return "", &Impl{} // ERROR "&Impl{} escapes to heap$"
}

func testTypeSwitch() {
	{
		var v A = &Impl{} // ERROR "&Impl{} does not escape$"
		switch v := v.(type) {
		case A:
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		case M:
			v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
		}
	}
	{
		var v A = &Impl{} // ERROR "&Impl{} does not escape$"
		switch v := v.(type) {
		case A:
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		case M:
			v.M()       // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
			v = &Impl{} // ERROR "&Impl{} does not escape$"
			v.M()       // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
		}
		v.(M).M() // ERROR "devirtualizing v.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	}
	{
		var v A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		switch v1 := v.(type) {
		case A:
			v1.A()
		case M:
			v1.M()
			v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		}
	}
	{
		var v A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		switch v := v.(type) {
		case A:
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		case M:
			v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
		case C:
			v.C()
		}
	}
	{
		var v A = &Impl{} // ERROR "&Impl{} does not escape$"
		switch v := v.(type) {
		case M:
			v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
		default:
			panic("does not implement M") // ERROR ".does not implement M. escapes to heap$"
		}
	}
}

func differentTypeAssign() {
	{
		var a A
		a = &Impl{}  // ERROR "&Impl{} escapes to heap$"
		a = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		a.A()
	}
	{
		a := A(&Impl{}) // ERROR "&Impl{} escapes to heap$"
		a = &Impl2{}    // ERROR "&Impl2{} escapes to heap$"
		a.A()
	}
	{
		a := A(&Impl{}) // ERROR "&Impl{} escapes to heap$"
		a.A()
		a = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
	}
	{
		a := A(&Impl{}) // ERROR "&Impl{} escapes to heap$"
		a = &Impl2{}    // ERROR "&Impl2{} escapes to heap$"
		var asAny any = a
		asAny.(A).A()
	}
	{
		a := A(&Impl{}) // ERROR "&Impl{} escapes to heap$"
		var asAny any = a
		asAny = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		asAny.(A).A()
	}
	{
		a := A(&Impl{}) // ERROR "&Impl{} escapes to heap$"
		var asAny any = a
		asAny.(A).A()
		asAny = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		a.A()            // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a A
		a = &Impl{} // ERROR "&Impl{} escapes to heap$"
		a = newImpl2()
		a.A()
	}
	{
		var a A
		a = &Impl{} // ERROR "&Impl{} escapes to heap$"
		_, a = newImpl2ret2()
		a.A()
	}
}

func assignWithTypeAssert() {
	{
		var i1 A = &Impl{}  // ERROR "&Impl{} does not escape$"
		var i2 A = &Impl2{} // ERROR "&Impl2{} does not escape$"
		i1 = i2.(*Impl)     // this will panic
		i1.A()              // ERROR "devirtualizing i1.A to \*Impl$" "inlining call to \(\*Impl\).A"
		i2.A()              // ERROR "devirtualizing i2.A to \*Impl2$" "inlining call to \(\*Impl2\).A"
	}
	{
		var i1 A = &Impl{}  // ERROR "&Impl{} does not escape$"
		var i2 A = &Impl2{} // ERROR "&Impl2{} does not escape$"
		i1, _ = i2.(*Impl)  // i1 is going to be nil
		i1.A()              // ERROR "devirtualizing i1.A to \*Impl$" "inlining call to \(\*Impl\).A"
		i2.A()              // ERROR "devirtualizing i2.A to \*Impl2$" "inlining call to \(\*Impl2\).A"
	}
}

func nilIface() {
	{
		var v A = &Impl{} // ERROR "&Impl{} does not escape$"
		v = nil
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A = &Impl{} // ERROR "&Impl{} does not escape$"
		v.A()             // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		v = nil
	}
	{
		var nilIface A
		var v A = &Impl{} // ERROR "&Impl{} does not escape$"
		v.A()             // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		v = nilIface
	}
	{
		var nilIface A
		var v A = &Impl{} // ERROR "&Impl{} does not escape$"
		v = nilIface
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A
		v.A()       // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		v = &Impl{} // ERROR "&Impl{} does not escape$"
	}
	{
		var v A
		var v2 A = v
		v2.A()       // ERROR "devirtualizing v2.A to \*Impl$" "inlining call to \(\*Impl\).A"
		v2 = &Impl{} // ERROR "&Impl{} does not escape$"
	}
	{
		var v A
		v.A()
	}
	{
		var v A
		var v2 A = v
		v2.A()
	}
	{
		var v A
		var v2 A
		v2 = v
		v2.A()
	}
}

func longDevirtTest() {
	var a interface {
		M
		A
	} = &Impl{} // ERROR "&Impl{} does not escape$"

	{
		var b A = a
		b.A()     // ERROR "devirtualizing b.A to \*Impl$" "inlining call to \(\*Impl\).A"
		b.(M).M() // ERROR "devirtualizing b.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	}
	{
		var b M = a
		b.M()     // ERROR "devirtualizing b.M to \*Impl$" "inlining call to \(\*Impl\).M"
		b.(A).A() // ERROR "devirtualizing b.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var b A = a.(M).(A)
		b.A()     // ERROR "devirtualizing b.A to \*Impl$" "inlining call to \(\*Impl\).A"
		b.(M).M() // ERROR "devirtualizing b.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	}
	{
		var b M = a.(A).(M)
		b.M()     // ERROR "devirtualizing b.M to \*Impl$" "inlining call to \(\*Impl\).M"
		b.(A).A() // ERROR "devirtualizing b.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	}

	if v, ok := a.(A); ok {
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}

	if v, ok := a.(M); ok {
		v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
	}

	{
		var c A = a

		if v, ok := c.(A); ok {
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}

		c = &Impl{} // ERROR "&Impl{} does not escape$"

		if v, ok := c.(M); ok {
			v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
		}

		if v, ok := c.(interface {
			A
			M
		}); ok {
			v.M() // ERROR "devirtualizing v.M to \*Impl$" "inlining call to \(\*Impl\).M"
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}
	}
}

func deferDevirt() {
	var a A
	defer func() { // ERROR "can inline deferDevirt.func1$" "func literal does not escape$"
		a = &Impl{} // ERROR "&Impl{} escapes to heap$"
	}()
	a = &Impl{} // ERROR "&Impl{} does not escape$"
	a.A()       // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
}

func deferNoDevirt() {
	var a A
	defer func() { // ERROR "can inline deferNoDevirt.func1$" "func literal does not escape$"
		a = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
	}()
	a = &Impl{} // ERROR "&Impl{} escapes to heap$"
	a.A()
}

//go:noinline
func closureDevirt() {
	var a A
	func() { // ERROR "func literal does not escape$"
		// defer so that it does not lnline.
		defer func() {}() // ERROR "can inline closureDevirt.func1.1$" "func literal does not escape$"
		a = &Impl{}       // ERROR "&Impl{} escapes to heap$"
	}()
	a = &Impl{} // ERROR "&Impl{} does not escape$"
	a.A()       // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
}

//go:noinline
func closureNoDevirt() {
	var a A
	func() { // ERROR "func literal does not escape$"
		// defer so that it does not lnline.
		defer func() {}() // ERROR "can inline closureNoDevirt.func1.1$" "func literal does not escape$"
		a = &Impl2{}      // ERROR "&Impl2{} escapes to heap$"
	}()
	a = &Impl{} // ERROR "&Impl{} escapes to heap$"
	a.A()
}

var global = "1"

func closureDevirt2() {
	var a A
	a = &Impl{}   // ERROR "&Impl{} does not escape$"
	c := func() { // ERROR "can inline closureDevirt2.func1$" "func literal does not escape$"
		a = &Impl{} // ERROR "&Impl{} escapes to heap$"
	}
	if global == "1" {
		c = func() { // ERROR "can inline closureDevirt2.func2$" "func literal does not escape$"
			a = &Impl{} // ERROR "&Impl{} escapes to heap$"
		}
	}
	a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	c()
}

func closureNoDevirt2() {
	var a A
	a = &Impl{}   // ERROR "&Impl{} escapes to heap$"
	c := func() { // ERROR "can inline closureNoDevirt2.func1$" "func literal does not escape$"
		a = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
	}
	if global == "1" {
		c = func() { // ERROR "can inline closureNoDevirt2.func2$" "func literal does not escape$"
			a = &Impl{} // ERROR "&Impl{} escapes to heap$"
		}
	}
	a.A()
	c()
}

//go:noinline
func closureDevirt3() {
	var a A = &Impl{} // ERROR "&Impl{} does not escape$"
	func() {          // ERROR "func literal does not escape$"
		// defer so that it does not lnline.
		defer func() {}() // ERROR "can inline closureDevirt3.func1.1$" "func literal does not escape$"
		a.A()             // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}()
	func() { // ERROR "can inline closureDevirt3.func2$"
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}() // ERROR "inlining call to closureDevirt3.func2" "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
}

//go:noinline
func closureNoDevirt3() {
	var a A = &Impl{} // ERROR "&Impl{} escapes to heap$"
	func() {          // ERROR "func literal does not escape$"
		// defer so that it does not lnline.
		defer func() {}() // ERROR "can inline closureNoDevirt3.func1.1$" "func literal does not escape$"
		a.A()
	}()
	func() { // ERROR "can inline closureNoDevirt3.func2$"
		a.A()
	}() // ERROR "inlining call to closureNoDevirt3.func2"
	a = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
}

//go:noinline
func varDeclaredInClosureReferencesOuter() {
	var a A = &Impl{} // ERROR "&Impl{} does not escape$"
	func() {          // ERROR "func literal does not escape$"
		// defer for noinline
		defer func() {}() // ERROR "can inline varDeclaredInClosureReferencesOuter.func1.1$" "func literal does not escape$"
		var v A = a
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}()
	func() { // ERROR "func literal does not escape$"
		// defer for noinline
		defer func() {}() // ERROR "can inline varDeclaredInClosureReferencesOuter.func2.1$" "func literal does not escape$"
		var v A = a
		v = &Impl{} // ERROR "&Impl{} does not escape$"
		v.A()       // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}()

	var b A = &Impl{} // ERROR "&Impl{} escapes to heap$"
	func() {          // ERROR "func literal does not escape$"
		// defer for noinline
		defer func() {}() // ERROR "can inline varDeclaredInClosureReferencesOuter.func3.1$" "func literal does not escape$"
		var v A = b
		v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		v.A()
	}()
	func() { // ERROR "func literal does not escape$"
		// defer for noinline
		defer func() {}() // ERROR "can inline varDeclaredInClosureReferencesOuter.func4.1$" "func literal does not escape$"
		var v A = b
		v.A()
		v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
	}()
}

//go:noinline
func testNamedReturn0() (v A) {
	v = &Impl{} // ERROR "&Impl{} escapes to heap$"
	v.A()
	return
}

//go:noinline
func testNamedReturn1() (v A) {
	v = &Impl{} // ERROR "&Impl{} escapes to heap$"
	v.A()
	return &Impl{} // ERROR "&Impl{} escapes to heap$"
}

func testNamedReturns3() (v A) {
	v = &Impl{}    // ERROR "&Impl{} escapes to heap$"
	defer func() { // ERROR "can inline testNamedReturns3.func1$" "func literal does not escape$"
		v.A()
	}()
	v.A()
	return &Impl2{} // ERROR "&Impl2{} escapes to heap$"
}

var (
	globalImpl    = &Impl{}
	globalImpl2   = &Impl2{}
	globalA     A = &Impl{}
	globalM     M = &Impl{}
)

func globals() {
	{
		globalA.A()
		globalA.(M).M()
		globalM.M()
		globalM.(A).A()

		a := globalA
		a.A()
		a.(M).M()

		m := globalM
		m.M()
		m.(A).A()
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape$"
		a = globalImpl
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape$"
		a = A(globalImpl)
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape$"
		a = M(globalImpl).(A)
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape$"
		a = globalA.(*Impl)
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
		a = globalM.(*Impl)
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		a = globalImpl2
		a.A()
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		a = globalA
		a.A()
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		a = globalM.(A)
		a.A()
	}
}

func mapsDevirt() {
	{
		m := make(map[int]*Impl) // ERROR "make\(map\[int\]\*Impl\) does not escape$"
		var v A = m[0]
		v.A()     // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		v.(M).M() // ERROR "devirtualizing v.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	}
	{
		m := make(map[int]*Impl) // ERROR "make\(map\[int\]\*Impl\) does not escape$"
		var v A
		var ok bool
		if v, ok = m[0]; ok {
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		m := make(map[int]*Impl) // ERROR "make\(map\[int\]\*Impl\) does not escape$"
		var v A
		v, _ = m[0]
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
}

func mapsNoDevirt() {
	{
		m := make(map[int]*Impl) // ERROR "make\(map\[int\]\*Impl\) does not escape$"
		var v A = m[0]
		v.A()
		v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		v.(M).M()
	}
	{
		m := make(map[int]*Impl) // ERROR "make\(map\[int\]\*Impl\) does not escape$"
		var v A
		var ok bool
		if v, ok = m[0]; ok {
			v.A()
		}
		v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		v.A()
	}
	{
		m := make(map[int]*Impl) // ERROR "make\(map\[int\]\*Impl\) does not escape$"
		var v A = &Impl{}        // ERROR "&Impl{} escapes to heap$"
		v, _ = m[0]
		v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		v.A()
	}

	{
		m := make(map[int]A) // ERROR "make\(map\[int\]A\) does not escape$"
		var v A = &Impl{}    // ERROR "&Impl{} escapes to heap$"
		v = m[0]
		v.A()
	}
	{
		m := make(map[int]A) // ERROR "make\(map\[int\]A\) does not escape$"
		var v A = &Impl{}    // ERROR "&Impl{} escapes to heap$"
		var ok bool
		if v, ok = m[0]; ok {
			v.A()
		}
		v.A()
	}
	{
		m := make(map[int]A) // ERROR "make\(map\[int\]A\) does not escape$"
		var v A = &Impl{}    // ERROR "&Impl{} escapes to heap$"
		v, _ = m[0]
		v.A()
	}
}

func chanDevirt() {
	{
		m := make(chan *Impl)
		var v A = <-m
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		m := make(chan *Impl)
		var v A
		v = <-m
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		m := make(chan *Impl)
		var v A
		v, _ = <-m
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		m := make(chan *Impl)
		var v A
		var ok bool
		if v, ok = <-m; ok {
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		m := make(chan *Impl)
		var v A
		var ok bool
		if v, ok = <-m; ok {
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}
		select {
		case <-m:
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		case v = <-m:
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		case v, ok = <-m:
			v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}
	}
}

func chanNoDevirt() {
	{
		m := make(chan *Impl)
		var v A = <-m
		v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A
		v = <-m
		v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A
		v, _ = <-m
		v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A
		var ok bool
		if v, ok = <-m; ok {
			v.A()
		}
		v = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		var ok bool
		if v, ok = <-m; ok {
			v.A()
		}
	}
	{
		m := make(chan *Impl)
		var v A = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		select {
		case v = <-m:
			v.A()
		}
		v.A()
	}
	{
		m := make(chan *Impl)
		var v A = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		select {
		case v, _ = <-m:
			v.A()
		}
		v.A()
	}

	{
		m := make(chan A)
		var v A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		v = <-m
		v.A()
	}
	{
		m := make(chan A)
		var v A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		v, _ = <-m
		v.A()
	}
	{
		m := make(chan A)
		var v A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		var ok bool
		if v, ok = <-m; ok {
			v.A()
		}
	}
	{
		m := make(chan A)
		var v A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		select {
		case v = <-m:
			v.A()
		}
		v.A()
	}
	{
		m := make(chan A)
		var v A = &Impl{} // ERROR "&Impl{} escapes to heap$"
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
		m := make(map[*Impl]struct{}) // ERROR "make\(map\[\*Impl\]struct {}\) does not escape$"
		v = &Impl{}                   // ERROR "&Impl{} does not escape$"
		for v = range m {
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A
		m := make(map[*Impl]*Impl) // ERROR "make\(map\[\*Impl\]\*Impl\) does not escape$"
		v = &Impl{}                // ERROR "&Impl{} does not escape$"
		for v = range m {
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A
		m := make(map[*Impl]*Impl) // ERROR "make\(map\[\*Impl\]\*Impl\) does not escape$"
		v = &Impl{}                // ERROR "&Impl{} does not escape$"
		for _, v = range m {
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A
		m := make(chan *Impl)
		v = &Impl{} // ERROR "&Impl{} does not escape$"
		for v = range m {
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A
		m := []*Impl{} // ERROR "\[\]\*Impl{} does not escape$"
		v = &Impl{}    // ERROR "&Impl{} does not escape$"
		for _, v = range m {
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A
		v = &Impl{}     // ERROR "&Impl{} does not escape$"
		impl := &Impl{} // ERROR "&Impl{} does not escape$"
		i := 0
		for v = impl; i < 10; i++ {
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A
		v = &Impl{}     // ERROR "&Impl{} does not escape$"
		impl := &Impl{} // ERROR "&Impl{} does not escape$"
		i := 0
		for v = impl; i < 10; i++ {
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A
		m := [1]*Impl{&Impl{}} // ERROR "&Impl{} does not escape$"
		v = &Impl{}            // ERROR "&Impl{} does not escape$"
		for _, v = range m {
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var v A
		m := [1]*Impl{&Impl{}} // ERROR "&Impl{} does not escape$"
		v = &Impl{}            // ERROR "&Impl{} does not escape$"
		for _, v = range &m {
		}
		v.A() // ERROR "devirtualizing v.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
}

func rangeNoDevirt() {
	{
		var v A = &Impl2{}            // ERROR "&Impl2{} escapes to heap$"
		m := make(map[*Impl]struct{}) // ERROR "make\(map\[\*Impl\]struct {}\) does not escape$"
		for v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl2{}         // ERROR "&Impl2{} escapes to heap$"
		m := make(map[*Impl]*Impl) // ERROR "make\(map\[\*Impl\]\*Impl\) does not escape$"
		for v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl2{}         // ERROR "&Impl2{} escapes to heap$"
		m := make(map[*Impl]*Impl) // ERROR "make\(map\[\*Impl\]\*Impl\) does not escape$"
		for _, v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		m := make(chan *Impl)
		for v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		m := []*Impl{}     // ERROR "\[\]\*Impl{} does not escape$"
		for _, v = range m {
		}
		v.A()
	}
	{
		var v A
		v = &Impl2{}    // ERROR "&Impl2{} escapes to heap$"
		impl := &Impl{} // ERROR "&Impl{} escapes to heap$"
		i := 0
		for v = impl; i < 10; i++ {
		}
		v.A()
	}
	{
		var v A
		v = &Impl2{}    // ERROR "&Impl2{} escapes to heap$"
		impl := &Impl{} // ERROR "&Impl{} escapes to heap$"
		i := 0
		for v = impl; i < 10; i++ {
		}
		v.A()
	}
	{
		var v A
		m := [1]*Impl{&Impl{}} // ERROR "&Impl{} escapes to heap$"
		v = &Impl2{}           // ERROR "&Impl2{} escapes to heap$"
		for _, v = range m {
		}
		v.A()
	}
	{
		var v A
		m := [1]*Impl{&Impl{}} // ERROR "&Impl{} escapes to heap$"
		v = &Impl2{}           // ERROR "&Impl2{} escapes to heap$"
		for _, v = range &m {
		}
		v.A()
	}

	{
		var v A = &Impl{}         // ERROR "&Impl{} escapes to heap$"
		m := make(map[A]struct{}) // ERROR "make\(map\[A\]struct {}\) does not escape$"
		for v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl{}  // ERROR "&Impl{} escapes to heap$"
		m := make(map[A]A) // ERROR "make\(map\[A\]A\) does not escape$"
		for v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl{}  // ERROR "&Impl{} escapes to heap$"
		m := make(map[A]A) // ERROR "make\(map\[A\]A\) does not escape$"
		for _, v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		m := make(chan A)
		for v = range m {
		}
		v.A()
	}
	{
		var v A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		m := []A{}        // ERROR "\[\]A{} does not escape$"
		for _, v = range m {
		}
		v.A()
	}

	{
		var v A
		m := [1]A{&Impl{}} // ERROR "&Impl{} escapes to heap$"
		v = &Impl{}        // ERROR "&Impl{} escapes to heap$"
		for _, v = range m {
		}
		v.A()
	}
	{
		var v A
		m := [1]A{&Impl{}} // ERROR "&Impl{} escapes to heap$"
		v = &Impl{}        // ERROR "&Impl{} escapes to heap$"
		for _, v = range &m {
		}
		v.A()
	}
}

var globalInt = 1

func testIfInit() {
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape$"
		var i = &Impl{}   // ERROR "&Impl{} does not escape$"
		if a = i; globalInt == 1 {
			a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}
		a.A()     // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
		a.(M).M() // ERROR "devirtualizing a.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		var i2 = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		if a = i2; globalInt == 1 {
			a.A()
		}
		a.A()
	}
}

func testSwitchInit() {
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape$"
		var i = &Impl{}   // ERROR "&Impl{} does not escape$"
		switch a = i; globalInt {
		case 12:
			a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
		}
		a.A()     // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
		a.(M).M() // ERROR "devirtualizing a.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		var i2 = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		switch a = i2; globalInt {
		case 12:
			a.A()
		}
		a.A()
	}
}

type implWrapper Impl

func (implWrapper) A() {} // ERROR "can inline implWrapper.A$"

//go:noinline
func devirtWrapperType() {
	{
		i := &Impl{} // ERROR "&Impl{} does not escape$"
		// This is an OCONVNOP, so we have to be careful, not to devirtualize it to Impl.A.
		var a A = (*implWrapper)(i)
		a.A() // ERROR "devirtualizing a.A to \*implWrapper$" "inlining call to implWrapper.A"
	}
	{
		i := Impl{}
		// This is an OCONVNOP, so we have to be careful, not to devirtualize it to Impl.A.
		var a A = (implWrapper)(i) // ERROR "implWrapper\(i\) does not escape$"
		a.A()                      // ERROR "devirtualizing a.A to implWrapper$" "inlining call to implWrapper.A"
	}
	{
		type anyWrapper any
		var foo any = &Impl{} // ERROR "&Impl\{\} does not escape"
		var bar anyWrapper = foo
		bar.(M).M() // ERROR "devirtualizing bar\.\(M\).M to \*Impl" "inlining call to \(\*Impl\)\.M"
	}
}

func selfAssigns() {
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape$"
		a = a
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape"
		var asAny any = a
		asAny = asAny
		asAny.(A).A() // ERROR "devirtualizing asAny.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape"
		var asAny any = a
		a = asAny.(A)
		asAny.(A).A() // ERROR "devirtualizing asAny.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
		a.(A).A()     // ERROR "devirtualizing a.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
		b := a
		b.(A).A() // ERROR "devirtualizing b.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape"
		var asAny any = a
		asAny = asAny
		a = asAny.(A)
		asAny = a
		asAny.(A).A() // ERROR "devirtualizing asAny.\(A\).A to \*Impl$" "inlining call to \(\*Impl\).A"
		asAny.(M).M() // ERROR "devirtualizing asAny.\(M\).M to \*Impl$" "inlining call to \(\*Impl\).M"
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} does not escape"
		var asAny A = a
		a = asAny.(A)
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
	{
		var a, b, c A
		c = &Impl{} // ERROR "&Impl{} does not escape$"
		a = c
		c = b
		b = c
		a = b
		b = a
		c = a
		a.A() // ERROR "devirtualizing a.A to \*Impl$" "inlining call to \(\*Impl\).A"
	}
}

func boolNoDevirt() {
	{
		m := make(map[int]*Impl) // ERROR "make\(map\[int\]\*Impl\) does not escape$"
		var v any = &Impl{}      // ERROR "&Impl{} escapes to heap$"
		_, v = m[0]              // ERROR ".autotmp_[0-9]+ escapes to heap$"
		v.(A).A()
	}
	{
		m := make(chan *Impl)
		var v any = &Impl{} // ERROR "&Impl{} escapes to heap$"
		select {
		case _, v = <-m: // ERROR ".autotmp_[0-9]+ escapes to heap$"
		}
		v.(A).A()
	}
	{
		m := make(chan *Impl)
		var v any = &Impl{} // ERROR "&Impl{} escapes to heap$"
		_, v = <-m          // ERROR ".autotmp_[0-9]+ escapes to heap$"
		v.(A).A()
	}
	{
		var a any = 4       // ERROR "4 does not escape$"
		var v any = &Impl{} // ERROR "&Impl{} escapes to heap$"
		_, v = a.(int)      // ERROR ".autotmp_[0-9]+ escapes to heap$"
		v.(A).A()
	}
}

func addrTaken() {
	{
		var a A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		var ptrA = &a
		a.A()
		_ = ptrA
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		var ptrA = &a
		*ptrA = &Impl{} // ERROR "&Impl{} escapes to heap$"
		a.A()
	}
	{
		var a A = &Impl{} // ERROR "&Impl{} escapes to heap$"
		var ptrA = &a
		*ptrA = &Impl2{} // ERROR "&Impl2{} escapes to heap$"
		a.A()
	}
}

func testInvalidAsserts() {
	any(0).(interface{ A() }).A() // ERROR "any\(0\) escapes to heap$"
	{
		var a M = &Impl{} // ERROR "&Impl{} escapes to heap$"
		a.(C).C()         // this will panic
		a.(any).(C).C()   // this will panic
	}
	{
		var a C = &CImpl{} // ERROR "&CImpl{} escapes to heap$"
		a.(M).M()          // this will panic
		a.(any).(M).M()    // this will panic
	}
	{
		var a C = &CImpl{} // ERROR "&CImpl{} does not escape$"

		// this will panic
		a.(M).(*Impl).M() // ERROR "inlining call to \(\*Impl\).M"

		// this will panic
		a.(any).(M).(*Impl).M() // ERROR "inlining call to \(\*Impl\).M"
	}
}

type namedBool bool

func (namedBool) M() {} // ERROR "can inline namedBool.M$"

//go:noinline
func namedBoolTest() {
	m := map[int]int{} // ERROR "map\[int\]int{} does not escape"
	var ok namedBool
	_, ok = m[5]
	var i M = ok // ERROR "ok does not escape"
	i.M()        // ERROR "devirtualizing i.M to namedBool$" "inlining call to namedBool.M"
}
