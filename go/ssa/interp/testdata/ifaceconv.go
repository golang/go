package main

// Tests of interface conversions and type assertions.

type I0 interface {
}
type I1 interface {
	f()
}
type I2 interface {
	f()
	g()
}

type C0 struct{}
type C1 struct{}

func (C1) f() {}

type C2 struct{}

func (C2) f() {}
func (C2) g() {}

func main() {
	var i0 I0
	var i1 I1
	var i2 I2

	// Nil always causes a type assertion to fail, even to the
	// same type.
	if _, ok := i0.(I0); ok {
		panic("nil i0.(I0) succeeded")
	}
	if _, ok := i1.(I1); ok {
		panic("nil i1.(I1) succeeded")
	}
	if _, ok := i2.(I2); ok {
		panic("nil i2.(I2) succeeded")
	}

	// Conversions can't fail, even with nil.
	_ = I0(i0)

	_ = I0(i1)
	_ = I1(i1)

	_ = I0(i2)
	_ = I1(i2)
	_ = I2(i2)

	// Non-nil type assertions pass or fail based on the concrete type.
	i1 = C1{}
	if _, ok := i1.(I0); !ok {
		panic("C1 i1.(I0) failed")
	}
	if _, ok := i1.(I1); !ok {
		panic("C1 i1.(I1) failed")
	}
	if _, ok := i1.(I2); ok {
		panic("C1 i1.(I2) succeeded")
	}

	i1 = C2{}
	if _, ok := i1.(I0); !ok {
		panic("C2 i1.(I0) failed")
	}
	if _, ok := i1.(I1); !ok {
		panic("C2 i1.(I1) failed")
	}
	if _, ok := i1.(I2); !ok {
		panic("C2 i1.(I2) failed")
	}

	// Conversions can't fail.
	i1 = C1{}
	if I0(i1) == nil {
		panic("C1 I0(i1) was nil")
	}
	if I1(i1) == nil {
		panic("C1 I1(i1) was nil")
	}
}
