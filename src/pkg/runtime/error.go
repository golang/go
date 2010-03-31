// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// The Error interface identifies a run time error.
type Error interface {
	String() string
	RuntimeError() // no-op that uniquely identifies runtime.Error
}

// A TypeAssertionError explains a failed type assertion.
type TypeAssertionError struct {
	interfaceType   Type // interface had this type
	concreteType    Type // concrete value had this type
	assertedType    Type // asserted type
	interfaceString string
	concreteString  string
	assertedString  string
	missingMethod   string // one method needed by Interface, missing from Concrete
}

func (e *TypeAssertionError) String() string {
	inter := e.interfaceString
	if inter == "" {
		inter = "interface"
	}
	if e.concreteType == nil {
		return "interface conversion: " + inter + " is nil, not " + e.assertedString
	}
	if e.missingMethod == "" {
		return "interface conversion: " + inter + " is " + e.concreteString +
			", not " + e.assertedString
	}
	return "interface conversion: " + e.concreteString + " is not " + e.assertedString +
		": missing method " + e.missingMethod
}

// Concrete returns the type of the concrete value in the failed type assertion.
// If the interface value was nil, Concrete returns nil.
func (e *TypeAssertionError) Concrete() Type {
	return e.concreteType
}

// Asserted returns the type incorrectly asserted by the type assertion.
func (e *TypeAssertionError) Asserted() Type {
	return e.assertedType
}

// If the type assertion is to an interface type, MissingMethod returns the
// name of a method needed to satisfy that interface type but not implemented
// by Concrete.  If there are multiple such methods,
// MissingMethod returns one; which one is unspecified.
// If the type assertion is not to an interface type, MissingMethod returns an empty string.
func (e *TypeAssertionError) MissingMethod() string {
	return e.missingMethod
}

func (*TypeAssertionError) RuntimeError() {}

// For calling from C.
func newTypeAssertionError(pt1, pt2, pt3 *Type, ps1, ps2, ps3 *string, pmeth *string, ret *interface{}) {
	var t1, t2, t3 Type
	var s1, s2, s3, meth string

	if pt1 != nil {
		t1 = *pt1
	}
	if pt2 != nil {
		t2 = *pt2
	}
	if pt3 != nil {
		t3 = *pt3
	}
	if ps1 != nil {
		s1 = *ps1
	}
	if ps2 != nil {
		s2 = *ps2
	}
	if ps3 != nil {
		s3 = *ps3
	}
	if pmeth != nil {
		meth = *pmeth
	}
	*ret = &TypeAssertionError{t1, t2, t3, s1, s2, s3, meth}
}

type stringer interface {
	String() string
}

// For calling from C.
// Prints an argument to panic.
// There's room for arbitrary complexity here, but we keep it
// simple and handle just a few important cases: int, string, and Stringer.
func printany(i interface{}) {
	switch v := i.(type) {
	case nil:
		print("nil")
	case stringer:
		print(v.String())
	case int:
		print(v)
	case string:
		print(v)
	default:
		print(i)
	}
}
