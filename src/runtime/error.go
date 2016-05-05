// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// The Error interface identifies a run time error.
type Error interface {
	error

	// RuntimeError is a no-op function but
	// serves to distinguish types that are run time
	// errors from ordinary errors: a type is a
	// run time error if it has a RuntimeError method.
	RuntimeError()
}

// A TypeAssertionError explains a failed type assertion.
type TypeAssertionError struct {
	interfaceString string
	concreteString  string
	assertedString  string
	missingMethod   string // one method needed by Interface, missing from Concrete
}

func (*TypeAssertionError) RuntimeError() {}

func (e *TypeAssertionError) Error() string {
	inter := e.interfaceString
	if inter == "" {
		inter = "interface"
	}
	if e.concreteString == "" {
		return "interface conversion: " + inter + " is nil, not " + e.assertedString
	}
	if e.missingMethod == "" {
		return "interface conversion: " + inter + " is " + e.concreteString +
			", not " + e.assertedString
	}
	return "interface conversion: " + e.concreteString + " is not " + e.assertedString +
		": missing method " + e.missingMethod
}

// An errorString represents a runtime error described by a single string.
type errorString string

func (e errorString) RuntimeError() {}

func (e errorString) Error() string {
	return "runtime error: " + string(e)
}

// plainError represents a runtime error described a string without
// the prefix "runtime error: " after invoking errorString.Error().
// See Issue #14965.
type plainError string

func (e plainError) RuntimeError() {}

func (e plainError) Error() string {
	return string(e)
}

type stringer interface {
	String() string
}

func typestring(x interface{}) string {
	e := efaceOf(&x)
	return e._type.string()
}

// For calling from C.
// Prints an argument passed to panic.
// There's room for arbitrary complexity here, but we keep it
// simple and handle just a few important cases: int, string, and Stringer.
func printany(i interface{}) {
	switch v := i.(type) {
	case nil:
		print("nil")
	case stringer:
		print(v.String())
	case error:
		print(v.Error())
	case int:
		print(v)
	case string:
		print(v)
	default:
		print("(", typestring(i), ") ", i)
	}
}

// called from generated code
func panicwrap(pkg, typ, meth string) {
	panic(plainError("value method " + pkg + "." + typ + "." + meth + " called using nil *" + typ + " pointer"))
}
