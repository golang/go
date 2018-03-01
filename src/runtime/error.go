// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe" // for go:linkname

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

// printany prints an argument passed to panic.
// If panic is called with a value that has a String or Error method,
// it has already been converted into a string by preprintpanics.
func printany(i interface{}) {
	switch v := i.(type) {
	case nil:
		print("nil")
	case bool:
		print(v)
	case int:
		print(v)
	case int8:
		print(v)
	case int16:
		print(v)
	case int32:
		print(v)
	case int64:
		print(v)
	case uint:
		print(v)
	case uint8:
		print(v)
	case uint16:
		print(v)
	case uint32:
		print(v)
	case uint64:
		print(v)
	case uintptr:
		print(v)
	case float32:
		print(v)
	case float64:
		print(v)
	case complex64:
		print(v)
	case complex128:
		print(v)
	case string:
		print(v)
	default:
		print("(", typestring(i), ") ", i)
	}
}

// strings.IndexByte is implemented in runtime/asm_$goarch.s
// but amusingly we need go:linkname to get access to it here in the runtime.
//go:linkname stringsIndexByte strings.IndexByte
func stringsIndexByte(s string, c byte) int

// panicwrap generates a panic for a call to a wrapped value method
// with a nil pointer receiver.
//
// It is called from the generated wrapper code.
func panicwrap() {
	pc := getcallerpc()
	name := funcname(findfunc(pc))
	// name is something like "main.(*T).F".
	// We want to extract pkg ("main"), typ ("T"), and meth ("F").
	// Do it by finding the parens.
	i := stringsIndexByte(name, '(')
	if i < 0 {
		throw("panicwrap: no ( in " + name)
	}
	pkg := name[:i-1]
	if i+2 >= len(name) || name[i-1:i+2] != ".(*" {
		throw("panicwrap: unexpected string after package name: " + name)
	}
	name = name[i+2:]
	i = stringsIndexByte(name, ')')
	if i < 0 {
		throw("panicwrap: no ) in " + name)
	}
	if i+2 >= len(name) || name[i:i+2] != ")." {
		throw("panicwrap: unexpected string after type name: " + name)
	}
	typ := name[:i]
	meth := name[i+2:]
	panic(plainError("value method " + pkg + "." + typ + "." + meth + " called using nil *" + typ + " pointer"))
}
