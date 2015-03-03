// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the printf checker.

package testdata

import (
	"fmt"
	"math"
	"os"
	"unsafe" // just for test case printing unsafe.Pointer
)

func UnsafePointerPrintfTest() {
	var up unsafe.Pointer
	fmt.Printf("%p, %x %X", up, up, up)
}

// Error methods that do not satisfy the Error interface and should be checked.
type errorTest1 int

func (errorTest1) Error(...interface{}) string {
	return "hi"
}

type errorTest2 int // Analogous to testing's *T type.
func (errorTest2) Error(...interface{}) {
}

type errorTest3 int

func (errorTest3) Error() { // No return value.
}

type errorTest4 int

func (errorTest4) Error() int { // Different return type.
	return 3
}

type errorTest5 int

func (errorTest5) error() { // niladic; don't complain if no args (was bug)
}

// This function never executes, but it serves as a simple test for the program.
// Test with make test.
func PrintfTests() {
	var b bool
	var i int
	var r rune
	var s string
	var x float64
	var p *int
	var imap map[int]int
	var fslice []float64
	var c complex64
	// Some good format/argtypes
	fmt.Printf("")
	fmt.Printf("%b %b %b", 3, i, x)
	fmt.Printf("%c %c %c %c", 3, i, 'x', r)
	fmt.Printf("%d %d %d", 3, i, imap)
	fmt.Printf("%e %e %e %e", 3e9, x, fslice, c)
	fmt.Printf("%E %E %E %E", 3e9, x, fslice, c)
	fmt.Printf("%f %f %f %f", 3e9, x, fslice, c)
	fmt.Printf("%F %F %F %F", 3e9, x, fslice, c)
	fmt.Printf("%g %g %g %g", 3e9, x, fslice, c)
	fmt.Printf("%G %G %G %G", 3e9, x, fslice, c)
	fmt.Printf("%b %b %b %b", 3e9, x, fslice, c)
	fmt.Printf("%o %o", 3, i)
	fmt.Printf("%p %p", p, nil)
	fmt.Printf("%q %q %q %q", 3, i, 'x', r)
	fmt.Printf("%s %s %s", "hi", s, []byte{65})
	fmt.Printf("%t %t", true, b)
	fmt.Printf("%T %T", 3, i)
	fmt.Printf("%U %U", 3, i)
	fmt.Printf("%v %v", 3, i)
	fmt.Printf("%x %x %x %x", 3, i, "hi", s)
	fmt.Printf("%X %X %X %X", 3, i, "hi", s)
	fmt.Printf("%.*s %d %g", 3, "hi", 23, 2.3)
	fmt.Printf("%s", &stringerv)
	fmt.Printf("%v", &stringerv)
	fmt.Printf("%T", &stringerv)
	fmt.Printf("%v", notstringerv)
	fmt.Printf("%T", notstringerv)
	fmt.Printf("%q", stringerarrayv)
	fmt.Printf("%v", stringerarrayv)
	fmt.Printf("%s", stringerarrayv)
	fmt.Printf("%v", notstringerarrayv)
	fmt.Printf("%T", notstringerarrayv)
	fmt.Printf("%d", new(Formatter))
	fmt.Printf("%*%", 2)               // Ridiculous but allowed.
	fmt.Printf("%s", interface{}(nil)) // Nothing useful we can say.

	fmt.Printf("%g", 1+2i)
	// Some bad format/argTypes
	fmt.Printf("%b", "hi")                     // ERROR "arg .hi. for printf verb %b of wrong type"
	fmt.Printf("%t", c)                        // ERROR "arg c for printf verb %t of wrong type"
	fmt.Printf("%t", 1+2i)                     // ERROR "arg 1 \+ 2i for printf verb %t of wrong type"
	fmt.Printf("%c", 2.3)                      // ERROR "arg 2.3 for printf verb %c of wrong type"
	fmt.Printf("%d", 2.3)                      // ERROR "arg 2.3 for printf verb %d of wrong type"
	fmt.Printf("%e", "hi")                     // ERROR "arg .hi. for printf verb %e of wrong type"
	fmt.Printf("%E", true)                     // ERROR "arg true for printf verb %E of wrong type"
	fmt.Printf("%f", "hi")                     // ERROR "arg .hi. for printf verb %f of wrong type"
	fmt.Printf("%F", 'x')                      // ERROR "arg 'x' for printf verb %F of wrong type"
	fmt.Printf("%g", "hi")                     // ERROR "arg .hi. for printf verb %g of wrong type"
	fmt.Printf("%g", imap)                     // ERROR "arg imap for printf verb %g of wrong type"
	fmt.Printf("%G", i)                        // ERROR "arg i for printf verb %G of wrong type"
	fmt.Printf("%o", x)                        // ERROR "arg x for printf verb %o of wrong type"
	fmt.Printf("%p", 23)                       // ERROR "arg 23 for printf verb %p of wrong type"
	fmt.Printf("%q", x)                        // ERROR "arg x for printf verb %q of wrong type"
	fmt.Printf("%s", b)                        // ERROR "arg b for printf verb %s of wrong type"
	fmt.Printf("%s", byte(65))                 // ERROR "arg byte\(65\) for printf verb %s of wrong type"
	fmt.Printf("%t", 23)                       // ERROR "arg 23 for printf verb %t of wrong type"
	fmt.Printf("%U", x)                        // ERROR "arg x for printf verb %U of wrong type"
	fmt.Printf("%x", nil)                      // ERROR "arg nil for printf verb %x of wrong type"
	fmt.Printf("%X", 2.3)                      // ERROR "arg 2.3 for printf verb %X of wrong type"
	fmt.Printf("%s", stringerv)                // ERROR "arg stringerv for printf verb %s of wrong type"
	fmt.Printf("%t", stringerv)                // ERROR "arg stringerv for printf verb %t of wrong type"
	fmt.Printf("%q", notstringerv)             // ERROR "arg notstringerv for printf verb %q of wrong type"
	fmt.Printf("%t", notstringerv)             // ERROR "arg notstringerv for printf verb %t of wrong type"
	fmt.Printf("%t", stringerarrayv)           // ERROR "arg stringerarrayv for printf verb %t of wrong type"
	fmt.Printf("%t", notstringerarrayv)        // ERROR "arg notstringerarrayv for printf verb %t of wrong type"
	fmt.Printf("%q", notstringerarrayv)        // ERROR "arg notstringerarrayv for printf verb %q of wrong type"
	fmt.Printf("%d", Formatter(true))          // correct (the type is responsible for formatting)
	fmt.Printf("%s", nonemptyinterface)        // correct (the dynamic type of nonemptyinterface may be a stringer)
	fmt.Printf("%.*s %d %g", 3, "hi", 23, 'x') // ERROR "arg 'x' for printf verb %g of wrong type"
	fmt.Println()                              // not an error
	fmt.Println("%s", "hi")                    // ERROR "possible formatting directive in Println call"
	fmt.Printf("%s", "hi", 3)                  // ERROR "wrong number of args for format in Printf call"
	_ = fmt.Sprintf("%"+("s"), "hi", 3)        // ERROR "wrong number of args for format in Sprintf call"
	fmt.Printf("%s%%%d", "hi", 3)              // correct
	fmt.Printf("%08s", "woo")                  // correct
	fmt.Printf("% 8s", "woo")                  // correct
	fmt.Printf("%.*d", 3, 3)                   // correct
	fmt.Printf("%.*d", 3, 3, 3, 3)             // ERROR "wrong number of args for format in Printf call.*4 args"
	fmt.Printf("%.*d", "hi", 3)                // ERROR "arg .hi. for \* in printf format not of type int"
	fmt.Printf("%.*d", i, 3)                   // correct
	fmt.Printf("%.*d", s, 3)                   // ERROR "arg s for \* in printf format not of type int"
	fmt.Printf("%*%", 0.22)                    // ERROR "arg 0.22 for \* in printf format not of type int"
	fmt.Printf("%q %q", multi()...)            // ok
	fmt.Printf("%#q", `blah`)                  // ok
	printf("now is the time", "buddy")         // ERROR "no formatting directive"
	Printf("now is the time", "buddy")         // ERROR "no formatting directive"
	Printf("hi")                               // ok
	const format = "%s %s\n"
	Printf(format, "hi", "there")
	Printf(format, "hi")              // ERROR "missing argument for Printf..%s..: format reads arg 2, have only 1"
	Printf("%s %d %.3v %q", "str", 4) // ERROR "missing argument for Printf..%.3v..: format reads arg 3, have only 2"
	f := new(stringer)
	f.Warn(0, "%s", "hello", 3)  // ERROR "possible formatting directive in Warn call"
	f.Warnf(0, "%s", "hello", 3) // ERROR "wrong number of args for format in Warnf call"
	f.Warnf(0, "%r", "hello")    // ERROR "unrecognized printf verb"
	f.Warnf(0, "%#s", "hello")   // ERROR "unrecognized printf flag"
	Printf("d%", 2)              // ERROR "missing verb at end of format string in Printf call"
	Printf("%d", percentDV)
	Printf("%d", &percentDV)
	Printf("%d", notPercentDV)  // ERROR "arg notPercentDV for printf verb %d of wrong type"
	Printf("%d", &notPercentDV) // ERROR "arg &notPercentDV for printf verb %d of wrong type"
	Printf("%p", &notPercentDV) // Works regardless: we print it as a pointer.
	Printf("%s", percentSV)
	Printf("%s", &percentSV)
	// Good argument reorderings.
	Printf("%[1]d", 3)
	Printf("%[1]*d", 3, 1)
	Printf("%[2]*[1]d", 1, 3)
	Printf("%[2]*.[1]*[3]d", 2, 3, 4)
	fmt.Fprintf(os.Stderr, "%[2]*.[1]*[3]d", 2, 3, 4) // Use Fprintf to make sure we count arguments correctly.
	// Bad argument reorderings.
	Printf("%[xd", 3)                    // ERROR "illegal syntax for printf argument index"
	Printf("%[x]d", 3)                   // ERROR "illegal syntax for printf argument index"
	Printf("%[3]*s", "hi", 2)            // ERROR "missing argument for Printf.* reads arg 3, have only 2"
	_ = fmt.Sprintf("%[3]d", 2)          // ERROR "missing argument for Sprintf.* reads arg 3, have only 1"
	Printf("%[2]*.[1]*[3]d", 2, "hi", 4) // ERROR "arg .hi. for \* in printf format not of type int"
	Printf("%[0]s", "arg1")              // ERROR "index value \[0\] for Printf.*; indexes start at 1"
	Printf("%[0]d", 1)                   // ERROR "index value \[0\] for Printf.*; indexes start at 1"
	// Something that satisfies the error interface.
	var e error
	fmt.Println(e.Error()) // ok
	// Something that looks like an error interface but isn't, such as the (*T).Error method
	// in the testing package.
	var et1 errorTest1
	fmt.Println(et1.Error())        // ERROR "no args in Error call"
	fmt.Println(et1.Error("hi"))    // ok
	fmt.Println(et1.Error("%d", 3)) // ERROR "possible formatting directive in Error call"
	var et2 errorTest2
	et2.Error()        // ERROR "no args in Error call"
	et2.Error("hi")    // ok, not an error method.
	et2.Error("%d", 3) // ERROR "possible formatting directive in Error call"
	var et3 errorTest3
	et3.Error() // ok, not an error method.
	var et4 errorTest4
	et4.Error() // ok, not an error method.
	var et5 errorTest5
	et5.error() // ok, not an error method.
	// Bug: used to recur forever.
	Printf("%p %x", recursiveStructV, recursiveStructV.next)
	Printf("%p %x", recursiveStruct1V, recursiveStruct1V.next)
	Printf("%p %x", recursiveSliceV, recursiveSliceV)
	Printf("%p %x", recursiveMapV, recursiveMapV)
	// Special handling for Log.
	math.Log(3)  // OK
	Log(3)       // OK
	Log("%d", 3) // ERROR "possible formatting directive in Log call"
	Logf("%d", 3)
	Logf("%d", "hi") // ERROR "arg .hi. for printf verb %d of wrong type: untyped string"

}

// Printf is used by the test so we must declare it.
func Printf(format string, args ...interface{}) {
	panic("don't call - testing only")
}

// printf is used by the test so we must declare it.
func printf(format string, args ...interface{}) {
	panic("don't call - testing only")
}

// multi is used by the test.
func multi() []interface{} {
	panic("don't call - testing only")
}

type stringer float64

var stringerv stringer

func (*stringer) String() string {
	return "string"
}

func (*stringer) Warn(int, ...interface{}) string {
	return "warn"
}

func (*stringer) Warnf(int, string, ...interface{}) string {
	return "warnf"
}

type notstringer struct {
	f float64
}

var notstringerv notstringer

type stringerarray [4]float64

func (stringerarray) String() string {
	return "string"
}

var stringerarrayv stringerarray

type notstringerarray [4]float64

var notstringerarrayv notstringerarray

var nonemptyinterface = interface {
	f()
}(nil)

// A data type we can print with "%d".
type percentDStruct struct {
	a int
	b []byte
	c *float64
}

var percentDV percentDStruct

// A data type we cannot print correctly with "%d".
type notPercentDStruct struct {
	a int
	b []byte
	c bool
}

var notPercentDV notPercentDStruct

// A data type we can print with "%s".
type percentSStruct struct {
	a string
	b []byte
	c stringerarray
}

var percentSV percentSStruct

type recursiveStringer int

func (s recursiveStringer) String() string {
	_ = fmt.Sprintf("%d", s)
	_ = fmt.Sprintf("%#v", s)
	_ = fmt.Sprintf("%v", s)  // ERROR "arg s for printf causes recursive call to String method"
	_ = fmt.Sprintf("%v", &s) // ERROR "arg &s for printf causes recursive call to String method"
	_ = fmt.Sprintf("%T", s)  // ok; does not recursively call String
	return fmt.Sprintln(s)    // ERROR "arg s for print causes recursive call to String method"
}

type recursivePtrStringer int

func (p *recursivePtrStringer) String() string {
	_ = fmt.Sprintf("%v", *p)
	return fmt.Sprintln(p) // ERROR "arg p for print causes recursive call to String method"
}

type Formatter bool

func (*Formatter) Format(fmt.State, rune) {
}

type RecursiveSlice []RecursiveSlice

var recursiveSliceV = &RecursiveSlice{}

type RecursiveMap map[int]RecursiveMap

var recursiveMapV = make(RecursiveMap)

type RecursiveStruct struct {
	next *RecursiveStruct
}

var recursiveStructV = &RecursiveStruct{}

type RecursiveStruct1 struct {
	next *Recursive2Struct
}

type RecursiveStruct2 struct {
	next *Recursive1Struct
}

var recursiveStruct1V = &RecursiveStruct1{}

// Fix for issue 7149: Missing return type on String method caused fault.
func (int) String() {
	return ""
}
