// runoutput

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// Check that expressions like (c*n + d*(n+k)) get correctly merged by
// the compiler into (c+d)*n + d*k (with c+d and d*k computed at
// compile time).
//
// The merging is performed by a combination of the multiplication
// merge rules
//  (c*n + d*n) -> (c+d)*n
// and the distributive multiplication rules
//  c * (d+x)  ->  c*d + c*x

// Generate a MergeTest that looks like this:
//
//   a8, b8 = m1*n8 + m2*(n8+k), (m1+m2)*n8 + m2*k
//   if a8 != b8 {
// 	   // print error msg and panic
//   }
func makeMergeAddTest(m1, m2, k int, size string) string {

	model := "    a" + size + ", b" + size
	model += fmt.Sprintf(" = %%d*n%s + %%d*(n%s+%%d), (%%d+%%d)*n%s + (%%d*%%d)", size, size, size)

	test := fmt.Sprintf(model, m1, m2, k, m1, m2, m2, k)
	test += fmt.Sprintf(`
    if a%s != b%s {
        fmt.Printf("MergeAddTest(%d, %d, %d, %s) failed\n")
        fmt.Printf("%%d != %%d\n", a%s, b%s)
        panic("FAIL")
    }
`, size, size, m1, m2, k, size, size, size)
	return test + "\n"
}

// Check that expressions like (c*n - d*(n+k)) get correctly merged by
// the compiler into (c-d)*n - d*k (with c-d and d*k computed at
// compile time).
//
// The merging is performed by a combination of the multiplication
// merge rules
//  (c*n - d*n) -> (c-d)*n
// and the distributive multiplication rules
//  c * (d-x)  ->  c*d - c*x

// Generate a MergeTest that looks like this:
//
//   a8, b8 = m1*n8 - m2*(n8+k), (m1-m2)*n8 - m2*k
//   if a8 != b8 {
// 	   // print error msg and panic
//   }
func makeMergeSubTest(m1, m2, k int, size string) string {

	model := "    a" + size + ", b" + size
	model += fmt.Sprintf(" = %%d*n%s - %%d*(n%s+%%d), (%%d-%%d)*n%s - (%%d*%%d)", size, size, size)

	test := fmt.Sprintf(model, m1, m2, k, m1, m2, m2, k)
	test += fmt.Sprintf(`
    if a%s != b%s {
        fmt.Printf("MergeSubTest(%d, %d, %d, %s) failed\n")
        fmt.Printf("%%d != %%d\n", a%s, b%s)
        panic("FAIL")
    }
`, size, size, m1, m2, k, size, size, size)
	return test + "\n"
}

func makeAllSizes(m1, m2, k int) string {
	var tests string
	tests += makeMergeAddTest(m1, m2, k, "8")
	tests += makeMergeAddTest(m1, m2, k, "16")
	tests += makeMergeAddTest(m1, m2, k, "32")
	tests += makeMergeAddTest(m1, m2, k, "64")
	tests += makeMergeSubTest(m1, m2, k, "8")
	tests += makeMergeSubTest(m1, m2, k, "16")
	tests += makeMergeSubTest(m1, m2, k, "32")
	tests += makeMergeSubTest(m1, m2, k, "64")
	tests += "\n"
	return tests
}

func main() {
	fmt.Println(`package main

import "fmt"

var n8 int8 = 42
var n16 int16 = 42
var n32 int32 = 42
var n64 int64 = 42

func main() {
    var a8, b8 int8
    var a16, b16 int16
    var a32, b32 int32
    var a64, b64 int64
`)

	fmt.Println(makeAllSizes(03, 05, 0)) // 3*n + 5*n
	fmt.Println(makeAllSizes(17, 33, 0))
	fmt.Println(makeAllSizes(80, 45, 0))
	fmt.Println(makeAllSizes(32, 64, 0))

	fmt.Println(makeAllSizes(7, 11, +1)) // 7*n + 11*(n+1)
	fmt.Println(makeAllSizes(9, 13, +2))
	fmt.Println(makeAllSizes(11, 16, -1))
	fmt.Println(makeAllSizes(17, 9, -2))

	fmt.Println("}")
}
