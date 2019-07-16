// skip

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generate test of index and slice bounds checks.
// The actual tests are index0.go, index1.go, index2.go.

package main

import (
	"bufio"
	"fmt"
	"os"
	"unsafe"
)

const prolog = `

package main

import (
	"runtime"
)

type quad struct { x, y, z, w int }

const (
	cj = 100011
	ci int = 100012
	ci8 int8 = 115
	ci16 int16 = 10016
	ci32 int32 = 100013
	ci64 int64 = 100014
	ci64big int64 = 1<<31
	ci64bigger int64 = 1<<32
	chuge = 1<<100
	cfgood = 2.0
	cfbad = 2.1

	cnj = -2
	cni int = -3
	cni8 int8 = -6
	cni16 int16 = -7
	cni32 int32 = -4
	cni64 int64 = -5
	cni64big int64 = -1<<31
	cni64bigger int64 = -1<<32
	cnhuge = -1<<100
	cnfgood = -2.0
	cnfbad = -2.1
)

var j int = 100020
var i int = 100021
var i8 int8 = 126
var i16 int16 = 10025
var i32 int32 = 100022
var i64 int64 = 100023
var i64big int64 = 1<<31
var i64bigger int64 = 1<<32
var huge uint64 = 1<<64 - 1
var fgood float64 = 2.0
var fbad float64 = 2.1

var nj int = -10
var ni int = -11
var ni8 int8 = -14
var ni16 int16 = -15
var ni32 int32 = -12
var ni64 int64 = -13
var ni64big int64 = -1<<31
var ni64bigger int64 = -1<<32
var nhuge int64 = -1<<63
var nfgood float64 = -2.0
var nfbad float64 = -2.1

var si []int = make([]int, 10)
var ai [10]int
var pai *[10]int = &ai

var sq []quad = make([]quad, 10)
var aq [10]quad
var paq *[10]quad = &aq

var sib []int = make([]int, 100000)
var aib [100000]int
var paib *[100000]int = &aib

var sqb []quad = make([]quad, 100000)
var aqb [100000]quad
var paqb *[100000]quad = &aqb

type T struct {
	si []int
	ai [10]int
	pai *[10]int
	sq []quad
	aq [10]quad
	paq *[10]quad

	sib []int
	aib [100000]int
	paib *[100000]int
	sqb []quad
	aqb [100000]quad
	paqb *[100000]quad
}

var t = T{si, ai, pai, sq, aq, paq, sib, aib, paib, sqb, aqb, paqb}

var pt = &T{si, ai, pai, sq, aq, paq, sib, aib, paib, sqb, aqb, paqb}

// test that f panics
func test(f func(), s string) {
	defer func() {
		if err := recover(); err == nil {
			_, file, line, _ := runtime.Caller(2)
			bug()
			print(file, ":", line, ": ", s, " did not panic\n")
		} else if !contains(err.(error).Error(), "out of range") {
			_, file, line, _ := runtime.Caller(2)
			bug()
			print(file, ":", line, ": ", s, " unexpected panic: ", err.(error).Error(), "\n")
		}
	}()
	f()
}

func contains(x, y string) bool {
	for i := 0; i+len(y) <= len(x); i++ {
		if x[i:i+len(y)] == y {
			return true
		}
	}
	return false
}


var X interface{}
func use(y interface{}) {
	X = y
}

var didBug = false

func bug() {
	if !didBug {
		didBug = true
		println("BUG")
	}
}

func main() {
`

// pass variable set in index[012].go
//	0 - dynamic checks
//	1 - static checks of invalid constants (cannot assign to types)
//	2 - static checks of array bounds

func testExpr(b *bufio.Writer, expr string) {
	if pass == 0 {
		fmt.Fprintf(b, "\ttest(func(){use(%s)}, %q)\n", expr, expr)
	} else {
		fmt.Fprintf(b, "\tuse(%s)  // ERROR \"index|overflow|truncated|must be integer\"\n", expr)
	}
}

func main() {
	b := bufio.NewWriter(os.Stdout)

	if pass == 0 {
		fmt.Fprint(b, "// run\n\n")
	} else {
		fmt.Fprint(b, "// errorcheck\n\n")
	}
	fmt.Fprint(b, prolog)

	var choices = [][]string{
		// Direct value, fetch from struct, fetch from struct pointer.
		// The last two cases get us to oindex_const_sudo in gsubr.c.
		[]string{"", "t.", "pt."},

		// Array, pointer to array, slice.
		[]string{"a", "pa", "s"},

		// Element is int, element is quad (struct).
		// This controls whether we end up in gsubr.c (i) or cgen.c (q).
		[]string{"i", "q"},

		// Small or big len.
		[]string{"", "b"},

		// Variable or constant.
		[]string{"", "c"},

		// Positive or negative.
		[]string{"", "n"},

		// Size of index.
		[]string{"j", "i", "i8", "i16", "i32", "i64", "i64big", "i64bigger", "huge", "fgood", "fbad"},
	}

	forall(choices, func(x []string) {
		p, a, e, big, c, n, i := x[0], x[1], x[2], x[3], x[4], x[5], x[6]

		// Pass: dynamic=0, static=1, 2.
		// Which cases should be caught statically?
		// Only constants, obviously.
		// Beyond that, must be one of these:
		//	indexing into array or pointer to array
		//	negative constant
		//	large constant
		thisPass := 0
		if c == "c" && (a == "a" || a == "pa" || n == "n" || i == "i64big" || i == "i64bigger" || i == "huge" || i == "fbad") {
			if i == "huge" {
				// Due to a detail of gc's internals,
				// the huge constant errors happen in an
				// earlier pass than the others and inhibits
				// the next pass from running.
				// So run it as a separate check.
				thisPass = 1
			} else if a == "s" && n == "" && (i == "i64big" || i == "i64bigger") && unsafe.Sizeof(int(0)) > 4 {
				// If int is 64 bits, these huge
				// numbers do fit in an int, so they
				// are not rejected at compile time.
				thisPass = 0
			} else {
				thisPass = 2
			}
		}

		pae := p + a + e + big
		cni := c + n + i

		// If we're using the big-len data, positive int8 and int16 cannot overflow.
		if big == "b" && n == "" && (i == "i8" || i == "i16") {
			if pass == 0 {
				fmt.Fprintf(b, "\tuse(%s[%s])\n", pae, cni)
				fmt.Fprintf(b, "\tuse(%s[0:%s])\n", pae, cni)
				fmt.Fprintf(b, "\tuse(%s[1:%s])\n", pae, cni)
				fmt.Fprintf(b, "\tuse(%s[%s:])\n", pae, cni)
				fmt.Fprintf(b, "\tuse(%s[%s:%s])\n", pae, cni, cni)
			}
			return
		}

		// Float variables cannot be used as indices.
		if c == "" && (i == "fgood" || i == "fbad") {
			return
		}
		// Integral float constat is ok.
		if c == "c" && n == "" && i == "fgood" {
			if pass == 0 {
				fmt.Fprintf(b, "\tuse(%s[%s])\n", pae, cni)
				fmt.Fprintf(b, "\tuse(%s[0:%s])\n", pae, cni)
				fmt.Fprintf(b, "\tuse(%s[1:%s])\n", pae, cni)
				fmt.Fprintf(b, "\tuse(%s[%s:])\n", pae, cni)
				fmt.Fprintf(b, "\tuse(%s[%s:%s])\n", pae, cni, cni)
			}
			return
		}

		// Only print the test case if it is appropriate for this pass.
		if thisPass == pass {
			// Index operation
			testExpr(b, pae+"["+cni+"]")

			// Slice operation.
			// Low index 0 is a special case in ggen.c
			// so test both 0 and 1.
			testExpr(b, pae+"[0:"+cni+"]")
			testExpr(b, pae+"[1:"+cni+"]")
			testExpr(b, pae+"["+cni+":]")
			testExpr(b, pae+"["+cni+":"+cni+"]")
		}
	})

	fmt.Fprintln(b, "}")
	b.Flush()
}

func forall(choices [][]string, f func([]string)) {
	x := make([]string, len(choices))

	var recurse func(d int)
	recurse = func(d int) {
		if d >= len(choices) {
			f(x)
			return
		}
		for _, x[d] = range choices[d] {
			recurse(d + 1)
		}
	}
	recurse(0)
}
