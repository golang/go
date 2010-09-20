// $G $D/$F.go && $L $F.$A &&
// ./$A.out -pass 0 >tmp.go && $G tmp.go && $L -o $A.out1 tmp.$A && ./$A.out1 &&
// ./$A.out -pass 1 >tmp.go && errchk $G -e tmp.go &&
// ./$A.out -pass 2 >tmp.go && errchk $G -e tmp.go
// rm -f tmp.go $A.out1

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generate test of index and slice bounds checks.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
)

const prolog = `

package main

import (
	"runtime"
)

type quad struct { x, y, z, w int }

const (
	cj = 11
	ci int = 12
	ci32 int32 = 13
	ci64 int64 = 14
	ci64big int64 = 1<<31
	ci64bigger int64 = 1<<32
	chuge = 1<<100

	cnj = -2
	cni int = -3
	cni32 int32 = -4
	cni64 int64 = -5
	cni64big int64 = -1<<31
	cni64bigger int64 = -1<<32
	cnhuge = -1<<100
)

var j int = 20
var i int = 21
var i32 int32 = 22
var i64 int64 = 23
var i64big int64 = 1<<31
var i64bigger int64 = 1<<32
var huge uint64 = 1<<64 - 1

var nj int = -10
var ni int = -11
var ni32 int32 = -12
var ni64 int64 = -13
var ni64big int64 = -1<<31
var ni64bigger int64 = -1<<32
var nhuge int64 = -1<<63

var si []int = make([]int, 10)
var ai [10]int
var pai *[10]int = &ai

var sq []quad = make([]quad, 10)
var aq [10]quad
var paq *[10]quad = &aq

type T struct {
	si []int
	ai [10]int
	pai *[10]int
	sq []quad
	aq [10]quad
	paq *[10]quad
}

var t = T{si, ai, pai, sq, aq, paq}

var pt = &T{si, ai, pai, sq, aq, paq}

// test that f panics
func test(f func(), s string) {
	defer func() {
		if err := recover(); err == nil {
			_, file, line, _ := runtime.Caller(2)
			bug()
			print(file, ":", line, ": ", s, " did not panic\n")
		}
	}()
	f()
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

// Passes:
//	0 - dynamic checks
//	1 - static checks of invalid constants (cannot assign to types)
//	2 - static checks of array bounds
var pass = flag.Int("pass", 0, "which test (0,1,2)")

func testExpr(b *bufio.Writer, expr string) {
	if *pass == 0 {
		fmt.Fprintf(b, "\ttest(func(){use(%s)}, %q)\n", expr, expr)
	} else {
		fmt.Fprintf(b, "\tuse(%s)  // ERROR \"index|overflow\"\n", expr)
	}
}

func main() {
	b := bufio.NewWriter(os.Stdout)

	flag.Parse()
	
	if *pass == 0 {
		fmt.Fprint(b, "// $G $D/$F.go && $L $F.$A && ./$A.out\n\n")
	} else {
		fmt.Fprint(b, "// errchk $G -e $D/$F.go\n\n")
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

		// Variable or constant.
		[]string{"", "c"},

		// Positive or negative.
		[]string{"", "n"},

		// Size of index.
		[]string{"j", "i", "i32", "i64", "i64big", "i64bigger", "huge"},
	}
	
	forall(choices, func(x []string) {
		p, a, e, c, n, i := x[0], x[1], x[2], x[3], x[4], x[5]

		// Pass: dynamic=0, static=1, 2.
		// Which cases should be caught statically?
		// Only constants, obviously.
		// Beyond that, must be one of these:
		//	indexing into array or pointer to array
		//	negative constant
		//	large constant
		thisPass := 0
		if c == "c" && (a == "a" || a == "pa" || n == "n" || i == "i64big" || i == "i64bigger" || i == "huge") {
			if i == "huge" {
				// Due to a detail of 6g's internals,
				// the huge constant errors happen in an
				// earlier pass than the others and inhibits
				// the next pass from running.
				// So run it as a separate check.
				thisPass = 1
			} else {
				thisPass = 2
			}
		}

		// Only print the test case if it is appropriate for this pass.
		if thisPass == *pass {
			pae := p+a+e
			cni := c+n+i
			
			// Index operation
			testExpr(b, pae + "[" + cni + "]")
			
			// Slice operation.
			// Low index 0 is a special case in ggen.c
			// so test both 0 and 1.
			testExpr(b, pae + "[0:" + cni + "]")
			testExpr(b, pae + "[1:" + cni + "]")
			testExpr(b, pae + "[" + cni + ":]")
			testExpr(b, pae + "[" + cni + ":" + cni + "]")
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
			recurse(d+1)
		}
	}
	recurse(0)
}
