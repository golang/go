// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is the input program for an end-to-end test of the DWARF produced
// by the compiler. It is compiled with various flags, then the resulting
// binary is "debugged" under the control of a harness.  Because the compile+debug
// step is time-consuming, the tests for different bugs are all accumulated here
// so that their cost is only the time to "n" through the additional code.

package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

type point struct {
	x, y int
}

type line struct {
	begin, end point
}

var zero int
var sink int

//go:noinline
func tinycall() {
}

func ensure(n int, sl []int) []int {
	for len(sl) <= n {
		sl = append(sl, 0)
	}
	return sl
}

var cannedInput string = `1
1
1
2
2
2
4
4
5
`

func test() {
	// For #19868
	l := line{point{1 + zero, 2 + zero}, point{3 + zero, 4 + zero}}
	tinycall()                // this forces l etc to stack
	dx := l.end.x - l.begin.x //gdb-dbg=(l.begin.x,l.end.y)//gdb-opt=(l,dx/O,dy/O)
	dy := l.end.y - l.begin.y //gdb-opt=(dx,dy/O)
	sink = dx + dy            //gdb-opt=(dx,dy)
	// For #21098
	hist := make([]int, 7)                                //gdb-opt=(dx/O,dy/O) // TODO sink is missing if this code is in 'test' instead of 'main'
	var reader io.Reader = strings.NewReader(cannedInput) //gdb-dbg=(hist/A) // TODO cannedInput/A is missing if this code is in 'test' instead of 'main'
	if len(os.Args) > 1 {
		var err error
		reader, err = os.Open(os.Args[1])
		if err != nil {
			fmt.Fprintf(os.Stderr, "There was an error opening %s: %v\n", os.Args[1], err)
			return
		}
	}
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() { //gdb-opt=(scanner/A)
		s := scanner.Text()
		i, err := strconv.ParseInt(s, 10, 64)
		if err != nil { //gdb-dbg=(i) //gdb-opt=(err,hist,i)
			fmt.Fprintf(os.Stderr, "There was an error: %v\n", err)
			return
		}
		hist = ensure(int(i), hist)
		hist[int(i)]++
	}
	t := 0
	n := 0
	for i, a := range hist {
		if a == 0 { //gdb-opt=(a,n,t)
			continue
		}
		t += i * a
		n += a
		fmt.Fprintf(os.Stderr, "%d\t%d\t%d\t%d\t%d\n", i, a, n, i*a, t) //gdb-dbg=(n,i,t)
	}
}

func main() {
	growstack() // Use stack early to prevent growth during test, which confuses gdb
	test()
}

var snk string

//go:noinline
func growstack() {
	snk = fmt.Sprintf("%#v,%#v,%#v", 1, true, "cat")
}
