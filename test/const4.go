// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test len constants and non-constants, http://golang.org/issue/3244.

package main

var b struct {
	a[10]int
}

var m map[string][20]int

var s [][30]int

const (
	n1 = len(b.a)
	n2 = len(m[""])
	n3 = len(s[10])
)

// Non-constants (see also const5.go).
var (
	n4 = len(f())
	n5 = len(<-c)
	n6 = cap(g())
	n7 = cap(<-c1)
)

var calledF = false

func f() *[40]int {
	calledF = true
	return nil
}

var c = func() chan *[50]int {
	c := make(chan *[50]int, 2)
	c <- nil
	c <- new([50]int)
	return c
}()

var calledG = false

func g() *[60]int {
	calledG = true
	return nil
}

var c1 = func() chan *[70]int {
	c := make(chan *[70]int, 2)
	c <- nil
	c <- new([70]int)
	return c
}()

func main() {
	if n1 != 10 || n2 != 20 || n3 != 30 || n4 != 40 || n5 != 50 || n6 != 60 || n7 != 70 {
		println("BUG:", n1, n2, n3, n4, n5, n6, n7)
	}
	if !calledF {
		println("BUG: did not call f")
	}
	if <-c == nil {
		println("BUG: did not receive from c")
	}
	if !calledG {
		println("BUG: did not call g")
	}
	if <-c1 == nil {
		println("BUG: did not receive from c1")
	}
}
