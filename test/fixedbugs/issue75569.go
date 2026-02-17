// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func fff(a []int, b bool, p, q *int) {
outer:
	n := a[0]
	a = a[1:]
	switch n {
	case 1:
		goto one
	case 2:
		goto two
	case 3:
		goto three
	case 4:
		goto four
	}

one:
	goto inner
two:
	goto outer
three:
	goto inner
four:
	goto innerSideEntry

inner:
	n = a[0]
	a = a[1:]
	switch n {
	case 1:
		goto outer
	case 2:
		goto inner
	case 3:
		goto innerSideEntry
	default:
		return
	}
innerSideEntry:
	n = a[0]
	a = a[1:]
	switch n {
	case 1:
		goto outer
	case 2:
		goto inner
	case 3:
		goto inner
	}
	ggg(p, q)
	goto inner
}

var b bool

func ggg(p, q *int) {
	n := *p + 5 // this +5 ends up in the entry block, well before the *p load
	if b {
		*q = 0
	}
	*p = n
}

func main() {
	var x, y int
	fff([]int{4, 4, 4}, false, &x, &y)
	if x != 5 {
		panic(x)
	}
}
