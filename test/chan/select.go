// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var counter uint
var shift uint

func GetValue() uint {
	counter++
	return 1 << shift
}

func Send(a, b chan uint) int {
	var i int

LOOP:
	for {
		select {
		case a <- GetValue():
			i++
			a = nil
		case b <- GetValue():
			i++
			b = nil
		default:
			break LOOP
		}
		shift++
	}
	return i
}

func main() {
	a := make(chan uint, 1)
	b := make(chan uint, 1)
	if v := Send(a, b); v != 2 {
		println("Send returned", v, "!= 2")
		panic("fail")
	}
	if av, bv := <-a, <-b; av|bv != 3 {
		println("bad values", av, bv)
		panic("fail")
	}
	if v := Send(a, nil); v != 1 {
		println("Send returned", v, "!= 1")
		panic("fail")
	}
	if counter != 10 {
		println("counter is", counter, "!= 10")
		panic("fail")
	}
}
