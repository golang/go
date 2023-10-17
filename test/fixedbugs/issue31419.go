// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 31419: race in getitab when two goroutines try
// to do the same failed interface conversion.

package main

type T int

func (t T) M() {}

type I interface {
	M()
	M2()
}

var t T
var e interface{} = &t
var ok = false
var ch = make(chan int)

func main() {
	_, ok = e.(I) // populate itab cache with a false result

	go f() // get itab in a loop

	var i I
	for k := 0; k < 10000; k++ {
		i, ok = e.(I) // read the cached itab
		if ok {
			println("iteration", k, "i =", i, "&t =", &t)
			panic("conversion succeeded")
		}
	}
	<-ch
}

func f() {
	for i := 0; i < 10000; i++ {
		f1()
	}
	ch <- 1
}

func f1() {
	defer func() {
		err := recover()
		if err == nil {
			panic("did not panic")
		}
	}()
	i := e.(I) // triggers itab.init, for getting the panic string
	_ = i
}
