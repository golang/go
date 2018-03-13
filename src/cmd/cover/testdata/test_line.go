// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program is processed by the cover command, and then testAll is called.
// The test driver in main.go can then compare the coverage statistics with expectation.

// The word 8 is replaced by the line number in this file. When the file is executed,
// the coverage processing has changed the line numbers, so we can't use runtime.Caller.

package main

import _ "unsafe" // for go:linkname

//go:linkname some_name some_name

const anything = 1e9 // Just some unlikely value that means "we got here, don't care how often"

func testAll() {
	testSimple()
	testBlockRun()
	testIf()
	testFor()
	testRange()
	testSwitch()
	testTypeSwitch()
	testSelect1()
	testSelect2()
	testPanic()
	testEmptySwitches()
	testFunctionLiteral()
	testGoto()
}

// The indexes of the counters in testPanic are known to main.go
const panicIndex = 3

// This test appears first because the index of its counters is known to main.go
func testPanic() {
	defer func() {
		recover()
	}()
	check(43, 1)
	panic("should not get next line")
	check(45, 0) // this is GoCover.Count[panicIndex]
	// The next counter is in testSimple and it will be non-zero.
	// If the panic above does not trigger a counter, the test will fail
	// because GoCover.Count[panicIndex] will be the one in testSimple.
}

func testSimple() {
	check(52, 1)
}

func testIf() {
	if true {
		check(57, 1)
	} else {
		check(59, 0)
	}
	if false {
		check(62, 0)
	} else {
		check(64, 1)
	}
	for i := 0; i < 3; i++ {
		if checkVal(67, 3, i) <= 2 {
			check(68, 3)
		}
		if checkVal(70, 3, i) <= 1 {
			check(71, 2)
		}
		if checkVal(73, 3, i) <= 0 {
			check(74, 1)
		}
	}
	for i := 0; i < 3; i++ {
		if checkVal(78, 3, i) <= 1 {
			check(79, 2)
		} else {
			check(81, 1)
		}
	}
	for i := 0; i < 3; i++ {
		if checkVal(85, 3, i) <= 0 {
			check(86, 1)
		} else if checkVal(87, 2, i) <= 1 {
			check(88, 1)
		} else if checkVal(89, 1, i) <= 2 {
			check(90, 1)
		} else if checkVal(91, 0, i) <= 3 {
			check(92, 0)
		}
	}
	if func(a, b int) bool { return a < b }(3, 4) {
		check(96, 1)
	}
}

func testFor() {
	for i := 0; i < 10; func() { i++; check(101, 10) }() {
		check(102, 10)
	}
}

func testRange() {
	for _, f := range []func(){
		func() { check(108, 1) },
	} {
		f()
		check(111, 1)
	}
}

func testBlockRun() {
	check(116, 1)
	{
		check(118, 1)
	}
	{
		check(121, 1)
	}
	check(123, 1)
	{
		check(125, 1)
	}
	{
		check(128, 1)
	}
	check(130, 1)
}

func testSwitch() {
	for i := 0; i < 5; func() { i++; check(134, 5) }() {
		switch i {
		case 0:
			check(137, 1)
		case 1:
			check(139, 1)
		case 2:
			check(141, 1)
		default:
			check(143, 2)
		}
	}
}

func testTypeSwitch() {
	var x = []interface{}{1, 2.0, "hi"}
	for _, v := range x {
		switch func() { check(151, 3) }(); v.(type) {
		case int:
			check(153, 1)
		case float64:
			check(155, 1)
		case string:
			check(157, 1)
		case complex128:
			check(159, 0)
		default:
			check(161, 0)
		}
	}
}

func testSelect1() {
	c := make(chan int)
	go func() {
		for i := 0; i < 1000; i++ {
			c <- i
		}
	}()
	for {
		select {
		case <-c:
			check(176, anything)
		case <-c:
			check(178, anything)
		default:
			check(180, 1)
			return
		}
	}
}

func testSelect2() {
	c1 := make(chan int, 1000)
	c2 := make(chan int, 1000)
	for i := 0; i < 1000; i++ {
		c1 <- i
		c2 <- i
	}
	for {
		select {
		case <-c1:
			check(196, 1000)
		case <-c2:
			check(198, 1000)
		default:
			check(200, 1)
			return
		}
	}
}

// Empty control statements created syntax errors. This function
// is here just to be sure that those are handled correctly now.
func testEmptySwitches() {
	check(209, 1)
	switch 3 {
	}
	check(212, 1)
	switch i := (interface{})(3).(int); i {
	}
	check(215, 1)
	c := make(chan int)
	go func() {
		check(218, 1)
		c <- 1
		select {}
	}()
	<-c
	check(223, 1)
}

func testFunctionLiteral() {
	a := func(f func()) error {
		f()
		f()
		return nil
	}

	b := func(f func()) bool {
		f()
		f()
		return true
	}

	check(239, 1)
	a(func() {
		check(241, 2)
	})

	if err := a(func() {
		check(245, 2)
	}); err != nil {
	}

	switch b(func() {
		check(250, 2)
	}) {
	}

	x := 2
	switch x {
	case func() int { check(256, 1); return 1 }():
		check(257, 0)
		panic("2=1")
	case func() int { check(259, 1); return 2 }():
		check(260, 1)
	case func() int { check(261, 0); return 3 }():
		check(262, 0)
		panic("2=3")
	}
}

func testGoto() {
	for i := 0; i < 2; i++ {
		if i == 0 {
			goto Label
		}
		check(272, 1)
	Label:
		check(274, 2)
	}
	// Now test that we don't inject empty statements
	// between a label and a loop.
loop:
	for {
		check(280, 1)
		break loop
	}
}

// This comment didn't appear in generated go code.
func haha() {
	// Needed for cover to add counter increment here.
	_ = 42
}

// Some someFunction.
//
//go:nosplit
func someFunction() {
}
