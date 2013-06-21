// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program is processed by the cover command, and then testAll is called.
// The test driver in main.go can then compare the coverage statistics with expectation.

// The word LINE is replaced by the line number in this file. When the file is executed,
// the coverage processing has changed the line numbers, so we can't use runtime.Caller.

package main

const anything = 1e9 // Just some unlikely value that means "we got here, don't care how often"

func testAll() {
	testSimple()
	testBlockRun()
	testFor()
	testSwitch()
	testSelect1()
	testSelect2()
}

func testSimple() {
	check(LINE, 1)
}

func testFor() {
	for i := 0; i < 10; i++ {
		check(LINE, 10)
	}
}

func testBlockRun() {
	check(LINE, 1)
	{
		check(LINE, 1)
	}
	{
		check(LINE, 1)
	}
	check(LINE, 1)
	{
		check(LINE, 1)
	}
	{
		check(LINE, 1)
	}
	check(LINE, 1)
}

func testSwitch() {
	for i := 0; i < 5; i++ {
		switch i {
		case 0:
			check(LINE, 1)
		case 1:
			check(LINE, 1)
		case 2:
			check(LINE, 1)
		default:
			check(LINE, 2)
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
			check(LINE, anything)
		case <-c:
			check(LINE, anything)
		default:
			check(LINE, 1)
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
			check(LINE, 1000)
		case <-c2:
			check(LINE, 1000)
		default:
			check(LINE, 1)
			return
		}
	}
}
