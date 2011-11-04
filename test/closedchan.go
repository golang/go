// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test close(c), receive of closed channel.
//
// TODO(rsc): Doesn't check behavior of close(c) when there
// are blocked senders/receivers.

package main

import "os"

var failed bool

type Chan interface {
	Send(int)
	Nbsend(int) bool
	Recv() (int)
	Nbrecv() (int, bool)
	Recv2() (int, bool)
	Nbrecv2() (int, bool, bool)
	Close()
	Impl() string
}

// direct channel operations when possible
type XChan chan int

func (c XChan) Send(x int) {
	c <- x
}

func (c XChan) Nbsend(x int) bool {
	select {
	case c <- x:
		return true
	default:
		return false
	}
	panic("nbsend")
}

func (c XChan) Recv() int {
	return <-c
}

func (c XChan) Nbrecv() (int, bool) {
	select {
	case x := <-c:
		return x, true
	default:
		return 0, false
	}
	panic("nbrecv")
}

func (c XChan) Recv2() (int, bool) {
	x, ok := <-c
	return x, ok
}

func (c XChan) Nbrecv2() (int, bool, bool) {
	select {
	case x, ok := <-c:
		return x, ok, true
	default:
		return 0, false, false
	}
	panic("nbrecv2")
}

func (c XChan) Close() {
	close(c)
}

func (c XChan) Impl() string {
	return "(<- operator)"
}

// indirect operations via select
type SChan chan int

func (c SChan) Send(x int) {
	select {
	case c <- x:
	}
}

func (c SChan) Nbsend(x int) bool {
	select {
	default:
		return false
	case c <- x:
		return true
	}
	panic("nbsend")
}

func (c SChan) Recv() int {
	select {
	case x := <-c:
		return x
	}
	panic("recv")
}

func (c SChan) Nbrecv() (int, bool) {
	select {
	default:
		return 0, false
	case x := <-c:
		return x, true
	}
	panic("nbrecv")
}

func (c SChan) Recv2() (int, bool) {
	select {
	case x, ok := <-c:
		return x, ok
	}
	panic("recv")
}

func (c SChan) Nbrecv2() (int, bool, bool) {
	select {
	default:
		return 0, false, false
	case x, ok := <-c:
		return x, ok, true
	}
	panic("nbrecv")
}

func (c SChan) Close() {
	close(c)
}

func (c SChan) Impl() string {
	return "(select)"
}

// indirect operations via larger selects
var dummy = make(chan bool)

type SSChan chan int

func (c SSChan) Send(x int) {
	select {
	case c <- x:
	case <-dummy:
	}
}

func (c SSChan) Nbsend(x int) bool {
	select {
	default:
		return false
	case <-dummy:
	case c <- x:
		return true
	}
	panic("nbsend")
}

func (c SSChan) Recv() int {
	select {
	case <-dummy:
	case x := <-c:
		return x
	}
	panic("recv")
}

func (c SSChan) Nbrecv() (int, bool) {
	select {
	case <-dummy:
	default:
		return 0, false
	case x := <-c:
		return x, true
	}
	panic("nbrecv")
}

func (c SSChan) Recv2() (int, bool) {
	select {
	case <-dummy:
	case x, ok := <-c:
		return x, ok
	}
	panic("recv")
}

func (c SSChan) Nbrecv2() (int, bool, bool) {
	select {
	case <-dummy:
	default:
		return 0, false, false
	case x, ok := <-c:
		return x, ok, true
	}
	panic("nbrecv")
}

func (c SSChan) Close() {
	close(c)
}

func (c SSChan) Impl() string {
	return "(select)"
}


func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("did not panic")
		}
	}()
	f()
}

func test1(c Chan) {
	for i := 0; i < 3; i++ {
		// recv a close signal (a zero value)
		if x := c.Recv(); x != 0 {
			println("test1: recv on closed:", x, c.Impl())
			failed = true
		}
		if x, ok := c.Recv2(); x != 0 || ok {
			println("test1: recv2 on closed:", x, ok, c.Impl())
			failed = true
		}

		// should work with select: received a value without blocking, so selected == true.
		x, selected := c.Nbrecv()
		if x != 0 || !selected {
			println("test1: recv on closed nb:", x, selected, c.Impl())
			failed = true
		}
		x, ok, selected := c.Nbrecv2()
		if x != 0 || ok || !selected {
			println("test1: recv2 on closed nb:", x, ok, selected, c.Impl())
			failed = true
		}
	}

	// send should work with ,ok too: sent a value without blocking, so ok == true.
	shouldPanic(func() { c.Nbsend(1) })

	// the value should have been discarded.
	if x := c.Recv(); x != 0 {
		println("test1: recv on closed got non-zero after send on closed:", x, c.Impl())
		failed = true
	}

	// similarly Send.
	shouldPanic(func() { c.Send(2) })
	if x := c.Recv(); x != 0 {
		println("test1: recv on closed got non-zero after send on closed:", x, c.Impl())
		failed = true
	}
}

func testasync1(c Chan) {
	// should be able to get the last value via Recv
	if x := c.Recv(); x != 1 {
		println("testasync1: Recv did not get 1:", x, c.Impl())
		failed = true
	}

	test1(c)
}

func testasync2(c Chan) {
	// should be able to get the last value via Recv2
	if x, ok := c.Recv2(); x != 1 || !ok {
		println("testasync1: Recv did not get 1, true:", x, ok, c.Impl())
		failed = true
	}

	test1(c)
}

func testasync3(c Chan) {
	// should be able to get the last value via Nbrecv
	if x, selected := c.Nbrecv(); x != 1 || !selected {
		println("testasync2: Nbrecv did not get 1, true:", x, selected, c.Impl())
		failed = true
	}

	test1(c)
}

func testasync4(c Chan) {
	// should be able to get the last value via Nbrecv2
	if x, ok, selected := c.Nbrecv2(); x != 1 || !ok || !selected {
		println("testasync2: Nbrecv did not get 1, true, true:", x, ok, selected, c.Impl())
		failed = true
	}
	test1(c)
}

func closedsync() chan int {
	c := make(chan int)
	close(c)
	return c
}

func closedasync() chan int {
	c := make(chan int, 2)
	c <- 1
	close(c)
	return c
}

var mks = []func(chan int) Chan {
	func(c chan int) Chan { return XChan(c) },
	func(c chan int) Chan { return SChan(c) },
	func(c chan int) Chan { return SSChan(c) },
}

var testcloseds = []func(Chan) {
	testasync1,
	testasync2,
	testasync3,
	testasync4,
}

func main() {
	for _, mk := range mks {
		test1(mk(closedsync()))
	}
	
	for _, testclosed := range testcloseds {
		for _, mk := range mks {
			testclosed(mk(closedasync()))
		}
	}
	
	var ch chan int	
	shouldPanic(func() {
		close(ch)
	})
	
	ch = make(chan int)
	close(ch)
	shouldPanic(func() {
		close(ch)
	})

	if failed {
		os.Exit(1)
	}
}
