// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test close(c), closed(c).
//
// TODO(rsc): Doesn't check behavior of close(c) when there
// are blocked senders/receivers.

package main

type Chan interface {
	Send(int)
	Nbsend(int) bool
	Recv() int
	Nbrecv() (int, bool)
	Close()
	Closed() bool
	Impl() string
}

// direct channel operations
type XChan chan int
func (c XChan) Send(x int) {
	c <- x
}

func (c XChan) Nbsend(x int) bool {
	return c <- x
}

func (c XChan) Recv() int {
	return <-c
}

func (c XChan) Nbrecv() (int, bool) {
	x, ok := <-c
	return x, ok
}

func (c XChan) Close() {
	close(c)
}

func (c XChan) Closed() bool {
	return closed(c)
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
	case c <- x:
		return true
	default:
		return false
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
	case x := <-c:
		return x, true
	default:
		return 0, false
	}
	panic("nbrecv")
}

func (c SChan) Close() {
	close(c)
}

func (c SChan) Closed() bool {
	return closed(c)
}

func (c SChan) Impl() string {
	return "(select)"
}

func test1(c Chan) {
	// not closed until the close signal (a zero value) has been received.
	if c.Closed() {
		println("test1: Closed before Recv zero:", c.Impl())
	}

	for i := 0; i < 3; i++ {
		// recv a close signal (a zero value)
		if x := c.Recv(); x != 0 {
			println("test1: recv on closed got non-zero:", x, c.Impl())
		}

		// should now be closed.
		if !c.Closed() {
			println("test1: not closed after recv zero", c.Impl())
		}

		// should work with ,ok: received a value without blocking, so ok == true.
		x, ok := c.Nbrecv()
		if !ok {
			println("test1: recv on closed got not ok", c.Impl())
		}
		if x != 0 {
			println("test1: recv ,ok on closed got non-zero:", x, c.Impl())
		}
	}

	// send should work with ,ok too: sent a value without blocking, so ok == true.
	ok := c.Nbsend(1)
	if !ok {
		println("test1: send on closed got not ok", c.Impl())
	}

	// but the value should have been discarded.
	if x := c.Recv(); x != 0 {
		println("test1: recv on closed got non-zero after send on closed:", x, c.Impl())
	}

	// similarly Send.
	c.Send(2)
	if x := c.Recv(); x != 0 {
		println("test1: recv on closed got non-zero after send on closed:", x, c.Impl())
	}
}

func testasync1(c Chan) {
	// not closed until the close signal (a zero value) has been received.
	if c.Closed() {
		println("testasync1: Closed before Recv zero:", c.Impl())
	}

	// should be able to get the last value via Recv
	if x := c.Recv(); x != 1 {
		println("testasync1: Recv did not get 1:", x, c.Impl())
	}

	test1(c)
}

func testasync2(c Chan) {
	// not closed until the close signal (a zero value) has been received.
	if c.Closed() {
		println("testasync2: Closed before Recv zero:", c.Impl())
	}

	// should be able to get the last value via Nbrecv
	if x, ok := c.Nbrecv(); !ok || x != 1 {
		println("testasync2: Nbrecv did not get 1, true:", x, ok, c.Impl())
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

func main() {
	test1(XChan(closedsync()))
	test1(SChan(closedsync()))

	testasync1(XChan(closedasync()))
	testasync1(SChan(closedasync()))
	testasync2(XChan(closedasync()))
	testasync2(SChan(closedasync()))
}
