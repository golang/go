// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test communication operations including select.

package main

import "os"
import "runtime"
import "sync"

var randx int

func nrand(n int) int {
	randx += 10007
	if randx >= 1000000 {
		randx -= 1000000
	}
	return randx % n
}

type Chan struct {
	sc, rc chan int // send and recv chan
	sv, rv int      // send and recv seq
}

var (
	nproc      int
	nprocLock  sync.Mutex
	cval       int
	end        int = 10000
	totr, tots int
	totLock    sync.Mutex
	nc         *Chan
)

func init() {
	nc = new(Chan)
}

func changeNproc(adjust int) int {
	nprocLock.Lock()
	nproc += adjust
	ret := nproc
	nprocLock.Unlock()
	return ret
}

func mkchan(c, n int) []*Chan {
	ca := make([]*Chan, n)
	for i := 0; i < n; i++ {
		cval = cval + 100
		ch := new(Chan)
		ch.sc = make(chan int, c)
		ch.rc = ch.sc
		ch.sv = cval
		ch.rv = cval
		ca[i] = ch
	}
	return ca
}

func expect(v, v0 int) (newv int) {
	if v == v0 {
		if v%100 == 75 {
			return end
		}
		return v + 1
	}
	print("got ", v, " expected ", v0+1, "\n")
	panic("fail")
}

func (c *Chan) send() bool {
	//	print("send ", c.sv, "\n");
	totLock.Lock()
	tots++
	totLock.Unlock()
	c.sv = expect(c.sv, c.sv)
	if c.sv == end {
		c.sc = nil
		return true
	}
	return false
}

func send(c *Chan) {
	for {
		for r := nrand(10); r >= 0; r-- {
			runtime.Gosched()
		}
		c.sc <- c.sv
		if c.send() {
			break
		}
	}
	changeNproc(-1)
}

func (c *Chan) recv(v int) bool {
	//	print("recv ", v, "\n");
	totLock.Lock()
	totr++
	totLock.Unlock()
	c.rv = expect(c.rv, v)
	if c.rv == end {
		c.rc = nil
		return true
	}
	return false
}

func recv(c *Chan) {
	var v int

	for {
		for r := nrand(10); r >= 0; r-- {
			runtime.Gosched()
		}
		v = <-c.rc
		if c.recv(v) {
			break
		}
	}
	changeNproc(-1)
}

func sel(r0, r1, r2, r3, s0, s1, s2, s3 *Chan) {
	var v int

	a := 0 // local chans running

	if r0.rc != nil {
		a++
	}
	if r1.rc != nil {
		a++
	}
	if r2.rc != nil {
		a++
	}
	if r3.rc != nil {
		a++
	}
	if s0.sc != nil {
		a++
	}
	if s1.sc != nil {
		a++
	}
	if s2.sc != nil {
		a++
	}
	if s3.sc != nil {
		a++
	}

	for {
		for r := nrand(5); r >= 0; r-- {
			runtime.Gosched()
		}

		select {
		case v = <-r0.rc:
			if r0.recv(v) {
				a--
			}
		case v = <-r1.rc:
			if r1.recv(v) {
				a--
			}
		case v = <-r2.rc:
			if r2.recv(v) {
				a--
			}
		case v = <-r3.rc:
			if r3.recv(v) {
				a--
			}
		case s0.sc <- s0.sv:
			if s0.send() {
				a--
			}
		case s1.sc <- s1.sv:
			if s1.send() {
				a--
			}
		case s2.sc <- s2.sv:
			if s2.send() {
				a--
			}
		case s3.sc <- s3.sv:
			if s3.send() {
				a--
			}
		}
		if a == 0 {
			break
		}
	}
	changeNproc(-1)
}

// direct send to direct recv
func test1(c *Chan) {
	changeNproc(2)
	go send(c)
	go recv(c)
}

// direct send to select recv
func test2(c int) {
	ca := mkchan(c, 4)

	changeNproc(4)
	go send(ca[0])
	go send(ca[1])
	go send(ca[2])
	go send(ca[3])

	changeNproc(1)
	go sel(ca[0], ca[1], ca[2], ca[3], nc, nc, nc, nc)
}

// select send to direct recv
func test3(c int) {
	ca := mkchan(c, 4)

	changeNproc(4)
	go recv(ca[0])
	go recv(ca[1])
	go recv(ca[2])
	go recv(ca[3])

	changeNproc(1)
	go sel(nc, nc, nc, nc, ca[0], ca[1], ca[2], ca[3])
}

// select send to select recv
func test4(c int) {
	ca := mkchan(c, 4)

	changeNproc(2)
	go sel(nc, nc, nc, nc, ca[0], ca[1], ca[2], ca[3])
	go sel(ca[0], ca[1], ca[2], ca[3], nc, nc, nc, nc)
}

func test5(c int) {
	ca := mkchan(c, 8)

	changeNproc(2)
	go sel(ca[4], ca[5], ca[6], ca[7], ca[0], ca[1], ca[2], ca[3])
	go sel(ca[0], ca[1], ca[2], ca[3], ca[4], ca[5], ca[6], ca[7])
}

func test6(c int) {
	ca := mkchan(c, 12)

	changeNproc(4)
	go send(ca[4])
	go send(ca[5])
	go send(ca[6])
	go send(ca[7])

	changeNproc(4)
	go recv(ca[8])
	go recv(ca[9])
	go recv(ca[10])
	go recv(ca[11])

	changeNproc(2)
	go sel(ca[4], ca[5], ca[6], ca[7], ca[0], ca[1], ca[2], ca[3])
	go sel(ca[0], ca[1], ca[2], ca[3], ca[8], ca[9], ca[10], ca[11])
}

// wait for outstanding tests to finish
func wait() {
	runtime.Gosched()
	for changeNproc(0) != 0 {
		runtime.Gosched()
	}
}

// run all tests with specified buffer size
func tests(c int) {
	ca := mkchan(c, 4)
	test1(ca[0])
	test1(ca[1])
	test1(ca[2])
	test1(ca[3])
	wait()

	test2(c)
	wait()

	test3(c)
	wait()

	test4(c)
	wait()

	test5(c)
	wait()

	test6(c)
	wait()
}

// run all test with 4 buffser sizes
func main() {

	tests(0)
	tests(1)
	tests(10)
	tests(100)

	t := 4 * // buffer sizes
		(4*4 + // tests 1,2,3,4 channels
			8 + // test 5 channels
			12) * // test 6 channels
		76 // sends/recvs on a channel

	if tots != t || totr != t {
		print("tots=", tots, " totr=", totr, " sb=", t, "\n")
		os.Exit(1)
	}
	os.Exit(0)
}
