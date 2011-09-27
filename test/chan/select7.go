// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test select when discarding a value.

package main

import "runtime"

func recv1(c <-chan int) {
	<-c
}

func recv2(c <-chan int) {
	select {
	case <-c:
	}
}

func recv3(c <-chan int) {
	c2 := make(chan int)
	select {
	case <-c:
	case <-c2:
	}
}

func send1(recv func(<-chan int)) {
	c := make(chan int)
	go recv(c)
	runtime.Gosched()
	c <- 1
}

func send2(recv func(<-chan int)) {
	c := make(chan int)
	go recv(c)
	runtime.Gosched()
	select {
	case c <- 1:
	}
}

func send3(recv func(<-chan int)) {
	c := make(chan int)
	go recv(c)
	runtime.Gosched()
	c2 := make(chan int)
	select {
	case c <- 1:
	case c2 <- 1:
	}
}

func main() {
	send1(recv1)
	send2(recv1)
	send3(recv1)
	send1(recv2)
	send2(recv2)
	send3(recv2)
	send1(recv3)
	send2(recv3)
	send3(recv3)
}
