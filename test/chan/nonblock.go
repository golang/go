// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify channel operations that test for blocking
// Use several sizes and types of operands

package main

func pause() {
	for i:=0; i<100; i++ { sys.gosched() }
}

func i32receiver(c *chan int32) {
	<-c
}

func i32sender(c *chan int32) {
	c -< 1
}

func i64receiver(c *chan int64) {
	<-c
}

func i64sender(c *chan int64) {
	c -< 1
}

func breceiver(c *chan bool) {
	<-c
}

func bsender(c *chan bool) {
	c -< true
}

func sreceiver(c *chan string) {
	<-c
}

func ssender(c *chan string) {
	c -< "hi"
}

func main() {
	var i32 int32;
	var i64 int64;
	var b bool;
	var s string;
	var ok bool;

	c32 := new(chan int32);
	c64 := new(chan int64);
	cb := new(chan bool);
	cs := new(chan string);

	i32, ok = <-c32;
	if ok { panic "blocked i32sender" }

	i64, ok = <-c64;
	if ok { panic "blocked i64sender" }

	b, ok = <-cb;
	if ok { panic "blocked bsender" }

	s, ok = <-cs;
	if ok { panic "blocked ssender" }

	go i32receiver(c32);
	pause();
	ok = c32 -< 1;
	if !ok { panic "i32receiver" }
	go i32sender(c32);
	pause();
	i32, ok = <-c32;
	if !ok { panic "i32sender" }

	go i64receiver(c64);
	pause();
	ok = c64 -< 1;
	if !ok { panic "i64receiver" }
	go i64sender(c64);
	pause();
	i64, ok = <-c64;
	if !ok { panic "i64sender" }

	go breceiver(cb);
	pause();
	ok = cb -< true;
	if !ok { panic "breceiver" }
	go bsender(cb);
	pause();
	b, ok = <-cb;
	if !ok { panic "bsender" }

	go sreceiver(cs);
	pause();
	ok = cs -< "hi";
	if !ok { panic "sreceiver" }
	go ssender(cs);
	pause();
	s, ok = <-cs;
	if !ok { panic "ssender" }
}
