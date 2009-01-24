// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify channel operations that test for blocking
// Use several sizes and types of operands

package main

import "time"

func i32receiver(c chan int32, strobe chan bool) {
	if <-c != 123 { panic("i32 value") }
	strobe <- true
}

func i32sender(c chan int32, strobe chan bool) {
	c <- 234;
	strobe <- true
}

func i64receiver(c chan int64, strobe chan bool) {
	if <-c != 123456 { panic("i64 value") }
	strobe <- true
}

func i64sender(c chan int64, strobe chan bool) {
	c <- 234567;
	strobe <- true
}

func breceiver(c chan bool, strobe chan bool) {
	if ! <-c { panic("b value") }
	strobe <- true
}

func bsender(c chan bool, strobe chan bool) {
	c <- true;
	strobe <- true
}

func sreceiver(c chan string, strobe chan bool) {
	if <-c != "hello" { panic("s value") }
	strobe <- true
}

func ssender(c chan string, strobe chan bool) {
	c <- "hello again";
	strobe <- true
}

var ticker = time.Tick(10*1000);	// 10 us
func sleep() {
	<-ticker;
	<-ticker;
	sys.Gosched();
	sys.Gosched();
	sys.Gosched();
}

func main() {
	var i32 int32;
	var i64 int64;
	var b bool;
	var s string;
	var ok bool;

	var sync = make(chan bool);

	for buffer := 0; buffer < 2; buffer++ {
		c32 := make(chan int32, buffer);
		c64 := make(chan int64, buffer);
		cb := make(chan bool, buffer);
		cs := make(chan string, buffer);

		i32, ok = <-c32;
		if ok { panic("blocked i32sender") }

		i64, ok = <-c64;
		if ok { panic("blocked i64sender") }

		b, ok = <-cb;
		if ok { panic("blocked bsender") }

		s, ok = <-cs;
		if ok { panic("blocked ssender") }

		go i32receiver(c32, sync);
		sleep();
		ok = c32 <- 123;
		if !ok { panic("i32receiver buffer=", buffer) }
		<-sync;

		go i32sender(c32, sync);
		if buffer > 0 { <-sync } else { sleep() }
		i32, ok = <-c32;
		if !ok { panic("i32sender buffer=", buffer) }
		if i32 != 234 { panic("i32sender value") }
		if buffer == 0 { <-sync }

		go i64receiver(c64, sync);
		sleep();
		ok = c64 <- 123456;
		if !ok { panic("i64receiver") }
		<-sync;

		go i64sender(c64, sync);
		if buffer > 0 { <-sync } else { sleep() }
		i64, ok = <-c64;
		if !ok { panic("i64sender") }
		if i64 != 234567 { panic("i64sender value") }
		if buffer == 0 { <-sync }

		go breceiver(cb, sync);
		sleep();
		ok = cb <- true;
		if !ok { panic("breceiver") }
		<-sync;

		go bsender(cb, sync);
		if buffer > 0 { <-sync } else { sleep() }
		b, ok = <-cb;
		if !ok { panic("bsender") }
		if !b{ panic("bsender value") }
		if buffer == 0 { <-sync }

		go sreceiver(cs, sync);
		sleep();
		ok = cs <- "hello";
		if !ok { panic("sreceiver") }
		<-sync;

		go ssender(cs, sync);
		if buffer > 0 { <-sync } else { sleep() }
		s, ok = <-cs;
		if !ok { panic("ssender") }
		if s != "hello again" { panic("ssender value") }
		if buffer == 0 { <-sync }
	}
	print("PASS\n")
}
