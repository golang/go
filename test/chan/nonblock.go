// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify channel operations that test for blocking
// Use several sizes and types of operands

package main

func pause() {
	for i:=0; i<100; i++ { sys.Gosched() }
}

func i32receiver(c chan int32) {
	if <-c != 123 { panic("i32 value") }
}

func i32sender(c chan int32) {
	c <- 234
}

func i64receiver(c chan int64) {
	if <-c != 123456 { panic("i64 value") }
}

func i64sender(c chan int64) {
	c <- 234567
}

func breceiver(c chan bool) {
	if ! <-c { panic("b value") }
}

func bsender(c chan bool) {
	c <- true
}

func sreceiver(c chan string) {
	if <-c != "hello" { panic("s value") }
}

func ssender(c chan string) {
	c <- "hello again"
}

func main() {
	var i32 int32;
	var i64 int64;
	var b bool;
	var s string;
	var ok bool;

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

		go i32receiver(c32);
		pause();
		ok = c32 <- 123;
		if !ok { panic("i32receiver") }
		go i32sender(c32);
		pause();
		i32, ok = <-c32;
		if !ok { panic("i32sender") }
		if i32 != 234 { panic("i32sender value") }

		go i64receiver(c64);
		pause();
		ok = c64 <- 123456;
		if !ok { panic("i64receiver") }
		go i64sender(c64);
		pause();
		i64, ok = <-c64;
		if !ok { panic("i64sender") }
		if i64 != 234567 { panic("i64sender value") }

		go breceiver(cb);
		pause();
		ok = cb <- true;
		if !ok { panic("breceiver") }
		go bsender(cb);
		pause();
		b, ok = <-cb;
		if !ok { panic("bsender") }
		if !b{ panic("bsender value") }

		go sreceiver(cs);
		pause();
		ok = cs <- "hello";
		if !ok { panic("sreceiver") }
		go ssender(cs);
		pause();
		s, ok = <-cs;
		if !ok { panic("ssender") }
		if s != "hello again" { panic("ssender value") }
	}
	print("PASS\n")
}
