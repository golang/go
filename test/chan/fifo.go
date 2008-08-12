// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that unbuffered channels act as pure fifos.

package main

const N = 10

func AsynchFifo() {
	ch := new(chan int, N);
	for i := 0; i < N; i++ {
		ch -< i
	}
	for i := 0; i < N; i++ {
		if <-ch != i {
			print("bad receive\n");
			sys.exit(1);
		}
	}
}

func Chain(ch *chan<- int, val int, in *chan<- int, out *chan-< int) {
	<-in;
	if <-ch != val {
		panic(val)
	}
	out -< 1
}

// thread together a daisy chain to read the elements in sequence
func SynchFifo() {
	ch := new(chan int);
	in := new(chan int);
	start := in;
	for i := 0; i < N; i++ {
		out := new(chan int);
		go Chain(ch, i, in, out);
		in = out;
	}
	start -< 0;
	for i := 0; i < N; i++ {
		ch -< i
	}
	<-in
}

func main() {
	AsynchFifo();
	SynchFifo();
}

