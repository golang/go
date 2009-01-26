// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const	N	= 1000;		// sent messages
const	M	= 10;		// receiving goroutines
const	W	= 2;		// channel buffering
var	h	[N]int;		// marking of send/recv

func
r(c chan int, m int)
{
	for {
	       	select {
		case r := <- c:
			if h[r] != 1 {
				panicln("r",
					"m=", m,
					"r=", r,
					"h=", h[r]
				);
			}
			h[r] = 2;
		}
        }
}

func
s(c chan int)
{
	for n:=0; n<N; n++ {
		r := n;
		if h[r] != 0 {
			panicln("s");
		}
		h[r] = 1;
		c <- r;
	}
}

func
main()
{
	c := make(chan int, W);
	for m:=0; m<M; m++ {
		go r(c, m);
		sys.Gosched();
	}
	sys.Gosched();
	sys.Gosched();
	s(c);
}
