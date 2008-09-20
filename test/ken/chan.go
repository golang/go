// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import	rand "rand"

var
(
	c0	*chan int;
	c1	*chan int;
	c2	*chan int;
	c3	*chan int;
	n	int;
	End	int	= 1000;
	totr	int;
	tots	int;
)

func
mkchan(c uint)
{
	n = 0;

	c0 = new(chan int, c);
	c1 = new(chan int, c);
	c2 = new(chan int, c);
	c3 = new(chan int, c);

//	print("c0=", c0, "\n");
//	print("c1=", c1, "\n");
//	print("c2=", c2, "\n");
//	print("c3=", c3, "\n");
}

func
expect(v, v0 int) (newv int)
{
	if v == v0 {
		if v%100 == 75 {
			return End;
		}
		return v+1;
	}
	panic("got ", v, " expected ", v0+1, "\n");
}

func
send(c *chan int, v0 int)
{
	n++;
	for {
		for r:=rand.nrand(10); r>=0; r-- {
			sys.gosched();
		}
		c <- v0;
		tots++;
		v0 = expect(v0, v0);
		if v0 == End {
			break;
		}
	}
	n--;
}

func
selsend()
{
	var v int;

	a := 4;		// local chans running
	n += a;		// total chans running
	v0 := 100;
	v1 := 200;
	v2 := 300;
	v3 := 400;

	// local copies of the chans
	// so we can nil them out
	l0 := c0;
	l1 := c1;
	l2 := c2;
	l3 := c3;

	for {
		for r:=rand.nrand(5); r>=0; r-- {
			sys.gosched();
		}

		select {
		case l0 <- v0:
			v0 = expect(v0, v0);
			if v0 == End {
				l0 = nil;
				a--;
			}
		case l1 <- v1:
			v1 = expect(v1, v1);
			if v1 == End {
				l1 = nil;
				a--;
			}
		case l2 <- v2:
			v2 = expect(v2, v2);
			if v2 == End {
				l2 = nil;
				a--;
			}
		case l3 <- v3:
			v3 = expect(v3, v3);
			if v3 == End {
				l3 = nil;
				a--;
			}
		}

		tots++;
		if a == 0 {
			break;
		}
	}
	n -= 4;
}

func
recv(c *chan int, v0 int)
{
	var v int;

	n++;
	for i:=0; i<100; i++ {
		for r:=rand.nrand(10); r>=0; r-- {
			sys.gosched();
		}
		v = <- c;
		totr++;
		v0 = expect(v0, v);
		if v0 == End {
			break;
		}
	}
	n--;
}

func
selrecv()
{
	var v int;

	a := 4;		// local chans running
	n += a;		// total chans running
	v0 := 100;
	v1 := 200;
	v2 := 300;
	v3 := 400;

	for {
		for r:=rand.nrand(5); r>=0; r-- {
			sys.gosched();
		}

		select {
		case v = <- c0:
			v0 = expect(v0, v);
			if v0 == End {
				a--;
			}
		case v = <- c1:
			v1 = expect(v1, v);
			if v1 == End {
				a--;
			}
		case v = <- c2:
			v2 = expect(v2, v);
			if v2 == End {
				a--;
			}
		case v = <- c3:
			v3 = expect(v3, v);
			if v3 == End {
				a--;
			}
		}

		totr++;
		if a == 0 {
			break;
		}
	}
	n -= 4;
}

// direct send to direct recv
func
test1(c *chan int, v0 int)
{
	go send(c, v0);
	go recv(c, v0);
}

// direct send to select recv
func
test2()
{
	go send(c0, 100);
	go send(c1, 200);
	go send(c2, 300);
	go send(c3, 400);
	go selrecv();
}

// select send to direct recv
func
test3()
{
	go recv(c0, 100);
	go recv(c1, 200);
	go recv(c2, 300);
	go recv(c3, 400);
	go selsend();
}

// wait for outstanding tests to finish
func
wait()
{
	sys.gosched();
	for n != 0 {
		sys.gosched();
	}
}

// run all tests with specified buffer size
func
tests(c uint)
{
	mkchan(c);
	test1(c0, 100);
	test1(c1, 200);
	test1(c2, 300);
	test1(c3, 400);
	wait();

	mkchan(c);
	test2();
	wait();

	mkchan(c);
	test3();
	wait();
}

// run all test with 4 buffser sizes
func
main()
{
	tests(0);
	tests(1);
	tests(10);
	tests(100);

	if tots != totr || tots != 3648 {
		print("tots=", tots, " totr=", totr, "\n");
		sys.exit(1);
	}
	sys.exit(0);
}
