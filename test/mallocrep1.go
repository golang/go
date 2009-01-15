// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Repeated malloc test.

package main

import (
	"flag";
	"fmt";
	"malloc";
	"strconv"
)

var chatty = flag.Bool("v", false, "chatty");
var reverse = flag.Bool("r", false, "reverse");
var longtest = flag.Bool("l", false, "long test");

var b []*byte;
var stats = malloc.GetStats();

func OkAmount(size, n uint64) bool {
	if n < size {
		return false
	}
	if size < 16*8 {
		if n > size+16 {
			return false
		}
	} else {
		if n > size*9/8 {
			return false
		}
	}
	return true
}

func AllocAndFree(size, count int) {
	if *chatty {
		fmt.Printf("size=%d count=%d ...\n", size, count);
	}
	n1 := stats.alloc;
	for i := 0; i < count; i++ {
		b[i] = malloc.Alloc(uint64(size));
		base, n := malloc.Lookup(b[i]);
		if base != b[i] || !OkAmount(uint64(size), n) {
			panicln("lookup failed: got", base, n, "for", b[i]);
		}
		if malloc.GetStats().sys > 1e9 {
			panicln("too much memory allocated");
		}
	}
	n2 := stats.alloc;
	if *chatty {
		fmt.Printf("size=%d count=%d stats=%+v\n", size, count, *stats);
	}
	n3 := stats.alloc;
	for j := 0; j < count; j++ {
		i := j;
		if *reverse {
			i = count - 1 - j;
		}
		alloc := stats.alloc;
		base, n := malloc.Lookup(b[i]);
		if base != b[i] || !OkAmount(uint64(size), n) {
			panicln("lookup failed: got", base, n, "for", b[i]);
		}
		malloc.Free(b[i]);
		if stats.alloc != alloc - n {
			panicln("free alloc got", stats.alloc, "expected", alloc - n, "after free of", n);
		}
		if malloc.GetStats().sys > 1e9 {
			panicln("too much memory allocated");
		}
	}
	n4 := stats.alloc;

	if *chatty {
		fmt.Printf("size=%d count=%d stats=%+v\n", size, count, *stats);
	}
	if n2-n1 != n3-n4 {
		panicln("wrong alloc count: ", n2-n1, n3-n4);
	}
}

func atoi(s string) int {
	i, xx1 := strconv.atoi(s);
	return i
}

func main() {
	flag.Parse();
	b = make([]*byte, 10000);
	if flag.NArg() > 0 {
		AllocAndFree(atoi(flag.Arg(0)), atoi(flag.Arg(1)));
		return;
	}
	for j := 1; j <= 1<<22; j<<=1 {
		n := len(b);
		max := uint64(1<<28);
		if !*longtest {
			max = 1<<22;
		}
		if uint64(j)*uint64(n) > max {
			n = int(max / uint64(j));
		}
		if n < 10 {
			n = 10;
		}
		for m := 1; m <= n; {
			AllocAndFree(j, m);
			if m == n {
				break
			}
			m = 5*m/4;
			if m < 4 {
				m++
			}
			if m > n {
				m = n
			}
		}
	}
}
