// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Repeated malloc test.

package main

import (
	"flag"
	"fmt"
	"runtime"
	"strconv"
)

var chatty = flag.Bool("v", false, "chatty")
var reverse = flag.Bool("r", false, "reverse")
var longtest = flag.Bool("l", false, "long test")

var b []*byte
var stats = new(runtime.MemStats)

func OkAmount(size, n uintptr) bool {
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
		fmt.Printf("size=%d count=%d ...\n", size, count)
	}
	runtime.ReadMemStats(stats)
	n1 := stats.Alloc
	for i := 0; i < count; i++ {
		b[i] = runtime.Alloc(uintptr(size))
		base, n := runtime.Lookup(b[i])
		if base != b[i] || !OkAmount(uintptr(size), n) {
			println("lookup failed: got", base, n, "for", b[i])
			panic("fail")
		}
		runtime.ReadMemStats(stats)
		if stats.Sys > 1e9 {
			println("too much memory allocated")
			panic("fail")
		}
	}
	runtime.ReadMemStats(stats)
	n2 := stats.Alloc
	if *chatty {
		fmt.Printf("size=%d count=%d stats=%+v\n", size, count, *stats)
	}
	n3 := stats.Alloc
	for j := 0; j < count; j++ {
		i := j
		if *reverse {
			i = count - 1 - j
		}
		alloc := uintptr(stats.Alloc)
		base, n := runtime.Lookup(b[i])
		if base != b[i] || !OkAmount(uintptr(size), n) {
			println("lookup failed: got", base, n, "for", b[i])
			panic("fail")
		}
		runtime.Free(b[i])
		runtime.ReadMemStats(stats)
		if stats.Alloc != uint64(alloc-n) {
			println("free alloc got", stats.Alloc, "expected", alloc-n, "after free of", n)
			panic("fail")
		}
		if stats.Sys > 1e9 {
			println("too much memory allocated")
			panic("fail")
		}
	}
	runtime.ReadMemStats(stats)
	n4 := stats.Alloc

	if *chatty {
		fmt.Printf("size=%d count=%d stats=%+v\n", size, count, *stats)
	}
	if n2-n1 != n3-n4 {
		println("wrong alloc count: ", n2-n1, n3-n4)
		panic("fail")
	}
}

func atoi(s string) int {
	i, _ := strconv.Atoi(s)
	return i
}

func main() {
	runtime.MemProfileRate = 0 // disable profiler
	flag.Parse()
	b = make([]*byte, 10000)
	if flag.NArg() > 0 {
		AllocAndFree(atoi(flag.Arg(0)), atoi(flag.Arg(1)))
		return
	}
	maxb := 1 << 22
	if !*longtest {
		maxb = 1 << 19
	}
	for j := 1; j <= maxb; j <<= 1 {
		n := len(b)
		max := uintptr(1 << 28)
		if !*longtest {
			max = uintptr(maxb)
		}
		if uintptr(j)*uintptr(n) > max {
			n = int(max / uintptr(j))
		}
		if n < 10 {
			n = 10
		}
		for m := 1; m <= n; {
			AllocAndFree(j, m)
			if m == n {
				break
			}
			m = 5 * m / 4
			if m < 4 {
				m++
			}
			if m > n {
				m = n
			}
		}
	}
}
