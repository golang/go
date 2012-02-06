// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// trivial malloc test

package main

import (
	"flag"
	"fmt"
	"runtime"
)

var chatty = flag.Bool("v", false, "chatty")

func main() {
	memstats := new(runtime.MemStats)
	runtime.Free(runtime.Alloc(1))
	runtime.ReadMemStats(memstats)
	if *chatty {
		fmt.Printf("%+v %v\n", memstats, uint64(0))
	}
}
