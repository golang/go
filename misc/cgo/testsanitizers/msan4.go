// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// The memory profiler can call copy from a slice on the system stack,
// which msan used to think meant a reference to uninitialized memory.

/*
#include <time.h>
#include <unistd.h>

extern void Nop(char*);

// Use weak as a hack to permit defining a function even though we use export.
void poison() __attribute__ ((weak));

// Poison the stack.
void poison() {
	char a[1024];
	Nop(&a[0]);
}

*/
import "C"

import (
	"runtime"
)

func main() {
	runtime.MemProfileRate = 1
	start(100)
}

func start(i int) {
	if i == 0 {
		return
	}
	C.poison()
	// Tie up a thread.
	// We won't actually wait for this sleep to complete.
	go func() { C.sleep(1) }()
	start(i - 1)
}

//export Nop
func Nop(*C.char) {
}
