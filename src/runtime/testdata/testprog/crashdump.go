// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime"
)

func init() {
	register("CrashDumpsAllThreads", CrashDumpsAllThreads)
}

func CrashDumpsAllThreads() {
	const count = 4
	runtime.GOMAXPROCS(count + 1)

	chans := make([]chan bool, count)
	for i := range chans {
		chans[i] = make(chan bool)
		go crashDumpsAllThreadsLoop(i, chans[i])
	}

	// Wait for all the goroutines to start executing.
	for _, c := range chans {
		<-c
	}

	// Tell our parent that all the goroutines are executing.
	if _, err := os.NewFile(3, "pipe").WriteString("x"); err != nil {
		fmt.Fprintf(os.Stderr, "write to pipe failed: %v\n", err)
		os.Exit(2)
	}

	select {}
}

func crashDumpsAllThreadsLoop(i int, c chan bool) {
	close(c)
	for {
		for j := 0; j < 0x7fffffff; j++ {
		}
	}
}
