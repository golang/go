// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
)

var finalizerDeadlockMode = flag.String("finalizer-deadlock-mode", "panic", "Trigger mode of FinalizerDeadlock")

func init() {
	register("FinalizerDeadlock", FinalizerDeadlock)
}

func FinalizerDeadlock() {
	flag.Parse()

	started := make(chan struct{})
	b := new([16]byte)
	runtime.SetFinalizer(b, func(*[16]byte) {
		started <- struct{}{}
		select {}
	})
	b = nil

	runtime.GC()

	<-started
	// We know the finalizer has started running. The goroutine might still
	// be running or it may now be blocked. Either is fine, the goroutine
	// should appear in stacks either way.

	mode := os.Getenv("GO_TEST_FINALIZER_DEADLOCK")
	switch mode {
	case "panic":
		panic("panic")
	case "stack":
		buf := make([]byte, 4096)
		for {
			n := runtime.Stack(buf, true)
			if n >= len(buf) {
				buf = make([]byte, 2*len(buf))
				continue
			}
			buf = buf[:n]
			break
		}
		fmt.Printf("%s\n", string(buf))
	case "pprof_proto":
		if err := pprof.Lookup("goroutine").WriteTo(os.Stdout, 0); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing profile: %v\n", err)
			os.Exit(1)
		}
	case "pprof_debug1":
		if err := pprof.Lookup("goroutine").WriteTo(os.Stdout, 1); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing profile: %v\n", err)
			os.Exit(1)
		}
	case "pprof_debug2":
		if err := pprof.Lookup("goroutine").WriteTo(os.Stdout, 2); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing profile: %v\n", err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown mode %q. GO_TEST_FINALIZER_DEADLOCK must be one of panic, stack, pprof_proto, pprof_debug1, pprof_debug2\n", mode)
		os.Exit(1)
	}
}
