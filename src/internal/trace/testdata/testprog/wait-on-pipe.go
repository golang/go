// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests a goroutine sitting blocked in a syscall for
// an entire generation. This is a regression test for
// #65196.

//go:build ignore

package main

import (
	"log"
	"os"
	"runtime/trace"
	"syscall"
	"time"
)

func main() {
	// Create a pipe to block on.
	var p [2]int
	if err := syscall.Pipe(p[:]); err != nil {
		log.Fatalf("failed to create pipe: %v", err)
	}
	rfd, wfd := p[0], p[1]

	// Create a goroutine that blocks on the pipe.
	done := make(chan struct{})
	go func() {
		var data [1]byte
		_, err := syscall.Read(rfd, data[:])
		if err != nil {
			log.Fatalf("failed to read from pipe: %v", err)
		}
		done <- struct{}{}
	}()

	// Give the goroutine ample chance to block on the pipe.
	time.Sleep(10 * time.Millisecond)

	// Start tracing.
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}

	// This isn't enough to have a full generation pass by default,
	// but it is generally enough in stress mode.
	time.Sleep(100 * time.Millisecond)

	// Write to the pipe to unblock it.
	if _, err := syscall.Write(wfd, []byte{10}); err != nil {
		log.Fatalf("failed to write to pipe: %v", err)
	}

	// Wait for the goroutine to unblock and start running.
	// This is helpful to catch incorrect information written
	// down for the syscall-blocked goroutine, since it'll start
	// executing, and that execution information will be
	// inconsistent.
	<-done

	// Stop tracing.
	trace.Stop()
}
