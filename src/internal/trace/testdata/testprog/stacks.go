// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests stack symbolization.

//go:build ignore

package main

import (
	"log"
	"net"
	"os"
	"runtime"
	"runtime/trace"
	"sync"
	"time"
)

func main() {
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}
	defer trace.Stop() // in case of early return

	// Now we will do a bunch of things for which we verify stacks later.
	// It is impossible to ensure that a goroutine has actually blocked
	// on a channel, in a select or otherwise. So we kick off goroutines
	// that need to block first in the hope that while we are executing
	// the rest of the test, they will block.
	go func() { // func1
		select {}
	}()
	go func() { // func2
		var c chan int
		c <- 0
	}()
	go func() { // func3
		var c chan int
		<-c
	}()
	done1 := make(chan bool)
	go func() { // func4
		<-done1
	}()
	done2 := make(chan bool)
	go func() { // func5
		done2 <- true
	}()
	c1 := make(chan int)
	c2 := make(chan int)
	go func() { // func6
		select {
		case <-c1:
		case <-c2:
		}
	}()
	var mu sync.Mutex
	mu.Lock()
	go func() { // func7
		mu.Lock()
		mu.Unlock()
	}()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { // func8
		wg.Wait()
	}()
	cv := sync.NewCond(&sync.Mutex{})
	go func() { // func9
		cv.L.Lock()
		cv.Wait()
		cv.L.Unlock()
	}()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	go func() { // func10
		c, err := ln.Accept()
		if err != nil {
			log.Printf("failed to accept: %v", err)
			return
		}
		c.Close()
	}()
	rp, wp, err := os.Pipe()
	if err != nil {
		log.Fatalf("failed to create a pipe: %v", err)
	}
	defer rp.Close()
	defer wp.Close()
	pipeReadDone := make(chan bool)
	go func() { // func11
		var data [1]byte
		rp.Read(data[:])
		pipeReadDone <- true
	}()
	go func() { // func12
		for {
			syncPreemptPoint()
		}
	}()

	time.Sleep(100 * time.Millisecond)
	runtime.GC()
	runtime.Gosched()
	time.Sleep(100 * time.Millisecond) // the last chance for the goroutines above to block
	done1 <- true
	<-done2
	select {
	case c1 <- 0:
	case c2 <- 0:
	}
	mu.Unlock()
	wg.Done()
	cv.Signal()
	c, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		log.Fatalf("failed to dial: %v", err)
	}
	c.Close()
	var data [1]byte
	wp.Write(data[:])
	<-pipeReadDone

	oldGoMaxProcs := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(oldGoMaxProcs + 1)

	trace.Stop()

	runtime.GOMAXPROCS(oldGoMaxProcs)
}

//go:noinline
func syncPreemptPoint() {
	if never {
		syncPreemptPoint()
	}
}

var never bool
