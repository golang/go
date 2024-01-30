// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"runtime"
	"runtime/trace"
	"sync"
	"syscall"
	"time"
)

func main() {
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatal(err)
	}

	// checkExecutionTimes relies on this.
	var wg sync.WaitGroup
	wg.Add(2)
	go cpu10(&wg)
	go cpu20(&wg)
	wg.Wait()

	// checkHeapMetrics relies on this.
	allocHog(25 * time.Millisecond)

	// checkProcStartStop relies on this.
	var wg2 sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg2.Add(1)
		go func() {
			defer wg2.Done()
			cpuHog(50 * time.Millisecond)
		}()
	}
	wg2.Wait()

	// checkSyscalls relies on this.
	done := make(chan error)
	go blockingSyscall(50*time.Millisecond, done)
	if err := <-done; err != nil {
		log.Fatal(err)
	}

	// checkNetworkUnblock relies on this.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatalf("listen failed: %v", err)
	}
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		time.Sleep(time.Millisecond)
		var buf [1]byte
		c.Write(buf[:])
		c.Close()
	}()
	c, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		log.Fatalf("dial failed: %v", err)
	}
	var tmp [1]byte
	c.Read(tmp[:])
	c.Close()

	trace.Stop()
}

// blockingSyscall blocks the current goroutine for duration d in a syscall and
// sends a message to done when it is done or if the syscall failed.
func blockingSyscall(d time.Duration, done chan<- error) {
	r, w, err := os.Pipe()
	if err != nil {
		done <- err
		return
	}
	start := time.Now()
	msg := []byte("hello")
	time.AfterFunc(d, func() { w.Write(msg) })
	_, err = syscall.Read(int(r.Fd()), make([]byte, len(msg)))
	if err == nil && time.Since(start) < d {
		err = fmt.Errorf("syscall returned too early: want=%s got=%s", d, time.Since(start))
	}
	done <- err
}

func cpu10(wg *sync.WaitGroup) {
	defer wg.Done()
	cpuHog(10 * time.Millisecond)
}

func cpu20(wg *sync.WaitGroup) {
	defer wg.Done()
	cpuHog(20 * time.Millisecond)
}

func cpuHog(dt time.Duration) {
	start := time.Now()
	for i := 0; ; i++ {
		if i%1000 == 0 && time.Since(start) > dt {
			return
		}
	}
}

func allocHog(dt time.Duration) {
	start := time.Now()
	var s [][]byte
	for i := 0; ; i++ {
		if i%1000 == 0 {
			if time.Since(start) > dt {
				return
			}
			// Take a break... this will generate a ton of events otherwise.
			time.Sleep(50 * time.Microsecond)
		}
		s = append(s, make([]byte, 1024))
	}
}
