// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bytes"
	"internal/trace"
	"io"
	"net"
	"os"
	"runtime"
	. "runtime/trace"
	"sync"
	"testing"
	"time"
)

func TestTraceStartStop(t *testing.T) {
	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}
	Stop()
	size := buf.Len()
	if size == 0 {
		t.Fatalf("trace is empty")
	}
	time.Sleep(100 * time.Millisecond)
	if size != buf.Len() {
		t.Fatalf("trace writes after stop: %v -> %v", size, buf.Len())
	}
}

func TestTraceDoubleStart(t *testing.T) {
	Stop()
	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}
	if err := Start(buf); err == nil {
		t.Fatalf("succeed to start tracing second time")
	}
	Stop()
	Stop()
}

func TestTrace(t *testing.T) {
	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}
	Stop()
	_, err := trace.Parse(buf)
	if err == trace.ErrTimeOrder {
		t.Skipf("skipping trace: %v", err)
	}
	if err != nil {
		t.Fatalf("failed to parse trace: %v", err)
	}
}

func parseTrace(t *testing.T, r io.Reader) ([]*trace.Event, map[uint64]*trace.GDesc, error) {
	events, err := trace.Parse(r)
	if err == trace.ErrTimeOrder {
		t.Skipf("skipping trace: %v", err)
	}
	if err != nil {
		return nil, nil, err
	}
	gs := trace.GoroutineStats(events)
	for goid := range gs {
		// We don't do any particular checks on the result at the moment.
		// But still check that RelatedGoroutines does not crash, hang, etc.
		_ = trace.RelatedGoroutines(events, goid)
	}
	return events, gs, nil
}

func TestTraceStress(t *testing.T) {
	var wg sync.WaitGroup
	done := make(chan bool)

	// Create a goroutine blocked before tracing.
	wg.Add(1)
	go func() {
		<-done
		wg.Done()
	}()

	// Create a goroutine blocked in syscall before tracing.
	rp, wp, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create pipe: %v", err)
	}
	defer func() {
		rp.Close()
		wp.Close()
	}()
	wg.Add(1)
	go func() {
		var tmp [1]byte
		rp.Read(tmp[:])
		<-done
		wg.Done()
	}()
	time.Sleep(time.Millisecond) // give the goroutine above time to block

	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}

	procs := runtime.GOMAXPROCS(10)
	time.Sleep(50 * time.Millisecond) // test proc stop/start events

	go func() {
		runtime.LockOSThread()
		for {
			select {
			case <-done:
				return
			default:
				runtime.Gosched()
			}
		}
	}()

	runtime.GC()
	// Trigger GC from malloc.
	n := int(1e3)
	if runtime.GOOS == "openbsd" && runtime.GOARCH == "arm" {
		// Reduce allocation to avoid running out of
		// memory on the builder - see issue/12032.
		n = 512
	}
	for i := 0; i < n; i++ {
		_ = make([]byte, 1<<20)
	}

	// Create a bunch of busy goroutines to load all Ps.
	for p := 0; p < 10; p++ {
		wg.Add(1)
		go func() {
			// Do something useful.
			tmp := make([]byte, 1<<16)
			for i := range tmp {
				tmp[i]++
			}
			_ = tmp
			<-done
			wg.Done()
		}()
	}

	// Block in syscall.
	wg.Add(1)
	go func() {
		var tmp [1]byte
		rp.Read(tmp[:])
		<-done
		wg.Done()
	}()

	// Test timers.
	timerDone := make(chan bool)
	go func() {
		time.Sleep(time.Millisecond)
		timerDone <- true
	}()
	<-timerDone

	// A bit of network.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen failed: %v", err)
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
		t.Fatalf("dial failed: %v", err)
	}
	var tmp [1]byte
	c.Read(tmp[:])
	c.Close()

	go func() {
		runtime.Gosched()
		select {}
	}()

	// Unblock helper goroutines and wait them to finish.
	wp.Write(tmp[:])
	wp.Write(tmp[:])
	close(done)
	wg.Wait()

	runtime.GOMAXPROCS(procs)

	Stop()
	_, _, err = parseTrace(t, buf)
	if err != nil {
		t.Fatalf("failed to parse trace: %v", err)
	}
}

// Do a bunch of various stuff (timers, GC, network, etc) in a separate goroutine.
// And concurrently with all that start/stop trace 3 times.
func TestTraceStressStartStop(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(8))
	outerDone := make(chan bool)

	go func() {
		defer func() {
			outerDone <- true
		}()

		var wg sync.WaitGroup
		done := make(chan bool)

		wg.Add(1)
		go func() {
			<-done
			wg.Done()
		}()

		rp, wp, err := os.Pipe()
		if err != nil {
			t.Fatalf("failed to create pipe: %v", err)
		}
		defer func() {
			rp.Close()
			wp.Close()
		}()
		wg.Add(1)
		go func() {
			var tmp [1]byte
			rp.Read(tmp[:])
			<-done
			wg.Done()
		}()
		time.Sleep(time.Millisecond)

		go func() {
			runtime.LockOSThread()
			for {
				select {
				case <-done:
					return
				default:
					runtime.Gosched()
				}
			}
		}()

		runtime.GC()
		// Trigger GC from malloc.
		n := int(1e3)
		if runtime.GOOS == "openbsd" && runtime.GOARCH == "arm" {
			// Reduce allocation to avoid running out of
			// memory on the builder - see issue/12032.
			n = 512
		}
		for i := 0; i < n; i++ {
			_ = make([]byte, 1<<20)
		}

		// Create a bunch of busy goroutines to load all Ps.
		for p := 0; p < 10; p++ {
			wg.Add(1)
			go func() {
				// Do something useful.
				tmp := make([]byte, 1<<16)
				for i := range tmp {
					tmp[i]++
				}
				_ = tmp
				<-done
				wg.Done()
			}()
		}

		// Block in syscall.
		wg.Add(1)
		go func() {
			var tmp [1]byte
			rp.Read(tmp[:])
			<-done
			wg.Done()
		}()

		runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

		// Test timers.
		timerDone := make(chan bool)
		go func() {
			time.Sleep(time.Millisecond)
			timerDone <- true
		}()
		<-timerDone

		// A bit of network.
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen failed: %v", err)
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
			t.Fatalf("dial failed: %v", err)
		}
		var tmp [1]byte
		c.Read(tmp[:])
		c.Close()

		go func() {
			runtime.Gosched()
			select {}
		}()

		// Unblock helper goroutines and wait them to finish.
		wp.Write(tmp[:])
		wp.Write(tmp[:])
		close(done)
		wg.Wait()
	}()

	for i := 0; i < 3; i++ {
		buf := new(bytes.Buffer)
		if err := Start(buf); err != nil {
			t.Fatalf("failed to start tracing: %v", err)
		}
		time.Sleep(time.Millisecond)
		Stop()
		if _, _, err := parseTrace(t, buf); err != nil {
			t.Fatalf("failed to parse trace: %v", err)
		}
	}
	<-outerDone
}

func TestTraceFutileWakeup(t *testing.T) {
	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}

	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(8))
	c0 := make(chan int, 1)
	c1 := make(chan int, 1)
	c2 := make(chan int, 1)
	const procs = 2
	var done sync.WaitGroup
	done.Add(4 * procs)
	for p := 0; p < procs; p++ {
		const iters = 1e3
		go func() {
			for i := 0; i < iters; i++ {
				runtime.Gosched()
				c0 <- 0
			}
			done.Done()
		}()
		go func() {
			for i := 0; i < iters; i++ {
				runtime.Gosched()
				<-c0
			}
			done.Done()
		}()
		go func() {
			for i := 0; i < iters; i++ {
				runtime.Gosched()
				select {
				case c1 <- 0:
				case c2 <- 0:
				}
			}
			done.Done()
		}()
		go func() {
			for i := 0; i < iters; i++ {
				runtime.Gosched()
				select {
				case <-c1:
				case <-c2:
				}
			}
			done.Done()
		}()
	}
	done.Wait()

	Stop()
	events, _, err := parseTrace(t, buf)
	if err != nil {
		t.Fatalf("failed to parse trace: %v", err)
	}
	// Check that (1) trace does not contain EvFutileWakeup events and
	// (2) there are no consecutive EvGoBlock/EvGCStart/EvGoBlock events
	// (we call runtime.Gosched between all operations, so these would be futile wakeups).
	gs := make(map[uint64]int)
	for _, ev := range events {
		switch ev.Type {
		case trace.EvFutileWakeup:
			t.Fatalf("found EvFutileWakeup event")
		case trace.EvGoBlockSend, trace.EvGoBlockRecv, trace.EvGoBlockSelect:
			if gs[ev.G] == 2 {
				t.Fatalf("goroutine %v blocked on %v at %v right after start",
					ev.G, trace.EventDescriptions[ev.Type].Name, ev.Ts)
			}
			if gs[ev.G] == 1 {
				t.Fatalf("goroutine %v blocked on %v at %v while blocked",
					ev.G, trace.EventDescriptions[ev.Type].Name, ev.Ts)
			}
			gs[ev.G] = 1
		case trace.EvGoStart:
			if gs[ev.G] == 1 {
				gs[ev.G] = 2
			}
		default:
			delete(gs, ev.G)
		}
	}
}
