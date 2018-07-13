// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bytes"
	"flag"
	"internal/race"
	"internal/trace"
	"io"
	"io/ioutil"
	"net"
	"os"
	"runtime"
	. "runtime/trace"
	"strconv"
	"sync"
	"testing"
	"time"
)

var (
	saveTraces = flag.Bool("savetraces", false, "save traces collected by tests")
)

// TestEventBatch tests Flush calls that happen during Start
// don't produce corrupted traces.
func TestEventBatch(t *testing.T) {
	if race.Enabled {
		t.Skip("skipping in race mode")
	}
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	// During Start, bunch of records are written to reflect the current
	// snapshot of the program, including state of each goroutines.
	// And some string constants are written to the trace to aid trace
	// parsing. This test checks Flush of the buffer occurred during
	// this process doesn't cause corrupted traces.
	// When a Flush is called during Start is complicated
	// so we test with a range of number of goroutines hoping that one
	// of them triggers Flush.
	// This range was chosen to fill up a ~64KB buffer with traceEvGoCreate
	// and traceEvGoWaiting events (12~13bytes per goroutine).
	for g := 4950; g < 5050; g++ {
		n := g
		t.Run("G="+strconv.Itoa(n), func(t *testing.T) {
			var wg sync.WaitGroup
			wg.Add(n)

			in := make(chan bool, 1000)
			for i := 0; i < n; i++ {
				go func() {
					<-in
					wg.Done()
				}()
			}
			buf := new(bytes.Buffer)
			if err := Start(buf); err != nil {
				t.Fatalf("failed to start tracing: %v", err)
			}

			for i := 0; i < n; i++ {
				in <- true
			}
			wg.Wait()
			Stop()

			_, err := trace.Parse(buf, "")
			if err == trace.ErrTimeOrder {
				t.Skipf("skipping trace: %v", err)
			}

			if err != nil {
				t.Fatalf("failed to parse trace: %v", err)
			}
		})
	}
}

func TestTraceStartStop(t *testing.T) {
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
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
	saveTrace(t, buf, "TestTraceStartStop")
}

func TestTraceDoubleStart(t *testing.T) {
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
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
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}
	Stop()
	saveTrace(t, buf, "TestTrace")
	_, err := trace.Parse(buf, "")
	if err == trace.ErrTimeOrder {
		t.Skipf("skipping trace: %v", err)
	}
	if err != nil {
		t.Fatalf("failed to parse trace: %v", err)
	}
}

func parseTrace(t *testing.T, r io.Reader) ([]*trace.Event, map[uint64]*trace.GDesc) {
	res, err := trace.Parse(r, "")
	if err == trace.ErrTimeOrder {
		t.Skipf("skipping trace: %v", err)
	}
	if err != nil {
		t.Fatalf("failed to parse trace: %v", err)
	}
	gs := trace.GoroutineStats(res.Events)
	for goid := range gs {
		// We don't do any particular checks on the result at the moment.
		// But still check that RelatedGoroutines does not crash, hang, etc.
		_ = trace.RelatedGoroutines(res.Events, goid)
	}
	return res.Events, gs
}

func testBrokenTimestamps(t *testing.T, data []byte) {
	// On some processors cputicks (used to generate trace timestamps)
	// produce non-monotonic timestamps. It is important that the parser
	// distinguishes logically inconsistent traces (e.g. missing, excessive
	// or misordered events) from broken timestamps. The former is a bug
	// in tracer, the latter is a machine issue.
	// So now that we have a consistent trace, test that (1) parser does
	// not return a logical error in case of broken timestamps
	// and (2) broken timestamps are eventually detected and reported.
	trace.BreakTimestampsForTesting = true
	defer func() {
		trace.BreakTimestampsForTesting = false
	}()
	for i := 0; i < 1e4; i++ {
		_, err := trace.Parse(bytes.NewReader(data), "")
		if err == trace.ErrTimeOrder {
			return
		}
		if err != nil {
			t.Fatalf("failed to parse trace: %v", err)
		}
	}
}

func TestTraceStress(t *testing.T) {
	if runtime.GOOS == "js" {
		t.Skip("no os.Pipe on js")
	}
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
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
	saveTrace(t, buf, "TestTraceStress")
	trace := buf.Bytes()
	parseTrace(t, buf)
	testBrokenTimestamps(t, trace)
}

// Do a bunch of various stuff (timers, GC, network, etc) in a separate goroutine.
// And concurrently with all that start/stop trace 3 times.
func TestTraceStressStartStop(t *testing.T) {
	if runtime.GOOS == "js" {
		t.Skip("no os.Pipe on js")
	}
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
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
			t.Errorf("failed to create pipe: %v", err)
			return
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
			t.Errorf("listen failed: %v", err)
			return
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
			t.Errorf("dial failed: %v", err)
			return
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
		saveTrace(t, buf, "TestTraceStressStartStop")
		trace := buf.Bytes()
		parseTrace(t, buf)
		testBrokenTimestamps(t, trace)
	}
	<-outerDone
}

func TestTraceFutileWakeup(t *testing.T) {
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
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
	saveTrace(t, buf, "TestTraceFutileWakeup")
	events, _ := parseTrace(t, buf)
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

func saveTrace(t *testing.T, buf *bytes.Buffer, name string) {
	if !*saveTraces {
		return
	}
	if err := ioutil.WriteFile(name+".trace", buf.Bytes(), 0600); err != nil {
		t.Errorf("failed to write trace file: %s", err)
	}
}
