// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"internal/goexperiment"
	"internal/profile"
	"internal/race"
	"internal/trace"
	"io"
	"net"
	"os"
	"runtime"
	"runtime/pprof"
	. "runtime/trace"
	"strconv"
	"strings"
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
	if goexperiment.ExecTracer2 {
		t.Skip("skipping because this test is incompatible with the new tracer")
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
	if goexperiment.ExecTracer2 {
		// An equivalent test exists in internal/trace/v2.
		t.Skip("skipping because this test is incompatible with the new tracer")
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
	switch runtime.GOOS {
	case "js", "wasip1":
		t.Skip("no os.Pipe on " + runtime.GOOS)
	}
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	if goexperiment.ExecTracer2 {
		// An equivalent test exists in internal/trace/v2.
		t.Skip("skipping because this test is incompatible with the new tracer")
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
	if isMemoryConstrained() {
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

// isMemoryConstrained reports whether the current machine is likely
// to be memory constrained.
// This was originally for the openbsd/arm builder (Issue 12032).
// TODO: move this to testenv? Make this look at memory? Look at GO_BUILDER_NAME?
func isMemoryConstrained() bool {
	if runtime.GOOS == "plan9" {
		return true
	}
	switch runtime.GOARCH {
	case "arm", "mips", "mipsle":
		return true
	}
	return false
}

// Do a bunch of various stuff (timers, GC, network, etc) in a separate goroutine.
// And concurrently with all that start/stop trace 3 times.
func TestTraceStressStartStop(t *testing.T) {
	switch runtime.GOOS {
	case "js", "wasip1":
		t.Skip("no os.Pipe on " + runtime.GOOS)
	}
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
	if goexperiment.ExecTracer2 {
		// An equivalent test exists in internal/trace/v2.
		t.Skip("skipping because this test is incompatible with the new tracer")
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
		if isMemoryConstrained() {
			// Reduce allocation to avoid running out of
			// memory on the builder.
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
	if goexperiment.ExecTracer2 {
		t.Skip("skipping because this test is incompatible with the new tracer")
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

func TestTraceCPUProfile(t *testing.T) {
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
	if goexperiment.ExecTracer2 {
		// An equivalent test exists in internal/trace/v2.
		t.Skip("skipping because this test is incompatible with the new tracer")
	}

	cpuBuf := new(bytes.Buffer)
	if err := pprof.StartCPUProfile(cpuBuf); err != nil {
		t.Skipf("failed to start CPU profile: %v", err)
	}

	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}

	dur := 100 * time.Millisecond
	func() {
		// Create a region in the execution trace. Set and clear goroutine
		// labels fully within that region, so we know that any CPU profile
		// sample with the label must also be eligible for inclusion in the
		// execution trace.
		ctx := context.Background()
		defer StartRegion(ctx, "cpuHogger").End()
		pprof.Do(ctx, pprof.Labels("tracing", "on"), func(ctx context.Context) {
			cpuHogger(cpuHog1, &salt1, dur)
		})
		// Be sure the execution trace's view, when filtered to this goroutine
		// via the explicit goroutine ID in each event, gets many more samples
		// than the CPU profiler when filtered to this goroutine via labels.
		cpuHogger(cpuHog1, &salt1, dur)
	}()

	Stop()
	pprof.StopCPUProfile()
	saveTrace(t, buf, "TestTraceCPUProfile")

	prof, err := profile.Parse(cpuBuf)
	if err != nil {
		t.Fatalf("failed to parse CPU profile: %v", err)
	}
	// Examine the CPU profiler's view. Filter it to only include samples from
	// the single test goroutine. Use labels to execute that filter: they should
	// apply to all work done while that goroutine is getg().m.curg, and they
	// should apply to no other goroutines.
	pprofSamples := 0
	pprofStacks := make(map[string]int)
	for _, s := range prof.Sample {
		if s.Label["tracing"] != nil {
			var fns []string
			var leaf string
			for _, loc := range s.Location {
				for _, line := range loc.Line {
					fns = append(fns, fmt.Sprintf("%s:%d", line.Function.Name, line.Line))
					leaf = line.Function.Name
				}
			}
			// runtime.sigprof synthesizes call stacks when "normal traceback is
			// impossible or has failed", using particular placeholder functions
			// to represent common failure cases. Look for those functions in
			// the leaf position as a sign that the call stack and its
			// symbolization are more complex than this test can handle.
			//
			// TODO: Make the symbolization done by the execution tracer and CPU
			// profiler match up even in these harder cases. See #53378.
			switch leaf {
			case "runtime._System", "runtime._GC", "runtime._ExternalCode", "runtime._VDSO":
				continue
			}
			stack := strings.Join(fns, " ")
			samples := int(s.Value[0])
			pprofSamples += samples
			pprofStacks[stack] += samples
		}
	}
	if pprofSamples == 0 {
		t.Skipf("CPU profile did not include any samples while tracing was active\n%s", prof)
	}

	// Examine the execution tracer's view of the CPU profile samples. Filter it
	// to only include samples from the single test goroutine. Use the goroutine
	// ID that was recorded in the events: that should reflect getg().m.curg,
	// same as the profiler's labels (even when the M is using its g0 stack).
	totalTraceSamples := 0
	traceSamples := 0
	traceStacks := make(map[string]int)
	events, _ := parseTrace(t, buf)
	var hogRegion *trace.Event
	for _, ev := range events {
		if ev.Type == trace.EvUserRegion && ev.Args[1] == 0 && ev.SArgs[0] == "cpuHogger" {
			// mode "0" indicates region start
			hogRegion = ev
		}
	}
	if hogRegion == nil {
		t.Fatalf("execution trace did not identify cpuHogger goroutine")
	} else if hogRegion.Link == nil {
		t.Fatalf("execution trace did not close cpuHogger region")
	}
	for _, ev := range events {
		if ev.Type == trace.EvCPUSample {
			totalTraceSamples++
			if ev.G == hogRegion.G {
				traceSamples++
				var fns []string
				for _, frame := range ev.Stk {
					if frame.Fn != "runtime.goexit" {
						fns = append(fns, fmt.Sprintf("%s:%d", frame.Fn, frame.Line))
					}
				}
				stack := strings.Join(fns, " ")
				traceStacks[stack]++
			}
		}
	}

	// The execution trace may drop CPU profile samples if the profiling buffer
	// overflows. Based on the size of profBufWordCount, that takes a bit over
	// 1900 CPU samples or 19 thread-seconds at a 100 Hz sample rate. If we've
	// hit that case, then we definitely have at least one full buffer's worth
	// of CPU samples, so we'll call that success.
	overflowed := totalTraceSamples >= 1900
	if traceSamples < pprofSamples {
		t.Logf("execution trace did not include all CPU profile samples; %d in profile, %d in trace", pprofSamples, traceSamples)
		if !overflowed {
			t.Fail()
		}
	}

	for stack, traceSamples := range traceStacks {
		pprofSamples := pprofStacks[stack]
		delete(pprofStacks, stack)
		if traceSamples < pprofSamples {
			t.Logf("execution trace did not include all CPU profile samples for stack %q; %d in profile, %d in trace",
				stack, pprofSamples, traceSamples)
			if !overflowed {
				t.Fail()
			}
		}
	}
	for stack, pprofSamples := range pprofStacks {
		t.Logf("CPU profile included %d samples at stack %q not present in execution trace", pprofSamples, stack)
		if !overflowed {
			t.Fail()
		}
	}

	if t.Failed() {
		t.Logf("execution trace CPU samples:")
		for stack, samples := range traceStacks {
			t.Logf("%d: %q", samples, stack)
		}
		t.Logf("CPU profile:\n%v", prof)
	}
}

func cpuHogger(f func(x int) int, y *int, dur time.Duration) {
	// We only need to get one 100 Hz clock tick, so we've got
	// a large safety buffer.
	// But do at least 500 iterations (which should take about 100ms),
	// otherwise TestCPUProfileMultithreaded can fail if only one
	// thread is scheduled during the testing period.
	t0 := time.Now()
	accum := *y
	for i := 0; i < 500 || time.Since(t0) < dur; i++ {
		accum = f(accum)
	}
	*y = accum
}

var (
	salt1 = 0
)

// The actual CPU hogging function.
// Must not call other functions nor access heap/globals in the loop,
// otherwise under race detector the samples will be in the race runtime.
func cpuHog1(x int) int {
	return cpuHog0(x, 1e5)
}

func cpuHog0(x, n int) int {
	foo := x
	for i := 0; i < n; i++ {
		if i%1000 == 0 {
			// Spend time in mcall, stored as gp.m.curg, with g0 running
			runtime.Gosched()
		}
		if foo > 0 {
			foo *= foo
		} else {
			foo *= foo + 1
		}
	}
	return foo
}

func saveTrace(t *testing.T, buf *bytes.Buffer, name string) {
	if !*saveTraces {
		return
	}
	if err := os.WriteFile(name+".trace", buf.Bytes(), 0600); err != nil {
		t.Errorf("failed to write trace file: %s", err)
	}
}
