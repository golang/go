// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"internal/trace"
	"net"
	"os"
	"runtime"
	. "runtime/trace"
	"strings"
	"sync"
	"testing"
	"text/tabwriter"
	"time"
)

// TestTraceSymbolize tests symbolization and that events has proper stacks.
// In particular that we strip bottom uninteresting frames like goexit,
// top uninteresting frames (runtime guts).
func TestTraceSymbolize(t *testing.T) {
	skipTraceSymbolizeTestIfNecessary(t)

	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}
	defer Stop() // in case of early return

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
		t.Fatalf("failed to listen: %v", err)
	}
	go func() { // func10
		c, err := ln.Accept()
		if err != nil {
			t.Errorf("failed to accept: %v", err)
			return
		}
		c.Close()
	}()
	rp, wp, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create a pipe: %v", err)
	}
	defer rp.Close()
	defer wp.Close()
	pipeReadDone := make(chan bool)
	go func() { // func11
		var data [1]byte
		rp.Read(data[:])
		pipeReadDone <- true
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
		t.Fatalf("failed to dial: %v", err)
	}
	c.Close()
	var data [1]byte
	wp.Write(data[:])
	<-pipeReadDone

	oldGoMaxProcs := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(oldGoMaxProcs + 1)

	Stop()

	runtime.GOMAXPROCS(oldGoMaxProcs)

	events, _ := parseTrace(t, buf)

	// Now check that the stacks are correct.
	type eventDesc struct {
		Type byte
		Stk  []frame
	}
	want := []eventDesc{
		{trace.EvGCStart, []frame{
			{"runtime.GC", 0},
			{"runtime/trace_test.TestTraceSymbolize", 0},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoStart, []frame{
			{"runtime/trace_test.TestTraceSymbolize.func1", 0},
		}},
		{trace.EvGoSched, []frame{
			{"runtime/trace_test.TestTraceSymbolize", 111},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoCreate, []frame{
			{"runtime/trace_test.TestTraceSymbolize", 40},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoStop, []frame{
			{"runtime.block", 0},
			{"runtime/trace_test.TestTraceSymbolize.func1", 0},
		}},
		{trace.EvGoStop, []frame{
			{"runtime.chansend1", 0},
			{"runtime/trace_test.TestTraceSymbolize.func2", 0},
		}},
		{trace.EvGoStop, []frame{
			{"runtime.chanrecv1", 0},
			{"runtime/trace_test.TestTraceSymbolize.func3", 0},
		}},
		{trace.EvGoBlockRecv, []frame{
			{"runtime.chanrecv1", 0},
			{"runtime/trace_test.TestTraceSymbolize.func4", 0},
		}},
		{trace.EvGoUnblock, []frame{
			{"runtime.chansend1", 0},
			{"runtime/trace_test.TestTraceSymbolize", 113},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockSend, []frame{
			{"runtime.chansend1", 0},
			{"runtime/trace_test.TestTraceSymbolize.func5", 0},
		}},
		{trace.EvGoUnblock, []frame{
			{"runtime.chanrecv1", 0},
			{"runtime/trace_test.TestTraceSymbolize", 114},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockSelect, []frame{
			{"runtime.selectgo", 0},
			{"runtime/trace_test.TestTraceSymbolize.func6", 0},
		}},
		{trace.EvGoUnblock, []frame{
			{"runtime.selectgo", 0},
			{"runtime/trace_test.TestTraceSymbolize", 115},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockSync, []frame{
			{"sync.(*Mutex).Lock", 0},
			{"runtime/trace_test.TestTraceSymbolize.func7", 0},
		}},
		{trace.EvGoUnblock, []frame{
			{"sync.(*Mutex).Unlock", 0},
			{"runtime/trace_test.TestTraceSymbolize", 0},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockSync, []frame{
			{"sync.(*WaitGroup).Wait", 0},
			{"runtime/trace_test.TestTraceSymbolize.func8", 0},
		}},
		{trace.EvGoUnblock, []frame{
			{"sync.(*WaitGroup).Add", 0},
			{"sync.(*WaitGroup).Done", 0},
			{"runtime/trace_test.TestTraceSymbolize", 120},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockCond, []frame{
			{"sync.(*Cond).Wait", 0},
			{"runtime/trace_test.TestTraceSymbolize.func9", 0},
		}},
		{trace.EvGoUnblock, []frame{
			{"sync.(*Cond).Signal", 0},
			{"runtime/trace_test.TestTraceSymbolize", 0},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoSleep, []frame{
			{"time.Sleep", 0},
			{"runtime/trace_test.TestTraceSymbolize", 0},
			{"testing.tRunner", 0},
		}},
		{trace.EvGomaxprocs, []frame{
			{"runtime.startTheWorld", 0}, // this is when the current gomaxprocs is logged.
			{"runtime.startTheWorldGC", 0},
			{"runtime.GOMAXPROCS", 0},
			{"runtime/trace_test.TestTraceSymbolize", 0},
			{"testing.tRunner", 0},
		}},
	}
	// Stacks for the following events are OS-dependent due to OS-specific code in net package.
	if runtime.GOOS != "windows" && runtime.GOOS != "plan9" {
		want = append(want, []eventDesc{
			{trace.EvGoBlockNet, []frame{
				{"internal/poll.(*FD).Accept", 0},
				{"net.(*netFD).accept", 0},
				{"net.(*TCPListener).accept", 0},
				{"net.(*TCPListener).Accept", 0},
				{"runtime/trace_test.TestTraceSymbolize.func10", 0},
			}},
			{trace.EvGoSysCall, []frame{
				{"syscall.read", 0},
				{"syscall.Read", 0},
				{"internal/poll.ignoringEINTR", 0},
				{"internal/poll.(*FD).Read", 0},
				{"os.(*File).read", 0},
				{"os.(*File).Read", 0},
				{"runtime/trace_test.TestTraceSymbolize.func11", 0},
			}},
		}...)
	}
	matched := make([]bool, len(want))
	for _, ev := range events {
	wantLoop:
		for i, w := range want {
			if matched[i] || w.Type != ev.Type || len(w.Stk) != len(ev.Stk) {
				continue
			}

			for fi, f := range ev.Stk {
				wf := w.Stk[fi]
				if wf.Fn != f.Fn || wf.Line != 0 && wf.Line != f.Line {
					continue wantLoop
				}
			}
			matched[i] = true
		}
	}
	for i, w := range want {
		if matched[i] {
			continue
		}
		seen, n := dumpEventStacks(w.Type, events)
		t.Errorf("Did not match event %v with stack\n%s\nSeen %d events of the type\n%s",
			trace.EventDescriptions[w.Type].Name, dumpFrames(w.Stk), n, seen)
	}
}

func skipTraceSymbolizeTestIfNecessary(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
}

func dumpEventStacks(typ byte, events []*trace.Event) ([]byte, int) {
	matched := 0
	o := new(bytes.Buffer)
	tw := tabwriter.NewWriter(o, 0, 8, 0, '\t', 0)
	for _, ev := range events {
		if ev.Type != typ {
			continue
		}
		matched++
		fmt.Fprintf(tw, "Offset %d\n", ev.Off)
		for _, f := range ev.Stk {
			fname := f.File
			if idx := strings.Index(fname, "/go/src/"); idx > 0 {
				fname = fname[idx:]
			}
			fmt.Fprintf(tw, "  %v\t%s:%d\n", f.Fn, fname, f.Line)
		}
	}
	tw.Flush()
	return o.Bytes(), matched
}

type frame struct {
	Fn   string
	Line int
}

func dumpFrames(frames []frame) []byte {
	o := new(bytes.Buffer)
	tw := tabwriter.NewWriter(o, 0, 8, 0, '\t', 0)

	for _, f := range frames {
		fmt.Fprintf(tw, "  %v\t :%d\n", f.Fn, f.Line)
	}
	tw.Flush()
	return o.Bytes()
}
