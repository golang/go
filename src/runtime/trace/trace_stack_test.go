// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bytes"
	"internal/testenv"
	"internal/trace"
	"net"
	"os"
	"runtime"
	. "runtime/trace"
	"sync"
	"testing"
	"time"
)

// TestTraceSymbolize tests symbolization and that events has proper stacks.
// In particular that we strip bottom uninteresting frames like goexit,
// top uninteresting frames (runtime guts).
func TestTraceSymbolize(t *testing.T) {
	testenv.MustHaveGoBuild(t)

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
	go func() {
		select {}
	}()
	go func() {
		var c chan int
		c <- 0
	}()
	go func() {
		var c chan int
		<-c
	}()
	done1 := make(chan bool)
	go func() {
		<-done1
	}()
	done2 := make(chan bool)
	go func() {
		done2 <- true
	}()
	c1 := make(chan int)
	c2 := make(chan int)
	go func() {
		select {
		case <-c1:
		case <-c2:
		}
	}()
	var mu sync.Mutex
	mu.Lock()
	go func() {
		mu.Lock()
		mu.Unlock()
	}()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		wg.Wait()
	}()
	cv := sync.NewCond(&sync.Mutex{})
	go func() {
		cv.L.Lock()
		cv.Wait()
		cv.L.Unlock()
	}()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	go func() {
		c, err := ln.Accept()
		if err != nil {
			t.Fatalf("failed to accept: %v", err)
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
	go func() {
		var data [1]byte
		rp.Read(data[:])
		pipeReadDone <- true
	}()

	time.Sleep(time.Millisecond)
	runtime.GC()
	runtime.Gosched()
	time.Sleep(time.Millisecond) // the last chance for the goroutines above to block
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

	Stop()
	events, _ := parseTrace(t, buf)

	// Now check that the stacks are correct.
	type frame struct {
		Fn   string
		Line int
	}
	type eventDesc struct {
		Type byte
		Stk  []frame
	}
	want := []eventDesc{
		{trace.EvGCStart, []frame{
			{"runtime.GC", 0},
			{"runtime/trace_test.TestTraceSymbolize", 106},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoStart, []frame{
			{"runtime/trace_test.TestTraceSymbolize.func1", 37},
		}},
		{trace.EvGoSched, []frame{
			{"runtime/trace_test.TestTraceSymbolize", 107},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoCreate, []frame{
			{"runtime/trace_test.TestTraceSymbolize", 39},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoStop, []frame{
			{"runtime.block", 0},
			{"runtime/trace_test.TestTraceSymbolize.func1", 38},
		}},
		{trace.EvGoStop, []frame{
			{"runtime.chansend1", 0},
			{"runtime/trace_test.TestTraceSymbolize.func2", 42},
		}},
		{trace.EvGoStop, []frame{
			{"runtime.chanrecv1", 0},
			{"runtime/trace_test.TestTraceSymbolize.func3", 46},
		}},
		{trace.EvGoBlockRecv, []frame{
			{"runtime.chanrecv1", 0},
			{"runtime/trace_test.TestTraceSymbolize.func4", 50},
		}},
		{trace.EvGoUnblock, []frame{
			{"runtime.chansend1", 0},
			{"runtime/trace_test.TestTraceSymbolize", 109},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockSend, []frame{
			{"runtime.chansend1", 0},
			{"runtime/trace_test.TestTraceSymbolize.func5", 54},
		}},
		{trace.EvGoUnblock, []frame{
			{"runtime.chanrecv1", 0},
			{"runtime/trace_test.TestTraceSymbolize", 110},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockSelect, []frame{
			{"runtime.selectgo", 0},
			{"runtime/trace_test.TestTraceSymbolize.func6", 59},
		}},
		{trace.EvGoUnblock, []frame{
			{"runtime.selectgo", 0},
			{"runtime/trace_test.TestTraceSymbolize", 111},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockSync, []frame{
			{"sync.(*Mutex).Lock", 0},
			{"runtime/trace_test.TestTraceSymbolize.func7", 67},
		}},
		{trace.EvGoUnblock, []frame{
			{"sync.(*Mutex).Unlock", 0},
			{"runtime/trace_test.TestTraceSymbolize", 115},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockSync, []frame{
			{"sync.(*WaitGroup).Wait", 0},
			{"runtime/trace_test.TestTraceSymbolize.func8", 73},
		}},
		{trace.EvGoUnblock, []frame{
			{"sync.(*WaitGroup).Add", 0},
			{"sync.(*WaitGroup).Done", 0},
			{"runtime/trace_test.TestTraceSymbolize", 116},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoBlockCond, []frame{
			{"sync.(*Cond).Wait", 0},
			{"runtime/trace_test.TestTraceSymbolize.func9", 78},
		}},
		{trace.EvGoUnblock, []frame{
			{"sync.(*Cond).Signal", 0},
			{"runtime/trace_test.TestTraceSymbolize", 117},
			{"testing.tRunner", 0},
		}},
		{trace.EvGoSleep, []frame{
			{"time.Sleep", 0},
			{"runtime/trace_test.TestTraceSymbolize", 108},
			{"testing.tRunner", 0},
		}},
	}
	// Stacks for the following events are OS-dependent due to OS-specific code in net package.
	if runtime.GOOS != "windows" && runtime.GOOS != "plan9" {
		want = append(want, []eventDesc{
			{trace.EvGoBlockNet, []frame{
				{"net.(*netFD).accept", 0},
				{"net.(*TCPListener).accept", 0},
				{"net.(*TCPListener).Accept", 0},
				{"runtime/trace_test.TestTraceSymbolize.func10", 86},
			}},
			{trace.EvGoSysCall, []frame{
				{"syscall.read", 0},
				{"syscall.Read", 0},
				{"os.(*File).read", 0},
				{"os.(*File).Read", 0},
				{"runtime/trace_test.TestTraceSymbolize.func11", 101},
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
	for i, m := range matched {
		if m {
			continue
		}
		w := want[i]
		t.Errorf("did not match event %v at %v:%v", trace.EventDescriptions[w.Type].Name, w.Stk[0].Fn, w.Stk[0].Line)
		t.Errorf("seen the following events of this type:")
		for _, ev := range events {
			if ev.Type != w.Type {
				continue
			}
			for _, f := range ev.Stk {
				t.Logf("  %v:%v", f.Fn, f.Line)
			}
			t.Logf("---")
		}
	}
}
