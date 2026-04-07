// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"net/http"
	"os"
	"runtime/trace"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	itrace "internal/trace"
	"internal/trace/testtrace"
)

// Regression test for go.dev/issue/74850.
func TestSyscallProfile74850(t *testing.T) {
	testtrace.MustHaveSyscallEvents(t)

	var buf bytes.Buffer
	err := trace.Start(&buf)
	if err != nil {
		t.Fatalf("start tracing: %v", err)
	}

	synctest.Test(t, func(t *testing.T) {
		go hidden1(t)
		go hidden2(t)
		go visible(t)
		synctest.Wait()
		time.Sleep(1 * time.Millisecond)
		synctest.Wait()
	})
	trace.Stop()

	if t.Failed() {
		return
	}

	parsed, err := parseTrace(&buf, int64(buf.Len()))
	if err != nil {
		t.Fatalf("parsing trace: %v", err)
	}

	records, err := pprofByGoroutine(computePprofSyscall(), parsed)(&http.Request{})
	if err != nil {
		t.Fatalf("failed to generate pprof: %v\n", err)
	}

	for _, r := range records {
		t.Logf("Record: n=%d, total=%v", r.Count, r.Time)
		for _, f := range r.Stack {
			t.Logf("\t%s", f.Func)
			t.Logf("\t\t%s:%d @ 0x%x", f.File, f.Line, f.PC)
		}
	}
	if len(records) == 0 {
		t.Error("empty profile")
	}

	// Make sure we see the right frames.
	wantSymbols := []string{"cmd/trace.visible", "cmd/trace.hidden1", "cmd/trace.hidden2"}
	haveSymbols := make([]bool, len(wantSymbols))
	for _, r := range records {
		for _, f := range r.Stack {
			for i, s := range wantSymbols {
				if strings.Contains(f.Func, s) {
					haveSymbols[i] = true
				}
			}
		}
	}
	for i, have := range haveSymbols {
		if !have {
			t.Errorf("expected %s in syscall profile", wantSymbols[i])
		}
	}
}

// Regression test: pcsForStack must not panic when the stack has more than
// pprofMaxStack (128) frames. Before the fix, pcs[i] was written before
// checking i >= len(pcs), causing an index-out-of-range panic.
func TestFillPCsOverflow(t *testing.T) {
	var pcs [pprofMaxStack]uint64

	// Build a stack with more frames than pprofMaxStack.
	n := pprofMaxStack + 10
	frames := make([]itrace.StackFrame, n)
	for i := range frames {
		frames[i].PC = uint64(0x1000 + i)
	}
	stack := itrace.MakeStack(frames)

	// This must not panic.
	pcsForStack(stack, &pcs)

	// Verify the first pprofMaxStack entries were written correctly.
	for i := 0; i < pprofMaxStack; i++ {
		if pcs[i] != uint64(0x1000+i) {
			t.Errorf("pcs[%d] = %#x, want %#x", i, pcs[i], 0x1000+i)
		}
	}
}

func stat(t *testing.T) {
	_, err := os.Stat(".")
	if err != nil {
		t.Errorf("os.Stat: %v", err)
	}
}

func hidden1(t *testing.T) {
	stat(t)
}

func hidden2(t *testing.T) {
	stat(t)
	stat(t)
}

func visible(t *testing.T) {
	stat(t)
	time.Sleep(1 * time.Millisecond)
}
