// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3_test

import (
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	v := m.Run()
	if v == 0 && goroutineLeaked() {
		os.Exit(1)
	}
	os.Exit(v)
}

func runningBenchmarks() bool {
	for i, arg := range os.Args {
		if strings.HasPrefix(arg, "-test.bench=") && !strings.HasSuffix(arg, "=") {
			return true
		}
		if arg == "-test.bench" && i < len(os.Args)-1 && os.Args[i+1] != "" {
			return true
		}
	}
	return false
}

func interestingGoroutines() (gs []string) {
	buf := make([]byte, 2<<20)
	buf = buf[:runtime.Stack(buf, true)]
	for g := range strings.SplitSeq(string(buf), "\n\n") {
		_, stack, _ := strings.Cut(g, "\n")
		stack = strings.TrimSpace(stack)
		if stack == "" ||
			strings.Contains(stack, "testing.(*M).before.func1") ||
			strings.Contains(stack, "os/signal.signal_recv") ||
			strings.Contains(stack, "created by net.startServer") ||
			strings.Contains(stack, "created by testing.RunTests") ||
			strings.Contains(stack, "closeWriteAndWait") ||
			strings.Contains(stack, "testing.Main(") ||
			// These only show up with GOTRACEBACK=2; Issue 5005 (comment 28)
			strings.Contains(stack, "runtime.goexit") ||
			strings.Contains(stack, "created by runtime.gc") ||
			strings.Contains(stack, "interestingGoroutines") ||
			strings.Contains(stack, "runtime.MHeap_Scavenger") {
			continue
		}
		gs = append(gs, stack)
	}
	slices.Sort(gs)
	return
}

// Verify the other tests didn't leave any goroutines running.
func goroutineLeaked() bool {
	if testing.Short() || runningBenchmarks() {
		// Don't worry about goroutine leaks in -short mode or in
		// benchmark mode. Too distracting when there are false positives.
		return false
	}

	var stackCount map[string]int
	for range 5 {
		n := 0
		stackCount = make(map[string]int)
		gs := interestingGoroutines()
		for _, g := range gs {
			stackCount[g]++
			n++
		}
		if n == 0 {
			return false
		}
		// Wait for goroutines to schedule and die off:
		time.Sleep(100 * time.Millisecond)
	}
	fmt.Fprintf(os.Stderr, "Too many goroutines running after net/http test(s).\n")
	for stack, count := range stackCount {
		fmt.Fprintf(os.Stderr, "%d instances of:\n%s\n", count, stack)
	}
	return true
}
