// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

var quietLog = log.New(io.Discard, "", 0)

func TestMain(m *testing.M) {
	*http.MaxWriteWaitBeforeConnReuse = 60 * time.Minute
	v := m.Run()
	if v == 0 && goroutineLeaked() {
		os.Exit(1)
	}
	os.Exit(v)
}

func interestingGoroutines() (gs []string) {
	buf := make([]byte, 2<<20)
	buf = buf[:runtime.Stack(buf, true)]
	for _, g := range strings.Split(string(buf), "\n\n") {
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
	for i := 0; i < 5; i++ {
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

// setParallel marks t as a parallel test if we're in short mode
// (all.bash), but as a serial test otherwise. Using t.Parallel isn't
// compatible with the afterTest func in non-short mode.
func setParallel(t *testing.T) {
	if strings.Contains(t.Name(), "HTTP2") {
		http.CondSkipHTTP2(t)
	}
	if testing.Short() {
		t.Parallel()
	}
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

var leakReported bool

func afterTest(t testing.TB) {
	http.DefaultTransport.(*http.Transport).CloseIdleConnections()
	if testing.Short() {
		return
	}
	if leakReported {
		// To avoid confusion, only report the first leak of each test run.
		// After the first leak has been reported, we can't tell whether the leaked
		// goroutines are a new leak from a subsequent test or just the same
		// goroutines from the first leak still hanging around, and we may add a lot
		// of latency waiting for them to exit at the end of each test.
		return
	}

	// We shouldn't be running the leak check for parallel tests, because we might
	// report the goroutines from a test that is still running as a leak from a
	// completely separate test that has just finished. So we use non-atomic loads
	// and stores for the leakReported variable, and store every time we start a
	// leak check so that the race detector will flag concurrent leak checks as a
	// race even if we don't detect any leaks.
	leakReported = true

	var bad string
	badSubstring := map[string]string{
		").readLoop(":  "a Transport",
		").writeLoop(": "a Transport",
		"created by net/http/httptest.(*Server).Start": "an httptest.Server",
		"timeoutHandler":        "a TimeoutHandler",
		"net.(*netFD).connect(": "a timing out dial",
		").noteClientGone(":     "a closenotifier sender",
	}
	var stacks string
	for i := 0; i < 2500; i++ {
		bad = ""
		stacks = strings.Join(interestingGoroutines(), "\n\n")
		for substr, what := range badSubstring {
			if strings.Contains(stacks, substr) {
				bad = what
			}
		}
		if bad == "" {
			leakReported = false
			return
		}
		// Bad stuff found, but goroutines might just still be
		// shutting down, so give it some time.
		time.Sleep(1 * time.Millisecond)
	}
	t.Errorf("Test appears to have leaked %s:\n%s", bad, stacks)
}

// waitCondition waits for fn to return true,
// checking immediately and then at exponentially increasing intervals.
func waitCondition(t testing.TB, delay time.Duration, fn func(time.Duration) bool) {
	t.Helper()
	start := time.Now()
	var since time.Duration
	for !fn(since) {
		time.Sleep(delay)
		delay = 2*delay - (delay / 2) // 1.5x, rounded up
		since = time.Since(start)
	}
}
