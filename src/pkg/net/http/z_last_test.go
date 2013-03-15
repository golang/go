// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"net/http"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

func interestingGoroutines() (gs []string) {
	buf := make([]byte, 2<<20)
	buf = buf[:runtime.Stack(buf, true)]
	for _, g := range strings.Split(string(buf), "\n\n") {
		sl := strings.SplitN(g, "\n", 2)
		if len(sl) != 2 {
			continue
		}
		stack := strings.TrimSpace(sl[1])
		if stack == "" ||
			strings.Contains(stack, "created by net.newPollServer") ||
			strings.Contains(stack, "created by net.startServer") ||
			strings.Contains(stack, "created by testing.RunTests") ||
			strings.Contains(stack, "closeWriteAndWait") ||
			strings.Contains(stack, "testing.Main(") {
			continue
		}
		gs = append(gs, stack)
	}
	sort.Strings(gs)
	return
}

// Verify the other tests didn't leave any goroutines running.
// This is in a file named z_last_test.go so it sorts at the end.
func TestGoroutinesRunning(t *testing.T) {
	gs := interestingGoroutines()

	n := 0
	stackCount := make(map[string]int)
	for _, g := range gs {
		stackCount[g]++
		n++
	}

	t.Logf("num goroutines = %d", n)
	if n > 0 {
		t.Error("Too many goroutines.")
		for stack, count := range stackCount {
			t.Logf("%d instances of:\n%s", count, stack)
		}
	}
}

func afterTest(t *testing.T) {
	http.DefaultTransport.(*http.Transport).CloseIdleConnections()
	if testing.Short() {
		return
	}
	buf := make([]byte, 1<<20)
	var stacks string
	var bad string
	badSubstring := map[string]string{
		").readLoop(":                                  "a Transport",
		").writeLoop(":                                 "a Transport",
		"created by net/http/httptest.(*Server).Start": "an httptest.Server",
		"timeoutHandler":                               "a TimeoutHandler",
		"net.(*netFD).connect(":                        "a timing out dial",
	}
	for i := 0; i < 4; i++ {
		bad = ""
		stacks = string(buf[:runtime.Stack(buf, true)])
		for substr, what := range badSubstring {
			if strings.Contains(stacks, substr) {
				bad = what
			}
		}
		if bad == "" {
			return
		}
		// Bad stuff found, but goroutines might just still be
		// shutting down, so give it some time.
		time.Sleep(250 * time.Millisecond)
	}
	gs := interestingGoroutines()
	t.Errorf("Test appears to have leaked %s:\n%s", bad, strings.Join(gs, "\n\n"))
}
