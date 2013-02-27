// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"net/http"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Verify the other tests didn't leave any goroutines running.
// This is in a file named z_last_test.go so it sorts at the end.
func TestGoroutinesRunning(t *testing.T) {
	n := runtime.NumGoroutine()
	t.Logf("num goroutines = %d", n)
	if n > 20 {
		// Currently 14 on Linux (blocked in epoll_wait,
		// waiting for on fds that are closed?), but give some
		// slop for now.
		buf := make([]byte, 1<<20)
		buf = buf[:runtime.Stack(buf, true)]
		t.Errorf("Too many goroutines:\n%s", buf)
	}
}

func checkLeakedTransports(t *testing.T) {
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
	t.Errorf("Test appears to have leaked %s:\n%s", bad, stacks)
}
