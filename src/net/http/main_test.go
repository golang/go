// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"io"
	"log"
	"net/http"
	"os"
	"testing"
	"time"
)

var quietLog = log.New(io.Discard, "", 0)

func TestMain(m *testing.M) {
	*http.MaxWriteWaitBeforeConnReuse = 60 * time.Minute
	os.Exit(m.Run())
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
