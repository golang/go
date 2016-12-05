// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	g0 := runtime.NumGoroutine()

	code := m.Run()
	if code != 0 {
		os.Exit(code)
	}

	// Check that there are no goroutines left behind.
	t0 := time.Now()
	stacks := make([]byte, 1<<20)
	for {
		g1 := runtime.NumGoroutine()
		if g1 == g0 {
			return
		}
		stacks = stacks[:runtime.Stack(stacks, true)]
		time.Sleep(50 * time.Millisecond)
		if time.Since(t0) > 2*time.Second {
			fmt.Fprintf(os.Stderr, "Unexpected leftover goroutines detected: %v -> %v\n%s\n", g0, g1, stacks)
			os.Exit(1)
		}
	}
}

func TestContextCancel(t *testing.T) {
	ctx := t.Context()
	// Tests we don't leak this goroutine:
	go func() {
		<-ctx.Done()
	}()
}
