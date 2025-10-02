// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/18101
 * Buggy version: f7a8e2f57b6bcf00b9abaf3da00598e4acd3a57f
 * fix commit-id: 822bd176cc725c6b50905ea615023200b395e14f
 * Flaky: 100/100
 */

package main

import (
	"context"
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("Cockroach18101", Cockroach18101)
}

const chanSize_cockroach18101 = 6

func restore_cockroach18101(ctx context.Context) bool {
	readyForImportCh := make(chan bool, chanSize_cockroach18101)
	go func() { // G2
		defer close(readyForImportCh)
		splitAndScatter_cockroach18101(ctx, readyForImportCh)
	}()
	for readyForImportSpan := range readyForImportCh {
		select {
		case <-ctx.Done():
			return readyForImportSpan
		}
	}
	return true
}

func splitAndScatter_cockroach18101(ctx context.Context, readyForImportCh chan bool) {
	for i := 0; i < chanSize_cockroach18101+2; i++ {
		readyForImportCh <- (false || i != 0)
	}
}

func Cockroach18101() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 100; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		go restore_cockroach18101(ctx) // G1
		go cancel()                    // helper goroutine
	}
}
