// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/13755
 * Buggy version: 7acb881bbb8f23e87b69fce9568d9a3316b5259c
 * fix commit-id: ef906076adc1d0e3721944829cfedfed51810088
 * Flaky: 100/100
 */

package main

import (
	"context"
	"os"
	"runtime"
	"runtime/pprof"
)

func init() {
	register("Cockroach13755", Cockroach13755)
}

type Rows_cockroach13755 struct {
	cancel context.CancelFunc
}

func (rs *Rows_cockroach13755) initContextClose(ctx context.Context) {
	ctx, rs.cancel = context.WithCancel(ctx)
	go rs.awaitDone(ctx)
}

func (rs *Rows_cockroach13755) awaitDone(ctx context.Context) {
	<-ctx.Done()
	rs.close(ctx.Err())
}

func (rs *Rows_cockroach13755) close(err error) {
	rs.cancel()
}

// Example of goroutine leak trace:
//
// G1 						      G2
//----------------------------------------
// initContextClose()
// .                    awaitDone()
// <<done>>             .
//                      <-tx.ctx.Done()
//----------------G2 leak-----------------

func Cockroach13755() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()

	rs := &Rows_cockroach13755{}
	rs.initContextClose(context.Background())
}
