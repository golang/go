/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/13755
 * Buggy version: 7acb881bbb8f23e87b69fce9568d9a3316b5259c
 * fix commit-id: ef906076adc1d0e3721944829cfedfed51810088
 * Flaky: 100/100
 * Description: The buggy code does not close the db query result (rows),
 * so that one goroutine running (*Rows).awaitDone is blocked forever.
 * The blocking goroutine is waiting for cancel signal from context.
 */

package main

import (
	"context"
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("Cockroach13755", Cockroach13755)
}

type Rows_cockroach13755 struct {
	cancel context.CancelFunc
}

func (rs *Rows_cockroach13755) initContextClose(ctx context.Context) {
	ctx, rs.cancel = context.WithCancel(ctx)
	// deadlocks: 1
	go rs.awaitDone(ctx)
}

func (rs *Rows_cockroach13755) awaitDone(ctx context.Context) {
	<-ctx.Done()
	rs.close(ctx.Err())
}

func (rs *Rows_cockroach13755) close(err error) {
	rs.cancel()
}

/// G1 						G2
/// initContextClose()
/// 						awaitDone()
/// 						<-tx.ctx.Done()
/// return
/// ---------------G2 leak-----------------

func Cockroach13755() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	rs := &Rows_cockroach13755{}
	rs.initContextClose(context.Background())
}
