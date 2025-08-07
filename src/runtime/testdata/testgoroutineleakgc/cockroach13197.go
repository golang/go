/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/13197
 * Buggy version: fff27aedabafe20cef57f75905fe340cab48c2a4
 * fix commit-id: 9bf770cd8f6eaff5441b80d3aec1a5614e8747e1
 * Flaky: 100/100
 * Description: One goroutine executing (*Tx).awaitDone() blocks and
 * waiting for a signal context.Done().
 */
package main

import (
	"context"
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("Cockroach13197", Cockroach13197)
}

type DB_cockroach13197 struct{}

func (db *DB_cockroach13197) begin(ctx context.Context) *Tx_cockroach13197 {
	ctx, cancel := context.WithCancel(ctx)
	tx := &Tx_cockroach13197{
		cancel: cancel,
		ctx:    ctx,
	}
	// deadlocks: 1
	go tx.awaitDone() // G2
	return tx
}

type Tx_cockroach13197 struct {
	cancel context.CancelFunc
	ctx    context.Context
}

func (tx *Tx_cockroach13197) awaitDone() {
	<-tx.ctx.Done()
}

func (tx *Tx_cockroach13197) Rollback() {
	tx.rollback()
}

func (tx *Tx_cockroach13197) rollback() {
	tx.close()
}

func (tx *Tx_cockroach13197) close() {
	tx.cancel()
}

/// G1 				G2
/// begin()
/// 				awaitDone()
/// 				<-tx.ctx.Done()
/// return
/// -----------G2 leak-------------

func Cockroach13197() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	db := &DB_cockroach13197{}
	db.begin(context.Background()) // G1
}
