// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/13197
 * Buggy version: fff27aedabafe20cef57f75905fe340cab48c2a4
 * fix commit-id: 9bf770cd8f6eaff5441b80d3aec1a5614e8747e1
 * Flaky: 100/100
 * Description: One goroutine executing (*Tx).awaitDone() blocks
 * waiting for a signal over context.Done() that never comes.
 */
package main

import (
	"context"
	"os"
	"runtime"
	"runtime/pprof"
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

// Example of goroutine leak trace:
//
// G1 				  G2
//--------------------------------
// begin()
// .            awaitDone()
// <<done>>     .
//              <-tx.ctx.Done()
//------------G2 leak-------------

func Cockroach13197() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()

	db := &DB_cockroach13197{}
	db.begin(context.Background()) // G1
}
