// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/16167
 * Buggy version: 36fa784aa846b46c29e077634c4e362635f6e74a
 * fix commit-id: d064942b067ab84628f79cbfda001fa3138d8d6e
 * Flaky: 1/100
 */

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Cockroach16167", Cockroach16167)
}

type PreparedStatements_cockroach16167 struct {
	session *Session_cockroach16167
}

func (ps PreparedStatements_cockroach16167) New(e *Executor_cockroach16167) {
	e.Prepare(ps.session)
}

type Session_cockroach16167 struct {
	PreparedStatements PreparedStatements_cockroach16167
}

func (s *Session_cockroach16167) resetForBatch(e *Executor_cockroach16167) {
	e.getDatabaseCache()
}

type Executor_cockroach16167 struct {
	systemConfigCond *sync.Cond
	systemConfigMu   sync.RWMutex // L1
}

func (e *Executor_cockroach16167) Start() {
	e.updateSystemConfig()
}

func (e *Executor_cockroach16167) execParsed(session *Session_cockroach16167) {
	e.systemConfigCond.L.Lock() // Same as e.systemConfigMu.RLock()
	runtime.Gosched()
	defer e.systemConfigCond.L.Unlock()
	runTxnAttempt_cockroach16167(e, session)
}

func (e *Executor_cockroach16167) execStmtsInCurrentTxn(session *Session_cockroach16167) {
	e.execStmtInOpenTxn(session)
}

func (e *Executor_cockroach16167) execStmtInOpenTxn(session *Session_cockroach16167) {
	session.PreparedStatements.New(e)
}

func (e *Executor_cockroach16167) Prepare(session *Session_cockroach16167) {
	session.resetForBatch(e)
}

func (e *Executor_cockroach16167) getDatabaseCache() {
	e.systemConfigMu.RLock()
	defer e.systemConfigMu.RUnlock()
}

func (e *Executor_cockroach16167) updateSystemConfig() {
	e.systemConfigMu.Lock()
	runtime.Gosched()
	defer e.systemConfigMu.Unlock()
}

func runTxnAttempt_cockroach16167(e *Executor_cockroach16167, session *Session_cockroach16167) {
	e.execStmtsInCurrentTxn(session)
}

func NewExectorAndSession_cockroach16167() (*Executor_cockroach16167, *Session_cockroach16167) {
	session := &Session_cockroach16167{}
	session.PreparedStatements = PreparedStatements_cockroach16167{session}
	e := &Executor_cockroach16167{}
	return e, session
}

func Cockroach16167() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() { // G1
			e, s := NewExectorAndSession_cockroach16167()
			e.systemConfigCond = sync.NewCond(e.systemConfigMu.RLocker())
			go e.Start()    // G2
			e.execParsed(s)
		}()
	}
}
