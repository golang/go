/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/16167
 * Buggy version: 36fa784aa846b46c29e077634c4e362635f6e74a
 * fix commit-id: d064942b067ab84628f79cbfda001fa3138d8d6e
 * Flaky: 1/100
 * Description:
 *   This is another example for deadlock caused by recursively
 * acquiring RWLock. There are two lock variables (systemConfigCond and systemConfigMu)
 * involved in this bug, but they are actually the same lock, which can be found from
 * the following code.
 *   There are two goroutine involved in this deadlock. The first goroutine acquires
 * systemConfigMu.Lock() firstly, then tries to acquire systemConfigMu.RLock(). The
 * second goroutine tries to acquire systemConfigMu.Lock(). If the second goroutine
 * interleaves in between the two lock operations of the first goroutine, deadlock will happen.
 */

package main

import (
	"runtime"
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
	systemConfigMu   sync.RWMutex
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

/// G1 							G2
/// e.Start()
/// e.updateSystemConfig()
/// 							e.execParsed()
/// 							e.systemConfigCond.L.Lock()
/// e.systemConfigMu.Lock()
/// 							e.systemConfigMu.RLock()
/// ----------------------G1,G2 deadlock--------------------

func Cockroach16167() {
	defer func() {
		time.Sleep(100 * time.Millisecond)
		runtime.GC()
	}()

	for i := 0; i < 100; i++ {
		go func() {
			// deadlocks: x > 0
			e, s := NewExectorAndSession_cockroach16167()
			e.systemConfigCond = sync.NewCond(e.systemConfigMu.RLocker())
			// deadlocks: x > 0
			go e.Start()    // G1
			e.execParsed(s) // G2
		}()
	}
}
