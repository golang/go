// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"context"
	"net/http/httptrace"
	"net/http/internal"
	"sync"
	"sync/atomic"
	"time"
)

// clientConnPool is the connection pool state for Transport,
// split out into a separate struct for organizational purposes.
//
// Every Transport has a pool.
//
// The pool has a set of entries for targets (things we can dial).
// Entries are removed from the pool when nothing references them any more
// (no conns, no dials, no waiting requests for a conn, etc.)
// An entry manages the set of connections for a target.
//
// The pool also has an LRU of connections, used to maintain the cross-target MaxIdleConn limit.
//
// All pool and entry state is guarded by clientConnPool.mu.
//
// Lock ordering is clientConnPool before persistConn.
//
// The pool mutex should never be held during blocking operations
// (such as dials or closing a net.Conn).
type clientConnPool struct {
	mu   sync.Mutex
	ents map[connectMethodKey]*poolEntry

	// lru is a list of idle connections, longest-idle at the front of the list.
	lru intrusiveList[poolConn]

	// Idle timer, fires when we should next check to see if conns
	// should be closed for idleness.
	idleTimer    *time.Timer
	idleTimerSet bool

	// dirty is a list of pool entries which need to have their state reconciled.
	// Pool state updates follow the following process:
	//   - lock the pool
	//   - perform updates, recording every touched entry in the dirty list
	//   - reconcile each entry on the dirty list
	//   - unlock the pool
	dirty intrusiveList[poolEntry]

	// closeIdle is set by CloseIdleConnections and cleared by poolGetConn.
	// While set, newly-idle connections are closed.
	// (This extends the effect of CloseIdleConnections until the next RoundTrip.)
	closeIdle bool
}

// poolConnState is the state of a pooled connection.
type poolConnState uint8

const (
	poolConnUnknown = 0

	// Broken: No longer usable, should be removed from the pool.
	poolConnBroken = poolConnState(1 << iota)

	// Idle: No in-flight requests, not broken.
	poolConnIdle

	// Active: At least one in-flight request, not broken.
	poolConnActive

	// Exclusive: One in-flight request, which cannot share the connection.
	// The connection will be closed when the request finishes.
	//
	// For example: Connection handling a request with Connection: close.
	poolConnExclusive

	// Draining: At least one in-flight request, can never take another request,
	// should be closed when all requests finish.
	// Not counted against MaxConnsPerHost.
	//
	// For example: HTTP/2 connection which has received a GOAWAY.
	poolConnDraining

	// Alt-conns encompass multitudes, and sit outside the other states.
	poolConnAlt
)

func (state poolConnState) is(other poolConnState) bool {
	return state&other != 0
}

// poolConn is a pooled connection.
//
// All non-atomic state is guarded by the pool lock, except where otherwise noted.
type poolConn struct {
	cc      genericClientConn // set once, never changes
	poolEnt *poolEntry        // set once, never changes

	availableLink intrusiveLink[poolConn] // poolEntry.availableConns
	lruLink       intrusiveLink[poolConn] // clientConnPool.lru

	idleAt time.Time // time the conn last became idle; non-zero if state is Idle

	// A conn's state is set by poolConnStateChangedLocked, never anything else.
	state poolConnState

	// These fields affect the state of the conn.
	// Call poolConnStateChangedLocked after changing one to update the state.
	//
	// For example, setting closing=true will cause a conn to close (if it hasn't already).
	closing   bool // the pool has decided to close and remove this conn
	exclusive bool // this conn is reserved for a single request, close when idle

	reused atomic.Bool // set when a request is sent, for GotConn reporting
}

func (pc *poolConn) isAlt() bool {
	_, ok := pc.cc.(*altClientConn)
	return ok
}

// poolEntry is per-connectMethod pool state.
type poolEntry struct {
	key connectMethodKey // set once, never changes

	dirtyLink intrusiveLink[poolEntry] // clientConnPool.dirty

	// Connections available for use, most recently used last.
	availableConns intrusiveList[poolConn]

	// Connection with isAlt() == true.
	//
	// This is historically a pseudo-"connection" backed by an http2.Transport,
	// which manages its own connection pool. Lack of visibility into this pool
	// causes a variety of bugs, such as connections-per-host tracking not accounting
	// for individual HTTP/2 connections.
	altConn *poolConn

	// Connection counts.
	numConns int // total
	numIdle  int // in state Idle
	sharedH2 int // HTTP/2, and not exclusive

	// Requests waiting for a connection and/or a dial result.
	//
	// A waiter is in the waiters list while it is waiting for a connection,
	// and in the dials list while a connection is being dialed for it.
	//
	// A waiter may be in both lists at the same time.
	waiters intrusiveList[poolWaiter]
	dials   intrusiveList[poolWaiter]

	// When set, we should only have one outstanding dial at a time.
	coalesceDials bool
}

// poolWaiter is a request for a connection from the pool.
type poolWaiter struct {
	cm  connectMethod   // set once, never changes
	ctx context.Context // set once, never changes

	// exclusive is set when this waiter needs a connection to itself,
	// and cannot share with other requests. Set once, never changes.
	exclusive bool

	waitLink intrusiveLink[poolWaiter] // poolEntry.waiters
	dialLink intrusiveLink[poolWaiter] // poolEntry.dials

	// While a waiter is waiting for a poolConn (or an error), it is in ent.waiters.
	//
	// A waiter may give up and stop waiting, removing itself from ent.waiters.
	//
	// If something delivers a result to the waiter, the deliverer:
	//   - ensures the waiter is still waiting (is in ent.waiters)
	//   - removes the waiter from ent.waiters (delivery ends the wait)
	//   - sends on w.donec to wake up the waiter
	//
	// The pconn, info, err, and dialed fields must not be changed after donec is signaled,
	// permitting the waiter to read them without acquiring the pool mutex.
	donec  chan struct{}
	pconn  *poolConn
	info   httptrace.GotConnInfo
	err    error
	dialed bool // set when a dial has started for the entry

	// Dial-related state.
	//
	// Every dial is associated with a request, even if the result of the dial
	// may be used for a different request, or many requests.
	//
	// We therefore use the poolWaiter for both
	// waiting-for-a-request and dialing-a-new-conn-for-a-request state.
	dialCancel context.CancelFunc // called to cancel the dial
}

// withPoolLock calls f with the pool locked.
//
// Before unlocking the pool, it unblocks waiting requests which can make progress:
// Available connections are delivered and new dials are started.
func (t *Transport) withPoolLock(f func()) {
	t.pool.mu.Lock()
	defer t.pool.mu.Unlock()
	f()
	// Reconcile dirty pool entries.
	//
	// Most pool operations are on a single entry,
	// but some (e.g., CloseIdleConnections) touch many.
	// We record touched entries in a dirty list,
	// so we can reconcile once per entry.
	//
	// After reconciling dirty entries, we check the pool-global MaxIdleConns.
	// This may close a connection belonging to an entry f didn't touch
	// (marking that entry dirty), so we loop and reconcile that entry.
	// Reconciliation never adds connections to the idle LRU,
	// so the loop is guaranteed to terminate.
	for {
		for t.pool.dirty.head != nil {
			ent := t.pool.dirty.head
			t.pool.dirty.remove(ent)
			t.poolReconcileEntryLocked(ent)
		}
		if t.MaxIdleConns == 0 || t.pool.lru.size <= max(t.MaxIdleConns, 0) {
			break
		}
		// Closing the oldest idle connection calls poolConnStateChangedLocked,
		// which marks its poolEnt dirty for the next iteration.
		t.poolCloseConnLocked(t.pool.lru.head, errTooManyIdle)
	}
}

// poolReconcileEntryLocked reconciles ent after its state has changed.
//
// Its main job is satisfying waiting requests by delivering newly-available
// connections or starting new dials.
func (t *Transport) poolReconcileEntryLocked(ent *poolEntry) {
	if ent.numConns == 0 && ent.waiters.size == 0 && ent.dials.size == 0 && ent.availableConns.size == 0 && ent.altConn == nil {
		// This entry contains nothing of use, and may be removed from the pool.
		if t.pool.ents[ent.key] == ent {
			delete(t.pool.ents, ent.key)
		}
		return
	}
	canGet := true
	for waiter := range ent.waiters.all() {
		if canGet {
			// Try fetching an idle conn.
			pconn, info := t.poolTryGetForEntryLocked(ent, waiter.exclusive)
			if pconn != nil {
				ent.waiters.remove(waiter)
				waiter.pconn = pconn
				waiter.info = info
				waiter.donec <- struct{}{}
				continue
			}
			// We're out of idle conns, so don't look on future iterations.
			if !waiter.exclusive {
				canGet = false
			}
		}
		if waiter.dialed {
			continue
		}
		canDial, atLimit := t.poolCanDialLocked(ent, waiter.exclusive)
		if atLimit && !canGet {
			// At the connection limit and out of idle conns.
			break
		}
		if !canDial {
			// This request cannot dial, but we aren't at the connection limit
			// so keep looking for any exclusive dials which could start.
			continue
		}

		// Start a new dial.
		//
		// Create a context for the dial which inherits the
		// request context's values, but not its cancellation
		// (since the dial may outlive the request).
		//
		// Note that we only cancel this context to cancel the dial.
		// If the dial finishes, we leave it uncanceled since cancelling
		// it can (for some user-provided DialContext functions)
		// close the connection.
		dialCtx, cancel := context.WithCancel(context.WithoutCancel(waiter.ctx))
		waiter.dialed = true
		waiter.dialCancel = cancel
		ent.dials.pushBack(waiter)
		go t.poolDialForEntry(dialCtx, ent, waiter)
	}
}

// poolCanDialLocked reports whether a new dial may start (canDial).
// It also reports whether the entry is at the MaxConnsPerHost limit, blocking all dials.
func (t *Transport) poolCanDialLocked(ent *poolEntry, exclusive bool) (canDial, atLimit bool) {
	if t.MaxConnsPerHost > 0 && ent.numConns+ent.dials.size >= t.MaxConnsPerHost {
		// At the MaxConnsPerHost limit.
		return false, true
	}
	if exclusive {
		// Dials for an exclusive connection ignore coalescing and
		// StrictMaxConcurrentRequests.
		return true, false
	}
	if ent.coalesceDials {
		for dial := range ent.dials.all() {
			if !dial.exclusive {
				// Coalese with this in-progress dial.
				// (Note that we don't want to coalesce with an exclusive dial,
				// since we wouldn't be able to share its connection.)
				return false, false
			}
		}
	}
	if ent.sharedH2 > 0 && t.strictMaxConcurrentRequests() {
		// StrictMaxConcurrentRequests limits us to one shared HTTP/2 connection.
		return false, false
	}
	// The dial may proceed.
	return true, false
}

func (t *Transport) strictMaxConcurrentRequests() bool {
	if t.HTTP2 != nil && t.HTTP2.StrictMaxConcurrentRequests {
		return true
	}
	if t.h2Config != nil && t.h2Config.HTTP2Config().StrictMaxConcurrentRequests {
		return true
	}
	return false
}

// poolMarkDirtyLocked marks a pool entry as being in need of reconciliation.
func (t *Transport) poolMarkDirtyLocked(ent *poolEntry) {
	if !t.pool.dirty.contains(ent) {
		t.pool.dirty.pushBack(ent)
	}
}

func (t *Transport) poolAddWaiterLocked(ent *poolEntry, w *poolWaiter) {
	ent.waiters.pushBack(w)
	t.poolMarkDirtyLocked(ent)
}

func (t *Transport) poolRemoveWaiterLocked(ent *poolEntry, w *poolWaiter) {
	ent.waiters.remove(w)
	t.poolMarkDirtyLocked(ent)
}

func (t *Transport) poolDeliverAndRemoveWaiterLocked(ent *poolEntry, w *poolWaiter, pconn *poolConn, err error) {
	w.pconn = pconn
	w.err = err
	t.poolRemoveWaiterLocked(ent, w)
	w.donec <- struct{}{}
}

func (t *Transport) poolRemoveDialLocked(ent *poolEntry, w *poolWaiter) {
	// Removing a dial marks the entry as dirty because it may free up a slot for
	// another dial to start.
	// In contrast, adding a dial is something that happens while reconciling a dirty entry.
	ent.dials.remove(w)
	t.poolMarkDirtyLocked(ent)
}

// poolGetEntryLocked returns the pool entry for key, creating it if necessary.
// t.pool.mu must be held.
func (t *Transport) poolGetEntryLocked(key connectMethodKey) *poolEntry {
	if t.pool.ents == nil {
		t.pool.ents = map[connectMethodKey]*poolEntry{}
		t.pool.lru.link = func(pconn *poolConn) *intrusiveLink[poolConn] {
			return &pconn.lruLink
		}
		t.pool.dirty.link = func(ent *poolEntry) *intrusiveLink[poolEntry] {
			return &ent.dirtyLink
		}
	}
	if ent, ok := t.pool.ents[key]; ok {
		return ent
	}
	ent := &poolEntry{
		key: key,
	}
	if !key.onlyH1 {
		p := t.protocols()
		switch key.scheme {
		case "http":
			// Only coalesce unencrypted dials if we're using HTTP/2.
			ent.coalesceDials = !p.HTTP1() && p.UnencryptedHTTP2()
		case "https":
			// Always coalesce encrypted dials, unless we're definitely using HTTP/1.
			ent.coalesceDials = p.HTTP2() || p.http3()
		}
	}
	ent.availableConns.link = func(pconn *poolConn) *intrusiveLink[poolConn] { return &pconn.availableLink }
	ent.waiters.link = func(w *poolWaiter) *intrusiveLink[poolWaiter] { return &w.waitLink }
	ent.dials.link = func(w *poolWaiter) *intrusiveLink[poolWaiter] { return &w.dialLink }
	t.pool.ents[key] = ent
	t.poolMarkDirtyLocked(ent)
	return ent
}

func (t *Transport) poolCloseConnLocked(pconn *poolConn, reason error) {
	if pconn.state == poolConnBroken {
		return // already closed
	}
	pconn.closing = true
	t.poolConnStateChangedLocked(pconn)
	go pconn.cc.CloseWithError(reason)
}

// poolCloseIdleConnections closes idle connections in the pool.
func (t *Transport) poolCloseIdleConnections() {
	var toClose []*poolConn
	t.withPoolLock(func() {
		t.pool.closeIdle = true
		for pconn := range t.pool.lru.all() {
			toClose = append(toClose, pconn)
		}
		for _, ent := range t.pool.ents {
			// Cancel dials in progress where the initiating request
			// is no longer waiting.
			for dial := range ent.dials.all() {
				if !ent.waiters.contains(dial) {
					dial.dialCancel()
				}
			}
			if ent.waiters.size == 0 && ent.altConn != nil {
				toClose = append(toClose, ent.altConn)
			}
		}
		for _, pconn := range toClose {
			pconn.closing = true
			t.poolConnStateChangedLocked(pconn)
		}
	})
	for _, pconn := range toClose {
		pconn.cc.CloseWithError(internal.ErrCloseIdleConnections)
	}
}

// poolConnStateHook is called when the state of a pooled connection changes.
func (t *Transport) poolConnStateHook(pconn *poolConn) error {
	var closeErr error
	t.withPoolLock(func() {
		if pconn.state == poolConnUnknown || pconn.state == poolConnBroken {
			// State hook invoked before pconn was added to the pool,
			// or after it was removed.
			return
		}
		closeErr = t.poolConnStateChangedLocked(pconn)
	})
	if closeErr != nil {
		pconn.cc.CloseWithError(closeErr)
	}
	return closeErr
}

// poolConnStateChangedLocked updates the pool in response to changes in pconn's state.
//
// If poolConnStateChangedLocked decides that an idle or broken connection should be closed,
// it returns a non-nil error; the caller must call pconn.cc.CloseWithError(closeErr)
// after releasing t.pool.mu.
func (t *Transport) poolConnStateChangedLocked(pconn *poolConn) (closeErr error) {
	oneIf := func(b bool) int {
		if b {
			return 1
		}
		return 0
	}

	var (
		ent          = pconn.poolEnt
		numInFlight  = pconn.cc.InFlight()
		numAvailable = pconn.cc.Available()
		pconnErr     = pconn.cc.Err()
	)

	if !pconn.closing && !pconn.isAlt() && numInFlight == 0 {
		// This connection is idle. Decide whether we're going to close it.
		switch {
		case pconnErr != nil:
			// Connection is unusable.
			closeErr = errConnBroken
		case pconn.exclusive:
			// Connection's single request has completed.
			closeErr = errCloseOnIdle
		case t.closeConnsOnIdle():
			// Never keep any idle connections.
			closeErr = errKeepAlivesDisabled
		case t.pool.closeIdle:
			// CloseIdleConnections was called, close newly-idle conns too.
			closeErr = errCloseIdle
		case ent.numIdle+oneIf(pconn.state != poolConnIdle) > t.maxIdleConnsPerHost():
			// MaxIdleConnsPerHost exceeded.
			closeErr = errTooManyIdleHost
		case numAvailable == 0:
			// Idle connection with no availability.
			// This is probably an HTTP/2 connection which has received
			// SETTINGS_MAX_CONCURRENT_STREAMS=0 from the server.
			// Just close it.
			// (Few servers send this state, and treating it as a GOAWAY is
			// both simple and reasonable behavior.)
			closeErr = errConnBroken
		}
		if closeErr != nil {
			pconn.closing = true
		}
	}

	var newState poolConnState
	switch {
	case pconn.closing:
		newState = poolConnBroken
	case pconn.isAlt():
		newState = poolConnAlt
	case pconn.exclusive: // inFlight > 0 ensured above
		newState = poolConnExclusive
	case pconnErr != nil: // inFlight > 0 ensured above
		newState = poolConnDraining
	case numInFlight > 0:
		newState = poolConnActive
	default:
		newState = poolConnIdle
	}

	prevState := pconn.state
	pconn.state = newState

	if newState != prevState {
		// Add/remove from the pool as a whole.
		const accountedStates = poolConnIdle | poolConnActive | poolConnExclusive | poolConnAlt
		ent.numConns -= oneIf(prevState.is(accountedStates))
		ent.numConns += oneIf(newState.is(accountedStates))

		if pconn.cc.Proto() == 2 {
			const sharedH2States = poolConnIdle | poolConnActive
			ent.sharedH2 -= oneIf(prevState.is(sharedH2States))
			ent.sharedH2 += oneIf(newState.is(sharedH2States))
		}

		switch prevState {
		case poolConnIdle:
			ent.numIdle--
			pconn.idleAt = time.Time{}
			t.pool.lru.remove(pconn)
		}
		switch newState {
		case poolConnIdle:
			ent.numIdle++
			pconn.idleAt = time.Now()
			t.pool.lru.pushBack(pconn)
			t.poolMaybeSetIdleTimerLocked()
		}
	}

	// Add/remove from available conns list.
	if pconn.isAlt() {
		if ent.altConn == nil && newState != poolConnBroken {
			ent.altConn = pconn
		} else if ent.altConn == pconn && newState == poolConnBroken {
			ent.altConn = nil
		}
	} else {
		available := newState.is(poolConnIdle|poolConnActive) && numAvailable > 0
		if available != ent.availableConns.contains(pconn) {
			if available {
				ent.availableConns.pushBack(pconn)
			} else {
				ent.availableConns.remove(pconn)
			}
		}
	}

	t.poolMarkDirtyLocked(ent)
	return closeErr
}

func (t *Transport) idleConnTimeout() time.Duration {
	// Unlike most config settings, historically IdleConnTimeout prefers the
	// http2.Transport's setting over the http.Transport.
	if t.h2Config != nil {
		if timeout := t.h2Config.IdleConnTimeout(); timeout != 0 {
			return timeout
		}
	}
	return t.IdleConnTimeout
}

func (t *Transport) poolMaybeSetIdleTimerLocked() {
	if t.pool.idleTimerSet {
		return
	}
	idleConnTimeout := t.idleConnTimeout()
	if idleConnTimeout <= 0 {
		return
	}
	if t.pool.idleTimer != nil {
		t.pool.idleTimer.Reset(idleConnTimeout)
	} else {
		t.pool.idleTimer = time.AfterFunc(idleConnTimeout, func() {
			t.poolIdleTimeout()
		})
	}
	t.pool.idleTimerSet = true
}

// poolIdleTimeout is run by the pool's idle timer.
func (t *Transport) poolIdleTimeout() {
	t.withPoolLock(func() {
		t.pool.idleTimerSet = false
		timeout := t.idleConnTimeout()
		if timeout <= 0 {
			return
		}
		now := time.Now()
		for pconn := range t.pool.lru.all() {
			if expiry := pconn.idleAt.Add(timeout); expiry.After(now) {
				break
			}
			t.poolCloseConnLocked(pconn, errIdleConnTimeout)
		}
		if head := t.pool.lru.head; head != nil {
			t.pool.idleTimer.Reset(head.idleAt.Add(timeout).Sub(now))
			t.pool.idleTimerSet = true
		}
	})
}

// poolRemoveConn entirely removes pconn from the pool.
func (t *Transport) poolRemoveConn(pconn *poolConn) {
	t.withPoolLock(func() {
		pconn.closing = true
		t.poolConnStateChangedLocked(pconn)
	})
}

// poolGetConn gets a usable connection from the pool,
// dialing a new one if necessary.
func (t *Transport) poolGetConn(ctx context.Context, cm connectMethod, connClose bool) (*poolConn, httptrace.GotConnInfo, error) {
	getConnCancelCause := func(ctx context.Context) error {
		err := context.Cause(ctx)
		if err == errRequestCanceled {
			err = errRequestCanceledConn
		}
		return err
	}

	// If the request context is already canceled, don't go looking for a conn.
	if err := ctx.Err(); err != nil {
		return nil, {}, getConnCancelCause(ctx)
	}

	// An exclusive request needs a conn not shared with any other requests.
	exclusive := t.DisableKeepAlives || connClose

	var (
		ent    *poolEntry
		pconn  *poolConn
		info   httptrace.GotConnInfo
		waiter *poolWaiter
	)
	t.withPoolLock(func() {
		t.pool.closeIdle = false
		ent = t.poolGetEntryLocked(cm.key())
		if ent.waiters.size == 0 {
			// Fast path: If we can get an entry without waiting, do so.
			pconn, info = t.poolTryGetForEntryLocked(ent, exclusive)
			if pconn != nil {
				return
			}
		}
		// Enqueue for a conn.
		// When withPoolLock returns, it will start a dial for us if possible.
		// (It would also deliver an available conn if one existed,
		// but we just checked for one above and found nothing.)
		waiter = &poolWaiter{
			cm:        cm,
			ctx:       ctx,
			exclusive: exclusive,
			donec:     make(chan struct{}, 1),
		}
		t.poolAddWaiterLocked(ent, waiter)
	})
	if pconn != nil {
		return pconn, info, nil
	}

	select {
	case <-waiter.donec:
		// A result was delivered to us.
		return waiter.pconn, waiter.info, waiter.err
	case <-ctx.Done():
		// We have given up on waiting.
		t.withPoolLock(func() {
			if ent.waiters.contains(waiter) {
				t.poolRemoveWaiterLocked(ent, waiter)
			} else {
			}
		})
		if waiter.pconn != nil {
			// Context cancellation raced with a result being delivered.
			// There's no point in returning the conn (if any) we have just
			// received. Release it back to the pool.
			waiter.pconn.cc.Release()
			t.poolConnStateHook(waiter.pconn) // not called by Release
		}
		return nil, {}, getConnCancelCause(ctx)
	}
}

// poolTryGetForEntryLocked attempts to fetch an idle connection from ent.
// It returns nil if no usable connection was found.
func (t *Transport) poolTryGetForEntryLocked(ent *poolEntry, exclusive bool) (*poolConn, httptrace.GotConnInfo) {
	// If IdleConnTimeout is set, calculate the oldest
	// poolConn.idleAt time we're willing to use a cached idle
	// conn.
	var oldTime time.Time
	if timeout := t.idleConnTimeout(); timeout > 0 {
		oldTime = time.Now().Add(-timeout)
	}
	if ent.altConn != nil {
		return ent.altConn, {} // altConn will call GotConnInfo itself
	}
	for pconn := range ent.availableConns.backward() {
		// See whether this connection has been idle too long, considering
		// only the wall time (the Round(0)), in case this is a laptop or VM
		// coming out of suspend with previously cached idle connections.
		if !oldTime.IsZero() && !pconn.idleAt.IsZero() && pconn.idleAt.Round(0).Before(oldTime) {
			// Async cleanup. Launch in its own goroutine (as if a
			// time.AfterFunc called it).
			t.poolCloseConnLocked(pconn, errIdleConnTimeout)
			continue
		}
		if exclusive && pconn.state != poolConnIdle {
			// Exclusive requests can't share a conn, so they need an Idle conn.
			continue
		}
		wasIdleAt := pconn.idleAt
		reserved := pconn.cc.Reserve() == nil
		if reserved && exclusive && !pconn.isAlt() {
			pconn.exclusive = true
		}
		if closeErr := t.poolConnStateChangedLocked(pconn); closeErr != nil {
			// Unexpectedly broken connection.
			if reserved {
				pconn.cc.Release()
			}
			go pconn.cc.CloseWithError(closeErr)
			continue
		}
		if !reserved {
			// We couldn't reserve a slot, despite this conn being on
			// the available list. We're probably racing a state change.
			continue
		}
		info := httptrace.GotConnInfo{
			Conn: pconn.cc.NetConn(),
		}
		if !wasIdleAt.IsZero() {
			info.WasIdle = true
			info.IdleTime = time.Since(wasIdleAt)
		}
		return pconn, info
	}
	return nil, {}
}

// poolDialForEntry dials a new connection.
//
// Note that dials always use a context derived from a request context,
// although a request may use a connection that was dialed with a different request's context.
func (t *Transport) poolDialForEntry(ctx context.Context, ent *poolEntry, waiter *poolWaiter) {
	pconn := &poolConn{
		poolEnt: ent,
	}
	cc, dialErr := t.dialConn(ctx, waiter.cm, false, func() error {
		return t.poolConnStateHook(pconn)
	})
	var (
		deliverConn *poolConn
		closeErr    error
	)
	t.withPoolLock(func() {
		deliverConn, closeErr = t.poolDialDoneLocked(ent, waiter, pconn, cc, dialErr)
		var deliverErr error
		if deliverConn == nil {
			deliverErr = closeErr
		}
		if ent.waiters.contains(waiter) {
			t.poolDeliverAndRemoveWaiterLocked(ent, waiter, deliverConn, deliverErr)
		}
		if dialErr != nil && ctx.Err() == nil && ent.coalesceDials && !waiter.exclusive {
			// Dialing failed. Deliver the error to waiters that coalesced with it.
			for w := range ent.waiters.all() {
				if !w.dialed {
					t.poolDeliverAndRemoveWaiterLocked(ent, w, nil, dialErr)
				}
			}
		}
	})
	if cc != nil && closeErr != nil {
		cc.CloseWithError(closeErr)
	}
}

func (t *Transport) poolDialDoneLocked(ent *poolEntry, waiter *poolWaiter, pconn *poolConn, cc genericClientConn, err error) (deliverConn *poolConn, closeErr error) {
	t.poolRemoveDialLocked(ent, waiter)

	if err != nil {
		return nil, err
	}

	pconn.cc = cc

	if ent.altConn != nil {
		// We only need one alt-conn (which represents a connection pool,
		// rather than a connection). If we already have one, keep using
		// it and throw this new conn away.
		return ent.altConn, errTooManyIdleHost
	}

	if cc.Proto() == 1 {
		// We've dialed an HTTP/1 connection, don't try coalescing
		// future dials.
		ent.coalesceDials = false
	}

	if pconn.isAlt() && t.closeConnsOnIdle() {
		// Don't pool alt-conns when idle conn retention is disabled.
		// The alt-conn is its own pool, but that pool is always empty when retention
		// is disabled. (This is not quite the entire story: The alt-conn can contain
		// a connection if a waiter gives up before the connection is added to the pool.
		// This connection will be used via Transport.alternateRoundTripper, not from
		// here. All of this is mercifully less relevant in the new merged pool.)
		if !ent.waiters.contains(waiter) {
			return nil, errKeepAlivesDisabled
		}
		pconn.closing = true
	}

	if !ent.waiters.contains(waiter) {
		// The request that requested this dial is no longer waiting.
		// See if we should add the conn to the pool for someone else to use,
		// or abandon it.
		return nil, t.poolConnStateChangedLocked(pconn)
	}

	if err := cc.Reserve(); err != nil {
		// Reserving a slot on a new connection should not fail.
		// Deliver the error to the waiter to unblock them.
		return nil, err
	}
	if waiter.exclusive && !pconn.isAlt() {
		pconn.exclusive = true
	}
	if err := t.poolConnStateChangedLocked(pconn); err != nil {
		return nil, err
	}
	waiter.info = httptrace.GotConnInfo{
		Conn:    cc.NetConn(),
		Reused:  false,
		WasIdle: false,
	}
	return pconn, nil
}

func (t *Transport) closeConnsOnIdle() bool {
	return t.DisableKeepAlives || t.MaxIdleConnsPerHost < 0 || t.MaxIdleConns < 0
}
