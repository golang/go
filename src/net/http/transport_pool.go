// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"context"
	"maps"
	"slices"
	"sync"
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
// Pool entries have a channel-based lock which permits get-connection requests
// to wait for available connections and/or dial slots; see clientConnPoolEntry for details.
//
// Lock ordering is (first to last): clientConnPoolEntry >> clientConnPool >> persistConn.
//
// Locking and unlocking entries is a channel operation, so we take some care to minimize
// the number of lock operations required in the hot path.
//
// The pool and entry locks should never be held during blocking operations
// (such as dials or closing a net.Conn).
type clientConnPool struct {
	mu        sync.Mutex
	ents      map[connectMethodKey]*clientConnPoolEntry
	lru       intrusiveList[poolConn]
	closeIdle bool
}

type poolConn struct {
	cc genericClientConn

	// Guarded by clientConnPool.mu.
	lruLink intrusiveLink[poolConn]

	// Guarded by clientConnPoolEntry's lock.
	poolEnt  *clientConnPoolEntry
	isInPool bool      // set when counted in pool entry's numConns
	idleAt   time.Time // time it last became idle
}

func (pc *poolConn) isBroken() bool {
	return pc.cc.Err() != nil
}

type clientConnPoolEntry struct {
	key connectMethodKey

	// These three channels (each with a 1-element buffer) act as a lock/condition variable.
	//
	//   - if the entry is locked, no channel contains a value; otherwise
	//   - if idleConns contains a usable conn, cangetc contains a value; otherwise
	//   - if a new conn may be dialed, candialc contains a value; otherwise
	//   - lockc contains a value.
	//
	// Lock ordering: Only lock one entry at a time. Lock entries before the pool mutex.
	cangetc  chan struct{}
	candialc chan struct{}
	lockc    chan struct{}

	// Idle (available for use) connections, most recently used last.
	idleConns []*poolConn

	// Connection with IsAlt() == true.
	//
	// This is historically a pseudo-"connection" backed by an http2.Transport,
	// which manages its own connection pool. Lack of visibility into this pool
	// causes a variety of bugs, such as connections-per-host tracking not accounting
	// for individual HTTP/2 connections.
	//
	// This may also be an HTTP/3 connection.
	//
	// TODO: Improve management of HTTP/2 and HTTP/3 connections.
	altConn *poolConn

	// Total connections, idle and otherwise.
	numConns int

	// Requests waiting for a connection.
	numWaiters int

	// Dials in progress.
	dials intrusiveList[poolDial]

	// Idle timer, fires when we should next check to see if conns
	// should be closed for idleness.
	idleTimer    *time.Timer
	idleTimerSet bool

	// Set when the entry is removed from its pool.
	removed bool
}

type poolDial struct {
	ent      *clientConnPoolEntry
	cm       connectMethod
	cancel   context.CancelFunc // call to cancel the dial
	resultc  chan pconnOrError  // sends the dial result to the requester
	donec    <-chan struct{}    // closed when the requester no longer needs the dial result
	link     intrusiveLink[poolDial]
	postDial func()
}

// lock unconditionally locks the entry.
func (ent *clientConnPoolEntry) lock() {
	select {
	case <-ent.lockc:
	case <-ent.cangetc:
	case <-ent.candialc:
	}
}

// unlock unlocks the entry.
//
// If the entry is no longer in use, unlock removes it from the pool.
// The pool mutex must not be held when calling unlock.
func (ent *clientConnPoolEntry) unlock(tr *Transport) {
	if !ent.removed && ent.numConns == 0 && ent.numWaiters == 0 && ent.dials.size == 0 && len(ent.idleConns) == 0 {
		// This entry is no longer in use; remove it from the pool.
		ent.removed = true
		if ent.idleTimer != nil {
			ent.idleTimer.Stop()
		}
		tr.pool.mu.Lock()
		delete(tr.pool.ents, ent.key)
		tr.pool.mu.Unlock()
	}
	switch {
	case len(ent.idleConns) > 0 || ent.altConn != nil:
		ent.cangetc <- struct{}{}
	case tr.MaxConnsPerHost <= 0 || ent.numConns+ent.dials.size < tr.MaxConnsPerHost:
		ent.candialc <- struct{}{}
	default:
		ent.lockc <- struct{}{}
	}
}

// registerConnLocked adds pconn to the entry's bookkeeping.
func (ent *clientConnPoolEntry) registerConnLocked(pconn *poolConn) {
	if pconn.isBroken() || pconn.isInPool {
		return
	}
	pconn.isInPool = true
	ent.numConns++
}

// unregisterConnLocked disassociates pconn from the entry.
func (ent *clientConnPoolEntry) unregisterConnLocked(pconn *poolConn) {
	if !pconn.isInPool {
		// This can happen if a newly-created connection breaks before we finish
		// registering it with the pool.
		return
	}
	pconn.isInPool = false
	ent.numConns--
}

// deleteIdleLocked deletes every element in idleConns that matches f.
func (ent *clientConnPoolEntry) deleteIdleLocked(t *Transport, f func(*poolConn) bool) {
	ent.idleConns = slices.DeleteFunc(ent.idleConns, func(pconn *poolConn) bool {
		delete := f(pconn)
		if delete {
			ent.unregisterConnLocked(pconn)
			t.removeFromIdleLRU(pconn)
		}
		return delete
	})
}

// getLockedPoolEntry returns the pool entry for key, creating it if necessary.
// The entry will be locked.
//
// If clearCloseIdle is set, pool.closeIdle is set to false.
// (This isn't especially principled from an abstraction perspective,
// but it lets us get the entry and update closeIdle under one mutex acquisition.)
func (t *Transport) getLockedPoolEntry(key connectMethodKey, clearCloseIdle bool) *clientConnPoolEntry {
	getEnt := func() (*clientConnPoolEntry, bool) {
		t.pool.mu.Lock()
		defer t.pool.mu.Unlock()
		if clearCloseIdle {
			t.pool.closeIdle = false
		}
		if t.pool.ents == nil {
			t.pool.ents = map[connectMethodKey]*clientConnPoolEntry{}
			t.pool.lru.link = func(pconn *poolConn) *intrusiveLink[poolConn] {
				return &pconn.lruLink
			}
		}
		if ent, ok := t.pool.ents[key]; ok {
			return ent, true
		}
		ent := &clientConnPoolEntry{
			key:      key,
			lockc:    make(chan struct{}, 1),
			cangetc:  make(chan struct{}, 1),
			candialc: make(chan struct{}, 1),
		}
		ent.dials.link = func(dial *poolDial) *intrusiveLink[poolDial] { return &dial.link }
		t.pool.ents[key] = ent
		return ent, false
	}
	for {
		ent, needLock := getEnt()
		if needLock {
			ent.lock()
		}
		if !ent.removed {
			return ent
		}
		// This entry was removed from the pool in the time between the lookup
		// and us locking it. Try again.
		ent.unlock(t)
	}
}

// poolCopyEntries returns the current set of pool entries as a slice.
func (t *Transport) poolCopyEntries() []*clientConnPoolEntry {
	t.pool.mu.Lock()
	defer t.pool.mu.Unlock()
	return slices.Collect(maps.Values(t.pool.ents))
}

// poolCloseIdleConnections closes idle connections in the pool.
func (t *Transport) poolCloseIdleConnections() {
	t.pool.mu.Lock()
	t.pool.closeIdle = true
	t.pool.mu.Unlock()

	for _, ent := range t.poolCopyEntries() {
		var toClose []*poolConn
		func() {
			ent.lock()
			defer ent.unlock(t)
			if ent.altConn != nil && ent.numWaiters == 0 {
				ent.unregisterConnLocked(ent.altConn)
				toClose = append(toClose, ent.altConn)
				ent.altConn = nil
			}
			ent.deleteIdleLocked(t, func(pconn *poolConn) bool {
				toClose = append(toClose, pconn)
				return true
			})

			// Cancel dials in progress where the initiating request
			// is no longer waiting.
			for dial := range ent.dials.all() {
				select {
				case <-dial.donec:
					dial.cancel()
				default:
				}
			}
		}()
		for _, pconn := range toClose {
			pconn.cc.CloseWithError(errCloseIdleConns)
		}
	}
}

// poolPutIdleConn indicates that a connection has become idle and may accept new requests.
func (t *Transport) poolPutIdleConn(pconn *poolConn) error {
	if pconn == nil || pconn.poolEnt == nil {
		return errNotInPool
	}
	ent := pconn.poolEnt
	if pconn.cc.IsAlt() {
		// HTTP/2 and HTTP/3 connections should never appear here.
		// Ignore them if they do.
		return nil
	}
	err, needExpire := func() (_ error, needExpire bool) {
		ent.lock()
		defer ent.unlock(t)
		return t.poolPutIdleConnLocked(ent, pconn)
	}()
	if needExpire {
		t.expireIdleLRU()
	}
	return err
}

// poolPutIdleConnLocked places pconn in its entry's idle list.
// The entry must be locked.
//
// It returns an error if pconn should be closed.
//
// It returns needExpire=true if the caller should call t.expireIdleLRU
// after dropping the entry lock. (We check inside poolPutIdleConnLocked to avoid
// a needless pool mutex acquisition in the common case when expiry is not needed.)
func (t *Transport) poolPutIdleConnLocked(ent *clientConnPoolEntry, pconn *poolConn) (_ error, needExpire bool) {
	t.pool.mu.Lock()
	defer t.pool.mu.Unlock()

	if pconn.isBroken() || !pconn.isInPool {
		return errConnBroken, false
	}
	if t.pool.lru.contains(pconn) {
		// This conn is already idle.
		// Move it to the back of the idle list.
		t.pool.lru.remove(pconn)
	} else {
		// This conn is entering the idle list.
		if t.keepAlivesDisabled() {
			return errKeepAlivesDisabled, false
		}
		if len(ent.idleConns) >= t.maxIdleConnsPerHost() {
			return errTooManyIdleHost, false
		}
		if pconn.cc.IsAlt() && ent.altConn != nil {
			// We already have a reusable, shared connection;
			// discard this one.
			return errTooManyIdleHost, false
		}
		if t.pool.closeIdle {
			return errCloseIdle, false
		}
		if !pconn.cc.IsAlt() {
			ent.idleConns = append(ent.idleConns, pconn)
		} else {
			ent.altConn = pconn
		}
	}

	if !pconn.cc.IsAlt() {
		t.pool.lru.pushBack(pconn)
	}

	// Arm the idle timer (if needed).
	pconn.idleAt = time.Now()
	if t.IdleConnTimeout > 0 && !ent.idleTimerSet && !pconn.cc.IsAlt() {
		if ent.idleTimer != nil {
			ent.idleTimer.Reset(t.IdleConnTimeout)
		} else {
			ent.idleTimer = time.AfterFunc(t.IdleConnTimeout, func() {
				t.poolIdleTimeout(ent)
			})
		}
		ent.idleTimerSet = true
	}
	needExpire = t.MaxIdleConns != 0 && t.pool.lru.size > t.MaxIdleConns
	return nil, needExpire
}

func (t *Transport) poolIdleTimeout(ent *clientConnPoolEntry) {
	ent.lock()
	defer ent.unlock(t)
	ent.idleTimerSet = false
	timeout := t.IdleConnTimeout
	if timeout <= 0 {
		return
	}
	now := time.Now()
	var next time.Time
	ent.deleteIdleLocked(t, func(pconn *poolConn) bool {
		if expiry := pconn.idleAt.Add(timeout); !expiry.After(now) {
			go pconn.cc.CloseWithError(errIdleConnTimeout)
			return true
		} else if next.IsZero() || expiry.Before(next) {
			next = expiry
		}
		return false
	})
	if !next.IsZero() {
		ent.idleTimer.Reset(next.Sub(now))
		ent.idleTimerSet = true
	}
}

// poolRemoveConn entirely removes pconn from the pool.
func (t *Transport) poolRemoveConn(pconn *poolConn) {
	if pconn == nil || pconn.poolEnt == nil {
		return
	}
	ent := pconn.poolEnt
	ent.lock()
	defer ent.unlock(t)
	if ent.altConn == pconn {
		ent.altConn = nil
	}
	if idx := slices.Index(ent.idleConns, pconn); idx >= 0 {
		ent.idleConns = slices.Delete(ent.idleConns, idx, idx+1)
		t.removeFromIdleLRU(pconn)
	}
	ent.unregisterConnLocked(pconn)
}

type pconnOrError struct {
	pconn *poolConn
	err   error
}

// poolGetConn gets a usable connection from the pool,
// dialing a new one if necessary.
func (t *Transport) poolGetConn(ctx context.Context, cm connectMethod) (*poolConn, error) {
	getConnCancelCause := func(ctx context.Context) error {
		err := context.Cause(ctx)
		if err == errRequestCanceled {
			err = errRequestCanceledConn
		}
		return err
	}

	if testHookPrePendingDial != nil {
		testHookPrePendingDial()
	}
	postDialHook := testHookPostPendingDial
	defer func() {
		if postDialHook != nil {
			postDialHook()
		}
	}()

	// If the request context is already canceled, don't go looking for a conn.
	if err := ctx.Err(); err != nil {
		return nil, getConnCancelCause(ctx)
	}

	// Fast path: Check to see if there's a usable conn.
	const clearCloseIdle = true
	ent, pconn := func() (*clientConnPoolEntry, *poolConn) {
		ent := t.getLockedPoolEntry(cm.key(), clearCloseIdle)
		defer ent.unlock(t)
		pconn := t.poolTryGetForEntryLocked(ent)
		if pconn == nil {
			// Increment the waiters count to keep this pool entry alive
			// while we're using it.
			ent.numWaiters++
		}
		return ent, pconn
	}()
	if pconn != nil {
		return pconn, nil
	}

	// Remove ourselves from the waiters before returning.
	// Running this in a defer ensures it happens after any later defers.
	// In particular, we must close getdonec before trying to lock the entry
	// to decrement numWaiters, to avoid a deadlock race when the request context
	// expires at the same time as a dial finishes.
	waiting := true
	defer func() {
		if !waiting {
			return
		}
		ent.lock()
		defer ent.unlock(t)
		ent.numWaiters--
	}()

	var (
		candialc    = ent.candialc
		dialresultc chan pconnOrError
		getdonec    chan struct{}
	)
	for {

		select {
		case <-ent.cangetc:
			// Reading from cangetc locks the entry, and indicates that there is
			// an available connection in the idle list.
			pconn := func() *poolConn {
				defer ent.unlock(t)
				pconn := t.poolTryGetForEntryLocked(ent)
				if pconn != nil {
					ent.numWaiters--
					waiting = false
				}
				return pconn
			}()
			if pconn != nil {
				return pconn, nil
			}
			// We didn't find a usable connection. Most likely we lost a race against
			// a closing connection (the conn is now dead, but it hadn't finished
			// removing itself from the pool). Go back and try again.
		case <-candialc:
			// Reading from candialc locks the entry, and indicates that we may
			// start a new dial.
			func() {
				defer ent.unlock(t)
				// Create a context for the dial which inherits the
				// request context's values, but not its cancellation
				// (since the dial may outlive the request).
				//
				// Note that we only cancel this context to cancel the dial.
				// If the dial finishes, we leave it uncanceled since cancelling
				// it can (for some user-provided DialContext functions)
				// close the connection.
				dialCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
				dialresultc = make(chan pconnOrError)
				getdonec = make(chan struct{})
				dial := &poolDial{
					ent:      ent,
					cm:       cm,
					cancel:   cancel,
					resultc:  dialresultc,
					donec:    getdonec,
					postDial: postDialHook,
				}
				ent.dials.pushBack(dial)
				postDialHook = nil // will be called by poolDialForEntry
				go t.poolDialForEntry(dialCtx, dial)
				candialc = nil // don't dial again
			}()
			defer close(getdonec)
		case res := <-dialresultc:
			// A dial we started has completed and provided us with
			// either a connection, or a dial failure.
			//
			// poolDialForEntry decrements numWaiters in this case
			// (avoiding a redundant lock on the entry).
			waiting = false
			return res.pconn, res.err
		case <-ctx.Done():
			// The request has timed out.
			return nil, getConnCancelCause(ctx)
		}
	}
}

// poolTryGetForEntryLocked attempts to fetch an idle connection from ent.
// ent must be locked.
// It returns nil if no usable connection was found.
func (t *Transport) poolTryGetForEntryLocked(ent *clientConnPoolEntry) *poolConn {
	if ent.altConn != nil {
		// If we've got an HTTP/2 or HTTP/3 connection, use it.
		return ent.altConn
	}

	// If IdleConnTimeout is set, calculate the oldest
	// poolConn.idleAt time we're willing to use a cached idle
	// conn.
	var oldTime time.Time
	if t.IdleConnTimeout > 0 {
		oldTime = time.Now().Add(-t.IdleConnTimeout)
	}

	for len(ent.idleConns) > 0 {
		pconn := ent.idleConns[len(ent.idleConns)-1]

		// See whether this connection has been idle too long, considering
		// only the wall time (the Round(0)), in case this is a laptop or VM
		// coming out of suspend with previously cached idle connections.
		tooOld := !oldTime.IsZero() && pconn.idleAt.Round(0).Before(oldTime)
		if tooOld {
			// Async cleanup. Launch in its own goroutine (as if a
			// time.AfterFunc called it).
			go pconn.cc.CloseWithError(errIdleConnTimeout)
		}
		if pconn.isBroken() || tooOld {
			// If either persistConn.readLoop has marked the connection
			// broken, but Transport.removeIdleConn has not yet removed it
			// from the idle list, or if this persistConn is too old (it was
			// idle too long), then ignore it and look for another. In both
			// cases it's already in the process of being closed.
			idx := len(ent.idleConns) - 1
			ent.idleConns = slices.Delete(ent.idleConns, idx, idx+1)
			ent.unregisterConnLocked(pconn)
			t.removeFromIdleLRU(pconn)
			continue
		}
		// HTTP/1: only one client can use pconn.
		// Remove it from the list.
		idx := len(ent.idleConns) - 1
		ent.idleConns = slices.Delete(ent.idleConns, idx, idx+1)
		t.removeFromIdleLRU(pconn)
		return pconn
	}
	return nil
}

// poolDialForEntry dials a new connection.
//
// Note that dials always use a context derived from a request context,
// although a request may use a connection that was dialed with a different request's context.
func (t *Transport) poolDialForEntry(ctx context.Context, dial *poolDial) {
	ent := dial.ent
	if dial.postDial != nil {
		defer dial.postDial()
	}

	pconn := &poolConn{
		poolEnt: ent,
	}
	cc, err := t.dialConn(ctx, dial.cm, false, pconn, nil)
	if err != nil {
		pconn = nil
	} else {
		pconn.cc = cc
	}
	needClose, needExpire := func() (needClose, needExpire bool) {
		ent.lock()
		defer ent.unlock(t)

		// The dial is complete: Remove the dial from the entry and register
		// the new connection (if any).
		ent.dials.remove(dial)
		if err == nil {
			ent.registerConnLocked(pconn)
		}

		// HTTP/2 and HTTP/3 connections can be shared.
		// Add to the idle connection pool before trying to deliver to w.
		deliver := pconnOrError{pconn: pconn, err: err}
		if err == nil && pconn.cc.IsAlt() {
			err, needExpire = t.poolPutIdleConnLocked(ent, pconn)
			if err != nil {
				// This connection should not go into the idle list.
				//
				// Note: Setting needClose replicates some odd behaviors
				// from the earlier pool implementation. If this is an HTTP/2
				// connection, calling pconn.close does not actually close
				// anything: The "connection" is a reference to the HTTP/2
				// transport, which manages its own connection pool.
				// If this is an HTTP/3 connection, pconn.close does close
				// the connection and may immediately break it while a RoundTrip
				// is in progress.
				//
				// TODO: Fix the above behaviors.
				ent.unregisterConnLocked(pconn)
				needClose = true
				if ent.altConn != nil {
					// We already have a sharable connection in the pool.
					// Keep the existing one, and discard this new one.
					//
					// Try to deliver the existing conn to poolGetConn.
					// We must always deliver a result to poolGetConn
					// to avoid it potentially hanging forever waiting
					// for a pooled conn. (In this branch, we observe that
					// there is currently a pooled conn, but it could become
					// unusable between now and poolGetConn trying to
					// retrieve it.)
					deliver = pconnOrError{pconn: ent.altConn}
				}
			}
		}

		// Try to deliver the result back to poolGetConn.
		//
		// Note that we're doing this with the entry still locked, so poolGetConn
		// needs to take care not to block on locking the entry until after it
		// closes donec. poolGetConn is free to lock the entry in a non-blocking fashion.
		delivered := false
		select {
		case dial.resultc <- deliver:
			ent.numWaiters--
			delivered = true
		case <-dial.donec:
		}
		if err != nil || delivered || needClose || pconn.cc.IsAlt() {
			// Either an error, or an unsharable HTTP/1 connection which was delivered
			// to the waiter, or a shared HTTP/2 or /3 connection already in the pool.
			return needClose, false
		}

		err, needExpire = t.poolPutIdleConnLocked(ent, pconn)
		if err != nil {
			// This connection should not go into the idle list.
			ent.unregisterConnLocked(pconn)
			return true, false
		}
		return false, needExpire
	}()
	if needClose {
		pconn.cc.CloseWithError(err)
	}
	if needExpire {
		t.expireIdleLRU()
	}
}

// removeFromIdleLRU removes pconn from the idle-conns LRU.
func (t *Transport) removeFromIdleLRU(pconn *poolConn) {
	t.pool.mu.Lock()
	defer t.pool.mu.Unlock()
	t.pool.lru.remove(pconn)
}

// expireIdleLRU removes idle conns from the pool until the number of idle conns
// is less than or equal to MaxIdleConns.
func (t *Transport) expireIdleLRU() {
	t.pool.mu.Lock()
	defer t.pool.mu.Unlock()
	if t.MaxIdleConns == 0 {
		return
	}
	maxIdle := max(t.MaxIdleConns, 0)
	// expireEntry expires conns from the head of the LRU which belong to ent.
	// It locks the entry and pool.
	expireEntry := func(ent *clientConnPoolEntry) {
		ent.lock()
		defer ent.unlock(t)
		t.pool.mu.Lock()
		defer t.pool.mu.Unlock()
		// Expire conns for the entry we currently have locked.
		// It's possible the head of the LRU changed while we were juggling locks;
		// that's okay. Worst case, we loop back and try again with the new head.
		for t.pool.lru.size > maxIdle {
			pconn := t.pool.lru.head
			if pconn.poolEnt != ent {
				break
			}
			if idx := slices.Index(ent.idleConns, pconn); idx >= 0 {
				ent.idleConns = slices.Delete(ent.idleConns, idx, idx+1)
				ent.unregisterConnLocked(pconn)
				go pconn.cc.CloseWithError(errTooManyIdle)
			}
			t.pool.lru.remove(pconn)
		}
	}
	for t.pool.lru.size > maxIdle {
		func() {
			// Lock ordering is ent >> pool, so we need to drop the pool mutex
			// before locking the entry.
			ent := t.pool.lru.head.poolEnt
			t.pool.mu.Unlock()
			defer t.pool.mu.Lock()
			expireEntry(ent)
		}()
	}
}
