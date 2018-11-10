// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bridge package to expose http internals to tests in the http_test
// package.

package http

import (
	"context"
	"net"
	"sort"
	"sync"
	"testing"
	"time"
)

var (
	DefaultUserAgent                  = defaultUserAgent
	NewLoggingConn                    = newLoggingConn
	ExportAppendTime                  = appendTime
	ExportRefererForURL               = refererForURL
	ExportServerNewConn               = (*Server).newConn
	ExportCloseWriteAndWait           = (*conn).closeWriteAndWait
	ExportErrRequestCanceled          = errRequestCanceled
	ExportErrRequestCanceledConn      = errRequestCanceledConn
	ExportErrServerClosedIdle         = errServerClosedIdle
	ExportServeFile                   = serveFile
	ExportScanETag                    = scanETag
	ExportHttp2ConfigureServer        = http2ConfigureServer
	Export_shouldCopyHeaderOnRedirect = shouldCopyHeaderOnRedirect
	Export_writeStatusLine            = writeStatusLine
)

func init() {
	// We only want to pay for this cost during testing.
	// When not under test, these values are always nil
	// and never assigned to.
	testHookMu = new(sync.Mutex)
}

var (
	SetEnterRoundTripHook = hookSetter(&testHookEnterRoundTrip)
	SetRoundTripRetried   = hookSetter(&testHookRoundTripRetried)
)

func SetReadLoopBeforeNextReadHook(f func()) {
	testHookMu.Lock()
	defer testHookMu.Unlock()
	unnilTestHook(&f)
	testHookReadLoopBeforeNextRead = f
}

// SetPendingDialHooks sets the hooks that run before and after handling
// pending dials.
func SetPendingDialHooks(before, after func()) {
	unnilTestHook(&before)
	unnilTestHook(&after)
	testHookPrePendingDial, testHookPostPendingDial = before, after
}

func SetTestHookServerServe(fn func(*Server, net.Listener)) { testHookServerServe = fn }

func NewTestTimeoutHandler(handler Handler, ch <-chan time.Time) Handler {
	return &timeoutHandler{
		handler:     handler,
		testTimeout: ch,
		// (no body)
	}
}

func ResetCachedEnvironment() {
	httpProxyEnv.reset()
	httpsProxyEnv.reset()
	noProxyEnv.reset()
}

func (t *Transport) NumPendingRequestsForTesting() int {
	t.reqMu.Lock()
	defer t.reqMu.Unlock()
	return len(t.reqCanceler)
}

func (t *Transport) IdleConnKeysForTesting() (keys []string) {
	keys = make([]string, 0)
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	for key := range t.idleConn {
		keys = append(keys, key.String())
	}
	sort.Strings(keys)
	return
}

func (t *Transport) IdleConnKeyCountForTesting() int {
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	return len(t.idleConn)
}

func (t *Transport) IdleConnStrsForTesting() []string {
	var ret []string
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	for _, conns := range t.idleConn {
		for _, pc := range conns {
			ret = append(ret, pc.conn.LocalAddr().String()+"/"+pc.conn.RemoteAddr().String())
		}
	}
	sort.Strings(ret)
	return ret
}

func (t *Transport) IdleConnStrsForTesting_h2() []string {
	var ret []string
	noDialPool := t.h2transport.ConnPool.(http2noDialClientConnPool)
	pool := noDialPool.http2clientConnPool

	pool.mu.Lock()
	defer pool.mu.Unlock()

	for k, cc := range pool.conns {
		for range cc {
			ret = append(ret, k)
		}
	}

	sort.Strings(ret)
	return ret
}

func (t *Transport) IdleConnCountForTesting(cacheKey string) int {
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	for k, conns := range t.idleConn {
		if k.String() == cacheKey {
			return len(conns)
		}
	}
	return 0
}

func (t *Transport) IdleConnChMapSizeForTesting() int {
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	return len(t.idleConnCh)
}

func (t *Transport) IsIdleForTesting() bool {
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	return t.wantIdle
}

func (t *Transport) RequestIdleConnChForTesting() {
	t.getIdleConnCh(connectMethod{nil, "http", "example.com"})
}

func (t *Transport) PutIdleTestConn() bool {
	c, _ := net.Pipe()
	return t.tryPutIdleConn(&persistConn{
		t:        t,
		conn:     c,                   // dummy
		closech:  make(chan struct{}), // so it can be closed
		cacheKey: connectMethodKey{"", "http", "example.com"},
	}) == nil
}

// All test hooks must be non-nil so they can be called directly,
// but the tests use nil to mean hook disabled.
func unnilTestHook(f *func()) {
	if *f == nil {
		*f = nop
	}
}

func hookSetter(dst *func()) func(func()) {
	return func(fn func()) {
		unnilTestHook(&fn)
		*dst = fn
	}
}

func ExportHttp2ConfigureTransport(t *Transport) error {
	t2, err := http2configureTransport(t)
	if err != nil {
		return err
	}
	t.h2transport = t2
	return nil
}

func (s *Server) ExportAllConnsIdle() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for c := range s.activeConn {
		st, ok := c.curState.Load().(ConnState)
		if !ok || st != StateIdle {
			return false
		}
	}
	return true
}

func (r *Request) WithT(t *testing.T) *Request {
	return r.WithContext(context.WithValue(r.Context(), tLogKey{}, t.Logf))
}
