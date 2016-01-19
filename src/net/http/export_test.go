// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bridge package to expose http internals to tests in the http_test
// package.

package http

import (
	"net"
	"sync"
	"time"
)

var (
	DefaultUserAgent              = defaultUserAgent
	NewLoggingConn                = newLoggingConn
	ExportAppendTime              = appendTime
	ExportRefererForURL           = refererForURL
	ExportServerNewConn           = (*Server).newConn
	ExportCloseWriteAndWait       = (*conn).closeWriteAndWait
	ExportErrRequestCanceled      = errRequestCanceled
	ExportErrRequestCanceledConn  = errRequestCanceledConn
	ExportServeFile               = serveFile
	ExportHttp2ConfigureTransport = http2ConfigureTransport
	ExportHttp2ConfigureServer    = http2ConfigureServer
)

func init() {
	// We only want to pay for this cost during testing.
	// When not under test, these values are always nil
	// and never assigned to.
	testHookMu = new(sync.Mutex)
}

var (
	SetEnterRoundTripHook  = hookSetter(&testHookEnterRoundTrip)
	SetTestHookWaitResLoop = hookSetter(&testHookWaitResLoop)
	SetRoundTripRetried    = hookSetter(&testHookRoundTripRetried)
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
		handler: handler,
		timeout: func() <-chan time.Time { return ch },
		// (no body and nil cancelTimer)
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
	if t.idleConn == nil {
		return
	}
	for key := range t.idleConn {
		keys = append(keys, key.String())
	}
	return
}

func (t *Transport) IdleConnCountForTesting(cacheKey string) int {
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	if t.idleConn == nil {
		return 0
	}
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
