// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bridge package to expose http internals to tests in the http_test
// package.

package http

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"slices"
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
	Export_is408Message               = is408Message
)

var MaxWriteWaitBeforeConnReuse = &maxWriteWaitBeforeConnReuse

func init() {
	// We only want to pay for this cost during testing.
	// When not under test, these values are always nil
	// and never assigned to.
	testHookMu = new(sync.Mutex)

	testHookClientDoResult = func(res *Response, err error) {
		if err != nil {
			if _, ok := err.(*url.Error); !ok {
				panic(fmt.Sprintf("unexpected Client.Do error of type %T; want *url.Error", err))
			}
		} else {
			if res == nil {
				panic("Client.Do returned nil, nil")
			}
			if res.Body == nil {
				panic("Client.Do returned nil res.Body and no error")
			}
		}
	}
}

func CondSkipHTTP2(t testing.TB) {
	if omitBundledHTTP2 {
		t.Skip("skipping HTTP/2 test when nethttpomithttp2 build tag in use")
	}
}

var (
	SetEnterRoundTripHook = hookSetter(&testHookEnterRoundTrip)
	SetRoundTripRetried   = hookSetter(&testHookRoundTripRetried)
)

func SetReadLoopBeforeNextReadHook(f func()) {
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

func NewTestTimeoutHandler(handler Handler, ctx context.Context) Handler {
	return &timeoutHandler{
		handler:     handler,
		testContext: ctx,
		// (no body)
	}
}

func ResetCachedEnvironment() {
	resetProxyConfig()
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
	slices.Sort(keys)
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
	slices.Sort(ret)
	return ret
}

func (t *Transport) IdleConnStrsForTesting_h2() []string {
	var ret []string
	noDialPool := t.h2transport.(*http2Transport).ConnPool.(http2noDialClientConnPool)
	pool := noDialPool.http2clientConnPool

	pool.mu.Lock()
	defer pool.mu.Unlock()

	for k, ccs := range pool.conns {
		for _, cc := range ccs {
			if cc.idleState().canTakeNewRequest {
				ret = append(ret, k)
			}
		}
	}

	slices.Sort(ret)
	return ret
}

func (t *Transport) IdleConnCountForTesting(scheme, addr string) int {
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	key := connectMethodKey{"", scheme, addr, false}
	cacheKey := key.String()
	for k, conns := range t.idleConn {
		if k.String() == cacheKey {
			return len(conns)
		}
	}
	return 0
}

func (t *Transport) IdleConnWaitMapSizeForTesting() int {
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	return len(t.idleConnWait)
}

func (t *Transport) IsIdleForTesting() bool {
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	return t.closeIdle
}

func (t *Transport) QueueForIdleConnForTesting() {
	t.queueForIdleConn(nil)
}

// PutIdleTestConn reports whether it was able to insert a fresh
// persistConn for scheme, addr into the idle connection pool.
func (t *Transport) PutIdleTestConn(scheme, addr string) bool {
	c, _ := net.Pipe()
	key := connectMethodKey{"", scheme, addr, false}

	if t.MaxConnsPerHost > 0 {
		// Transport is tracking conns-per-host.
		// Increment connection count to account
		// for new persistConn created below.
		t.connsPerHostMu.Lock()
		if t.connsPerHost == nil {
			t.connsPerHost = make(map[connectMethodKey]int)
		}
		t.connsPerHost[key]++
		t.connsPerHostMu.Unlock()
	}

	return t.tryPutIdleConn(&persistConn{
		t:        t,
		conn:     c,                   // dummy
		closech:  make(chan struct{}), // so it can be closed
		cacheKey: key,
	}) == nil
}

// PutIdleTestConnH2 reports whether it was able to insert a fresh
// HTTP/2 persistConn for scheme, addr into the idle connection pool.
func (t *Transport) PutIdleTestConnH2(scheme, addr string, alt RoundTripper) bool {
	key := connectMethodKey{"", scheme, addr, false}

	if t.MaxConnsPerHost > 0 {
		// Transport is tracking conns-per-host.
		// Increment connection count to account
		// for new persistConn created below.
		t.connsPerHostMu.Lock()
		if t.connsPerHost == nil {
			t.connsPerHost = make(map[connectMethodKey]int)
		}
		t.connsPerHost[key]++
		t.connsPerHostMu.Unlock()
	}

	return t.tryPutIdleConn(&persistConn{
		t:        t,
		alt:      alt,
		cacheKey: key,
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
	t2, err := http2configureTransports(t)
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
		st, unixSec := c.getState()
		if unixSec == 0 || st != StateIdle {
			return false
		}
	}
	return true
}

func (s *Server) ExportAllConnsByState() map[ConnState]int {
	states := map[ConnState]int{}
	s.mu.Lock()
	defer s.mu.Unlock()
	for c := range s.activeConn {
		st, _ := c.getState()
		states[st] += 1
	}
	return states
}

func (r *Request) WithT(t *testing.T) *Request {
	return r.WithContext(context.WithValue(r.Context(), tLogKey{}, t.Logf))
}

func ExportSetH2GoawayTimeout(d time.Duration) (restore func()) {
	old := http2goAwayTimeout
	http2goAwayTimeout = d
	return func() { http2goAwayTimeout = old }
}

func (r *Request) ExportIsReplayable() bool { return r.isReplayable() }

// ExportCloseTransportConnsAbruptly closes all idle connections from
// tr in an abrupt way, just reaching into the underlying Conns and
// closing them, without telling the Transport or its persistConns
// that it's doing so. This is to simulate the server closing connections
// on the Transport.
func ExportCloseTransportConnsAbruptly(tr *Transport) {
	tr.idleMu.Lock()
	for _, pcs := range tr.idleConn {
		for _, pc := range pcs {
			pc.conn.Close()
		}
	}
	tr.idleMu.Unlock()
}

// ResponseWriterConnForTesting returns w's underlying connection, if w
// is a regular *response ResponseWriter.
func ResponseWriterConnForTesting(w ResponseWriter) (c net.Conn, ok bool) {
	if r, ok := w.(*response); ok {
		return r.conn.rwc, true
	}
	return nil, false
}

func init() {
	// Set the default rstAvoidanceDelay to the minimum possible value to shake
	// out tests that unexpectedly depend on it. Such tests should use
	// runTimeSensitiveTest and SetRSTAvoidanceDelay to explicitly raise the delay
	// if needed.
	rstAvoidanceDelay = 1 * time.Nanosecond
}

// SetRSTAvoidanceDelay sets how long we are willing to wait between calling
// CloseWrite on a connection and fully closing the connection.
func SetRSTAvoidanceDelay(t *testing.T, d time.Duration) {
	prevDelay := rstAvoidanceDelay
	t.Cleanup(func() {
		rstAvoidanceDelay = prevDelay
	})
	rstAvoidanceDelay = d
}
