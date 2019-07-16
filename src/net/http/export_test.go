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
	Export_is408Message               = is408Message
)

const MaxWriteWaitBeforeConnReuse = maxWriteWaitBeforeConnReuse

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
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-ch
		cancel()
	}()
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
	noDialPool := t.h2transport.(*http2Transport).ConnPool.(http2noDialClientConnPool)
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
		st, unixSec := c.getState()
		if unixSec == 0 || st != StateIdle {
			return false
		}
	}
	return true
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
