// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bridge package to expose http internals to tests in the http_test
// package.

package http

import (
	"net"
	"net/url"
	"time"
)

func NewLoggingConn(baseName string, c net.Conn) net.Conn {
	return newLoggingConn(baseName, c)
}

var ExportAppendTime = appendTime

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
	return t.putIdleConn(&persistConn{
		t:        t,
		conn:     c,                   // dummy
		closech:  make(chan struct{}), // so it can be closed
		cacheKey: connectMethodKey{"", "http", "example.com"},
	})
}

func NewTestTimeoutHandler(handler Handler, ch <-chan time.Time) Handler {
	f := func() <-chan time.Time {
		return ch
	}
	return &timeoutHandler{handler, f, ""}
}

func ResetCachedEnvironment() {
	httpProxyEnv.reset()
	httpsProxyEnv.reset()
	noProxyEnv.reset()
}

var DefaultUserAgent = defaultUserAgent

func ExportRefererForURL(lastReq, newReq *url.URL) string {
	return refererForURL(lastReq, newReq)
}

// SetPendingDialHooks sets the hooks that run before and after handling
// pending dials.
func SetPendingDialHooks(before, after func()) {
	prePendingDial, postPendingDial = before, after
}

var ExportServerNewConn = (*Server).newConn

var ExportCloseWriteAndWait = (*conn).closeWriteAndWait
