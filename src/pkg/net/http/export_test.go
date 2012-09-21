// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bridge package to expose http internals to tests in the http_test
// package.

package http

import "time"

func (t *Transport) IdleConnKeysForTesting() (keys []string) {
	keys = make([]string, 0)
	t.idleLk.Lock()
	defer t.idleLk.Unlock()
	if t.idleConn == nil {
		return
	}
	for key := range t.idleConn {
		keys = append(keys, key)
	}
	return
}

func (t *Transport) IdleConnCountForTesting(cacheKey string) int {
	t.idleLk.Lock()
	defer t.idleLk.Unlock()
	if t.idleConn == nil {
		return 0
	}
	conns, ok := t.idleConn[cacheKey]
	if !ok {
		return 0
	}
	return len(conns)
}

func NewTestTimeoutHandler(handler Handler, ch <-chan time.Time) Handler {
	f := func() <-chan time.Time {
		return ch
	}
	return &timeoutHandler{handler, f, ""}
}
