// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"encoding/hex"
	"errors"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
)

func unhex(s string) []byte {
	b, err := hex.DecodeString(strings.Map(func(c rune) rune {
		switch c {
		case ' ', '\t', '\n':
			return -1 // ignore
		}
		return c
	}, s))
	if err != nil {
		panic(err)
	}
	return b
}

// testReader implements io.Reader.
type testReader struct {
	readFunc func([]byte) (int, error)
}

func (r testReader) Read(p []byte) (n int, err error) { return r.readFunc(p) }

var errTestBodyClosed = errors.New("test body closed")

// testRequestBody is a Request.Body which blocks reads until it is closed,
// and records the number of times it has been closed.
type testRequestBody struct {
	closec chan struct{}

	mu     sync.Mutex
	closes int
}

func newTestRequestBody() *testRequestBody {
	return &testRequestBody{closec: make(chan struct{})}
}

// Read blocks until the body is closed.
func (b *testRequestBody) Read([]byte) (int, error) {
	<-b.closec
	return 0, errTestBodyClosed
}

func (b *testRequestBody) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closes++; b.closes == 1 {
		close(b.closec)
	}
	return nil
}

// closeCount returns the number of times the body has been closed.
func (b *testRequestBody) closeCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.closes
}

// synctestSubtest runs f in a subtest in a synctest.Run bubble.
func synctestSubtest(t *testing.T, name string, f func(t *testing.T)) {
	t.Run(name, func(t *testing.T) {
		synctest.Test(t, f)
	})
}
