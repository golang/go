// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package randutil contains internal randomness utilities for various
// crypto packages.
package randutil

import (
	"io"
	"sync"
)

var (
	closedChanOnce sync.Once
	closedChan     chan struct{}
)

// MaybeReadByte reads a single byte from r with ~50% probability. This is used
// to ensure that callers do not depend on non-guaranteed behaviour, e.g.
// assuming that rsa.GenerateKey is deterministic w.r.t. a given random stream.
//
// This does not affect tests that pass a stream of fixed bytes as the random
// source (e.g. a zeroReader).
func MaybeReadByte(r io.Reader) {
	closedChanOnce.Do(func() {
		closedChan = make(chan struct{})
		close(closedChan)
	})

	select {
	case <-closedChan:
		return
	case <-closedChan:
		var buf [1]byte
		r.Read(buf[:])
	}
}
