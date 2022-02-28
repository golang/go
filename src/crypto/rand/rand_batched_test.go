// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || freebsd || dragonfly || solaris

package rand

import (
	"bytes"
	"testing"
)

func TestBatched(t *testing.T) {
	fillBatched := batched(func(p []byte) bool {
		for i := range p {
			p[i] = byte(i)
		}
		return true
	}, 5)

	p := make([]byte, 13)
	if !fillBatched(p) {
		t.Fatal("batched function returned false")
	}
	expected := []byte{0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2}
	if !bytes.Equal(expected, p) {
		t.Errorf("incorrect batch result: got %x, want %x", p, expected)
	}
}

func TestBatchedError(t *testing.T) {
	b := batched(func(p []byte) bool { return false }, 5)
	if b(make([]byte, 13)) {
		t.Fatal("batched function should have returned false")
	}
}

func TestBatchedEmpty(t *testing.T) {
	b := batched(func(p []byte) bool { return false }, 5)
	if !b(make([]byte, 0)) {
		t.Fatal("empty slice should always return true")
	}
}
