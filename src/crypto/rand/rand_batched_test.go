// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || freebsd || dragonfly || solaris

package rand

import (
	"bytes"
	"errors"
	"testing"
)

func TestBatched(t *testing.T) {
	fillBatched := batched(func(p []byte) error {
		for i := range p {
			p[i] = byte(i)
		}
		return nil
	}, 5)

	p := make([]byte, 13)
	if err := fillBatched(p); err != nil {
		t.Fatalf("batched function returned error: %s", err)
	}
	expected := []byte{0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2}
	if !bytes.Equal(expected, p) {
		t.Errorf("incorrect batch result: got %x, want %x", p, expected)
	}
}

func TestBatchedError(t *testing.T) {
	b := batched(func(p []byte) error { return errors.New("") }, 5)
	if b(make([]byte, 13)) == nil {
		t.Fatal("batched function should have returned error")
	}
}

func TestBatchedEmpty(t *testing.T) {
	b := batched(func(p []byte) error { return errors.New("") }, 5)
	if err := b(make([]byte, 0)); err != nil {
		t.Fatalf("empty slice should always return nil: %s", err)
	}
}
