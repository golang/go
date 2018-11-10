// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!netgo
// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"context"
	"testing"
)

func TestCgoLookupIP(t *testing.T) {
	ctx := context.Background()
	_, err, ok := cgoLookupIP(ctx, "localhost")
	if !ok {
		t.Errorf("cgoLookupIP must not be a placeholder")
	}
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupIPWithCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err, ok := cgoLookupIP(ctx, "localhost")
	if !ok {
		t.Errorf("cgoLookupIP must not be a placeholder")
	}
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupPort(t *testing.T) {
	ctx := context.Background()
	_, err, ok := cgoLookupPort(ctx, "tcp", "smtp")
	if !ok {
		t.Errorf("cgoLookupPort must not be a placeholder")
	}
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupPortWithCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err, ok := cgoLookupPort(ctx, "tcp", "smtp")
	if !ok {
		t.Errorf("cgoLookupPort must not be a placeholder")
	}
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupPTR(t *testing.T) {
	ctx := context.Background()
	_, err, ok := cgoLookupPTR(ctx, "127.0.0.1")
	if !ok {
		t.Errorf("cgoLookupPTR must not be a placeholder")
	}
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupPTRWithCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err, ok := cgoLookupPTR(ctx, "127.0.0.1")
	if !ok {
		t.Errorf("cgoLookupPTR must not be a placeholder")
	}
	if err != nil {
		t.Error(err)
	}
}
