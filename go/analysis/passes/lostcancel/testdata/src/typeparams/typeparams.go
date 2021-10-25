// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the lostcancel checker.

//go:build go1.18

package typeparams

import (
	"context"
	"time"
)

var bg = context.Background()

func _[T any]() {
	var _, cancel = context.WithCancel(bg) // want `the cancel function is not used on all paths \(possible context leak\)`
	if false {
		_ = cancel
	}
} // want "this return statement may be reached without using the cancel var defined on line 19"

func _[T any]() {
	_, cancel := context.WithCancel(bg)
	defer cancel() // ok
}

// User-defined Context that matches type "context.Context"
type C1[P1 any, P2 any] interface {
	Deadline() (deadline time.Time, ok P1)
	Done() <-chan struct{}
	Err() error
	Value(key P2) P2
}

func _(bg C1[bool, interface{}]) {
	ctx, _ := context.WithCancel(bg)    // want "the cancel function returned by context.WithCancel should be called, not discarded, to avoid a context leak"
	ctx, _ = context.WithTimeout(bg, 0) // want "the cancel function returned by context.WithTimeout should be called, not discarded, to avoid a context leak"
	_ = ctx
}

// User-defined Context that doesn't match type "context.Context"
type C2[P any] interface {
	WithCancel(parent C1[P, bool]) (ctx C1[P, bool], cancel func())
}

func _(c C2[interface{}]) {
	ctx, _ := c.WithCancel(nil) // not "context.WithCancel()"
	_ = ctx
}

// Further regression test for Go issue 16143.
func _() {
	type C[P any] struct{ f func() P }
	var x C[int]
	x.f()
}
