// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlog

import "sync"

// PanicOnExit0 reports whether to panic on a call to os.Exit(0).
// This is in the testlog package because, like other definitions in
// package testlog, it is a hook between the testing package and the
// os package. This is used to ensure that an early call to os.Exit(0)
// does not cause a test to pass.
func PanicOnExit0() bool {
	panicOnExit0.mu.Lock()
	defer panicOnExit0.mu.Unlock()
	return panicOnExit0.val
}

// panicOnExit0 is the flag used for PanicOnExit0. This uses a lock
// because the value can be cleared via a timer call that may race
// with calls to os.Exit
var panicOnExit0 struct {
	mu  sync.Mutex
	val bool
}

// SetPanicOnExit0 sets panicOnExit0 to v.
func SetPanicOnExit0(v bool) {
	panicOnExit0.mu.Lock()
	defer panicOnExit0.mu.Unlock()
	panicOnExit0.val = v
}
