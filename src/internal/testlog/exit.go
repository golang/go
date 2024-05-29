// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlog

import "sync"

// PanicOnExit reports whether to panic on a call to os.Exit.
// This is in the testlog package because, like other definitions in
// package testlog, it is a hook between the testing package and the
// os package. This is used to ensure that an early call to os.Exit(0)
// does not cause a test to pass.
func PanicOnExit() bool {
	panicOnExit.mu.Lock()
	defer panicOnExit.mu.Unlock()
	return panicOnExit.val
}

// panicOnExit is the flag used for PanicOnExit. This uses a lock
// because the value can be cleared via a timer call that may race
// with calls to os.Exit
var panicOnExit struct {
	mu  sync.Mutex
	val bool
}

// SetPanicOnExit sets panicOnExit to v.
func SetPanicOnExit(v bool) {
	panicOnExit.mu.Lock()
	defer panicOnExit.mu.Unlock()
	panicOnExit.val = v
}
