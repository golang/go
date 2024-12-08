// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

type RF_State int

// These constants are shared between the compiler, which uses them for state functions
// and panic indicators, and the runtime, which turns them into more meaningful strings
// For best code generation, RF_DONE and RF_READY should be 0 and 1.
const (
	RF_DONE          = RF_State(iota) // body of loop has exited in a non-panic way
	RF_READY                          // body of loop has not exited yet, is not running  -- this is not a panic index
	RF_PANIC                          // body of loop is either currently running, or has panicked
	RF_EXHAUSTED                      // iterator function return, i.e., sequence is "exhausted"
	RF_MISSING_PANIC = 4              // body of loop panicked but iterator function defer-recovered it away
)
