// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"fmt"
	"os"
)

// This file contains helpers that can be used to instrument code while
// debugging.

// debugEnabled toggles the helpers below.
const debugEnabled = false

// If debugEnabled is true, debugf formats its arguments and prints to stderr.
// If debugEnabled is false, it is a no-op.
func debugf(format string, args ...interface{}) {
	if !debugEnabled {
		return
	}
	if false {
		_ = fmt.Sprintf(format, args...) // encourage vet to validate format strings
	}
	fmt.Fprintf(os.Stderr, ">>> "+format+"\n", args...)
}

// assert panics with the given msg if cond is not true.
func assert(cond bool, msg string) {
	if !cond {
		panic(msg)
	}
}
