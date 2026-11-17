// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// The compiler reads this policy before a vector prefix. The getter must not
// grow the stack or schedule while reading the current goroutine's state.
//
//go:nosplit
//go:noinline
func logosAutovecPanicOnFault() bool {
	return getg().paniconfault
}
