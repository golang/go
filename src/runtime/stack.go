// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	// Goroutine preemption request.
	// Stored into g->stackguard0 to cause split stack check failure.
	// Must be greater than any real sp.
	// 0xfffffade in hex.
	stackPreempt = ^uintptr(1313)
)
