// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check that atomic instructions without dynamic checks are
// generated for architectures that support them

package codegen

import "sync/atomic"

type Counter struct {
	count int32
}

func (c *Counter) Increment() {
	// Check that ARm64 v8.0 has both atomic instruction (LDADDALW) and a dynamic check
	// (for arm64HasATOMICS), while ARM64 v8.1 has only atomic and no dynamic check.
	// arm64/v8.0:"LDADDALW"
	// arm64/v8.1:"LDADDALW"
	// arm64/v8.0:".*arm64HasATOMICS"
	// arm64/v8.1:-".*arm64HasATOMICS"
	atomic.AddInt32(&c.count, 1)
}

