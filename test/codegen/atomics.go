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
	// amd64:"LOCK" -"CMPXCHG"
	atomic.AddInt32(&c.count, 1)
}

func atomicLogical64(x *atomic.Uint64) uint64 {
	var r uint64

	// arm64/v8.0:"LDCLRALD"
	// arm64/v8.1:"LDCLRALD"
	// arm64/v8.0:".*arm64HasATOMICS"
	// arm64/v8.1:-".*arm64HasATOMICS"
	// On amd64, make sure we use LOCK+AND instead of CMPXCHG when we don't use the result.
	// amd64:"LOCK" -"CMPXCHGQ"
	x.And(11)
	// arm64/v8.0:"LDCLRALD"
	// arm64/v8.1:"LDCLRALD"
	// arm64/v8.0:".*arm64HasATOMICS"
	// arm64/v8.1:-".*arm64HasATOMICS"
	// amd64:"LOCK" "CMPXCHGQ"
	r += x.And(22)

	// arm64/v8.0:"LDORALD"
	// arm64/v8.1:"LDORALD"
	// arm64/v8.0:".*arm64HasATOMICS"
	// arm64/v8.1:-".*arm64HasATOMICS"
	// On amd64, make sure we use LOCK+OR instead of CMPXCHG when we don't use the result.
	// amd64:"LOCK" -"CMPXCHGQ"
	x.Or(33)
	// arm64/v8.0:"LDORALD"
	// arm64/v8.1:"LDORALD"
	// arm64/v8.0:".*arm64HasATOMICS"
	// arm64/v8.1:-".*arm64HasATOMICS"
	// amd64:"LOCK" "CMPXCHGQ"
	r += x.Or(44)

	return r
}

func atomicLogical32(x *atomic.Uint32) uint32 {
	var r uint32

	// arm64/v8.0:"LDCLRALW"
	// arm64/v8.1:"LDCLRALW"
	// arm64/v8.0:".*arm64HasATOMICS"
	// arm64/v8.1:-".*arm64HasATOMICS"
	// On amd64, make sure we use LOCK+AND instead of CMPXCHG when we don't use the result.
	// amd64:"LOCK" -"CMPXCHGL"
	x.And(11)
	// arm64/v8.0:"LDCLRALW"
	// arm64/v8.1:"LDCLRALW"
	// arm64/v8.0:".*arm64HasATOMICS"
	// arm64/v8.1:-".*arm64HasATOMICS"
	// amd64:"LOCK" "CMPXCHGL"
	r += x.And(22)

	// arm64/v8.0:"LDORALW"
	// arm64/v8.1:"LDORALW"
	// arm64/v8.0:".*arm64HasATOMICS"
	// arm64/v8.1:-".*arm64HasATOMICS"
	// On amd64, make sure we use LOCK+OR instead of CMPXCHG when we don't use the result.
	// amd64:"LOCK" -"CMPXCHGL"
	x.Or(33)
	// arm64/v8.0:"LDORALW"
	// arm64/v8.1:"LDORALW"
	// arm64/v8.0:".*arm64HasATOMICS"
	// arm64/v8.1:-".*arm64HasATOMICS"
	// amd64:"LOCK" "CMPXCHGL"
	r += x.Or(44)

	return r
}
