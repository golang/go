// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "runtime"

// This file contains code generation tests related to the use of the
// stack.

// check that stack stores are optimized away

// 386:"TEXT\t.*, [$]0-4"
// amd64:"TEXT\t.*, [$]0-8"
// arm:"TEXT\t.*, [$]-4-4"
// arm64:"TEXT\t.*, [$]-8-8"
// s390x:"TEXT\t.*, [$]0-8"
// ppc64le:"TEXT\t.*, [$]0-8"
// mips:"TEXT\t.*, [$]-4-4"
func StackStore() int {
	var x int
	return *(&x)
}

// Check that assembly output has matching offset and base register
// (Issue #21064).

// amd64:`.*b\+24\(SP\)`
// arm:`.*b\+4\(FP\)`
func check_asmout(a, b int) int {
	runtime.GC() // use some frame
	return b
}
