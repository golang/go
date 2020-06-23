// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytealg

import "internal/cpu"

const MaxBruteForce = 64

func init() {
	// Note: we're kind of lucky that this flag is available at this point.
	// The runtime sets HasVX when processing auxv records, and that happens
	// to happen *before* running the init functions of packages that
	// the runtime depends on.
	// TODO: it would really be nicer for internal/cpu to figure out this
	// flag by itself. Then we wouldn't need to depend on quirks of
	// early startup initialization order.
	if cpu.S390X.HasVX {
		MaxLen = 64
	}
}

// Cutover reports the number of failures of IndexByte we should tolerate
// before switching over to Index.
// n is the number of bytes processed so far.
// See the bytes.Index implementation for details.
func Cutover(n int) int {
	// 1 error per 8 characters, plus a few slop to start.
	return (n + 16) / 8
}
