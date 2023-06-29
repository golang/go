// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build debugtrace

package inlheur

var debugTrace = 0

func enableDebugTrace(x int) {
	debugTrace = x
}

func disableDebugTrace() {
	debugTrace = 0
}
