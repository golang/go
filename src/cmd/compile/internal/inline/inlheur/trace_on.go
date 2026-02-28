// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build debugtrace

package inlheur

import (
	"os"
	"strconv"
)

var debugTrace = 0

func enableDebugTrace(x int) {
	debugTrace = x
}

func enableDebugTraceIfEnv() {
	v := os.Getenv("DEBUG_TRACE_INLHEUR")
	if v == "" {
		return
	}
	if v[0] == '*' {
		if !UnitTesting() {
			return
		}
		v = v[1:]
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		return
	}
	debugTrace = i
}

func disableDebugTrace() {
	debugTrace = 0
}
