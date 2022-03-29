// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.8
// +build !go1.8

package gc

import (
	"cmd/compile/internal/base"
	"runtime"
)

func startMutexProfiling() {
	base.Fatalf("mutex profiling unavailable in version %v", runtime.Version())
}
