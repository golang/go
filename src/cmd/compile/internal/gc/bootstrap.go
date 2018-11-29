// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.8

package gc

import "runtime"

func startMutexProfiling() {
	Fatalf("mutex profiling unavailable in version %v", runtime.Version())
}
