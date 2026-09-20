// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build sparc64

package cpu

// The L1 line is 32 bytes; false sharing is governed by the 64-byte L2 line.
const cacheLineSize = 64

func initOptions() {}
