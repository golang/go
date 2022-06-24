// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build loong64

package cpu

// CacheLinePadSize is used to prevent false sharing of cache lines.
// We choose 64 because Loongson 3A5000 the L1 Dcache is 4-way 256-line 64-byte-per-line.
const CacheLinePadSize = 64

func doinit() {}
