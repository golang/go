// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 arm mips mipsle wasm darwin,arm64

// wasm is a treated as a 32-bit architecture for the purposes of the page
// allocator, even though it has 64-bit pointers. This is because any wasm
// pointer always has its top 32 bits as zero, so the effective heap address
// space is only 2^32 bytes in size (see heapAddrBits).

// darwin/arm64 is treated as a 32-bit architecture for the purposes of the
// page allocator, even though it has 64-bit pointers and a 33-bit address
// space (see heapAddrBits). The 33 bit address space cannot be rounded up
// to 64 bits because there are too many summary levels to fit in just 33
// bits.

package runtime

const (
	// The number of levels in the radix tree.
	summaryLevels = 4
)
