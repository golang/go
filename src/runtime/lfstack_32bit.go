// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 arm

package runtime

// On 32-bit systems, the stored uint64 has a 32-bit pointer and 32-bit count.
const (
	lfPtrBits   = 32
	lfCountMask = 1<<32 - 1
)
