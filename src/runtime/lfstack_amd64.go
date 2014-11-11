// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Amd64 uses 48-bit virtual addresses, 47-th bit is used as kernel/user flag.
// So we use 17msb of pointers as ABA counter.
const (
	lfPtrBits   = 47
	lfCountMask = 1<<17 - 1
)
