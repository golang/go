// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// round returns size rounded up to the next multiple of align;
// align must be a power of two.
func round(size, align Addr) Addr {
	return (size + align - 1) &^ (align - 1)
}
