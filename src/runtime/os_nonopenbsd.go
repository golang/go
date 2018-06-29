// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !openbsd

package runtime

// osStackAlloc performs OS-specific initialization before s is used
// as stack memory.
func osStackAlloc(s *mspan) {
}

// osStackFree undoes the effect of osStackAlloc before s is returned
// to the heap.
func osStackFree(s *mspan) {
}
