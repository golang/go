// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && mips64

package runtime

//go:noescape
func thrsleep(ident uintptr, clock_id int32, tsp *timespec, lock uintptr, abort *uint32) int32

//go:noescape
func thrwakeup(ident uintptr, n int32) int32

func osyield()

//go:nosplit
func osyield_no_g() {
	osyield()
}
