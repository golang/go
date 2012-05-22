// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic

func loadUint64(addr *uint64) (val uint64) {
	for {
		val = *addr
		if CompareAndSwapUint64(addr, val, val) {
			break
		}
	}
	return
}

func storeUint64(addr *uint64, val uint64) {
	for {
		old := *addr
		if CompareAndSwapUint64(addr, old, val) {
			break
		}
	}
	return
}

func addUint64(val *uint64, delta uint64) (new uint64) {
	for {
		old := *val
		new = old + delta
		if CompareAndSwapUint64(val, old, new) {
			break
		}
	}
	return
}
