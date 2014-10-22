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

func swapUint64(addr *uint64, new uint64) (old uint64) {
	for {
		old = *addr
		if CompareAndSwapUint64(addr, old, new) {
			break
		}
	}
	return
}

// Additional ARM-specific assembly routines.
// Declaration here to give assembly routines correct stack maps for arguments.
func armCompareAndSwapUint32(addr *uint32, old, new uint32) (swapped bool)
func armCompareAndSwapUint64(addr *uint64, old, new uint64) (swapped bool)
func generalCAS64(addr *uint64, old, new uint64) (swapped bool)
func armAddUint32(addr *uint32, delta uint32) (new uint32)
func armAddUint64(addr *uint64, delta uint64) (new uint64)
func armSwapUint32(addr *uint32, new uint32) (old uint32)
func armSwapUint64(addr *uint64, new uint64) (old uint64)
func armLoadUint64(addr *uint64) (val uint64)
func armStoreUint64(addr *uint64, val uint64)
