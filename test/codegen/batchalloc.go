// asmcheck

//go:build !goexperiment.cgocheck2 && !goexperiment.runtimefreegc

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type batchAllocObject [4]uint64

func batchAllocSlice(dst []*batchAllocObject) {
	for i := range dst {
		// amd64:`CALL runtime.allocSlice32` `CALL runtime.mallocgcSmallNoScanSC4`
		// arm64:-`CALL runtime.allocSlice32`
		dst[i] = new(batchAllocObject)
	}
}

func batchAllocKeepIndex(dst []*batchAllocObject) int {
	i := 42
	for i = range dst {
		// amd64:-`CALL runtime.allocSlice32`
		dst[i] = new(batchAllocObject)
	}
	return i
}

func batchAllocForeignSlice(dst, src []*batchAllocObject) {
	for i := range src {
		// amd64:-`CALL runtime.allocSlice32`
		dst[i] = new(batchAllocObject)
	}
}

func batchAllocPointerful(dst []*[4]*int) {
	for i := range dst {
		// amd64:-`CALL runtime.allocSlice32`
		dst[i] = new([4]*int)
	}
}
