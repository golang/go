// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/coverage/rtcov"
	"unsafe"
)

//go:linkname coverage_getCovCounterList internal/coverage/cfile.getCovCounterList
func coverage_getCovCounterList() []rtcov.CovCounterBlob {
	res := []rtcov.CovCounterBlob{}
	u32sz := unsafe.Sizeof(uint32(0))
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		if datap.covctrs == datap.ecovctrs {
			continue
		}
		res = append(res, rtcov.CovCounterBlob{
			Counters: (*uint32)(unsafe.Pointer(datap.covctrs)),
			Len:      uint64((datap.ecovctrs - datap.covctrs) / u32sz),
		})
	}
	return res
}
