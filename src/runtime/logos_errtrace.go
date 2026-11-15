// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:linkname logosTraceRecord
func logosTraceRecord(site unsafe.Pointer, errTyp unsafe.Pointer, errData uintptr) {
	gp := getg()
	if errTyp != gp.logosErrLastTyp || errData != gp.logosErrLast {
		gp.logosErrLen = 0
		gp.logosErrLastTyp = errTyp
		gp.logosErrLast = errData
	}
	if gp.logosErrLen < uint8(len(gp.logosErrSite)) {
		gp.logosErrSite[gp.logosErrLen] = site
		gp.logosErrLen++
	} else {
		copy(gp.logosErrSite[:], gp.logosErrSite[1:])
		gp.logosErrSite[len(gp.logosErrSite)-1] = site
	}
}

//go:linkname logosTraceRead
func logosTraceRead(buf []unsafe.Pointer) int {
	gp := getg()
	n := gp.logosErrLen
	copy(buf, gp.logosErrSite[:n])
	return int(n)
}

func logosClearErrTrace(gp *g) {
	gp.logosErrSite = [8]unsafe.Pointer{}
	gp.logosErrLen = 0
	gp.logosErrLastTyp = nil
	gp.logosErrLast = 0
}
