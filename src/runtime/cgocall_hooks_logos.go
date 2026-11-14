// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/sys"
	"unsafe"
)

//go:nosplit
func logosCgocallRestoreG0SchedSP(mp *m) {
	sched := &mp.g0.sched
	sched.sp = *(*uintptr)(unsafe.Pointer(sched.sp + alignUp(sys.MinFrameSize, sys.StackAlign)))
}

//go:nosplit
func logosCgocallUnwindFastForbidden(mp *m) {
	logosCgocallRestoreG0SchedSP(mp)
	if mp.ncgo > 0 {
		mp.ncgo--
	}
	unlockOSThread()
}
