// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
)

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}

func sysargs(argc int32, argv **byte) {
	// OpenBSD does not have auxv, however we still need to initialise cpu.HWCaps.
	// For now specify the bare minimum until we add some form of capabilities
	// detection. See issue #31746.
	cpu.HWCap = 1<<1 | 1<<0 // ASIMD, FP
}
