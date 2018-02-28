// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

var hardDiv bool // TODO: set if a hardware divider is available

func checkgoarm() {
	// TODO(minux): FP checks like in os_linux_arm.go.

	// osinit not called yet, so ncpu not set: must use getncpu directly.
	if getncpu() > 1 && goarm < 7 {
		print("runtime: this system has multiple CPUs and must use\n")
		print("atomic synchronization instructions. Recompile using GOARM=7.\n")
		exit(1)
	}
}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// TODO: need more entropy to better seed fastrand.
	return nanotime()
}
