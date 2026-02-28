// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (ppc64 || ppc64le) && !aix && !linux

package cpu

func osinit() {
	// Other operating systems do not support reading HWCap from auxiliary vector,
	// reading privileged system registers or sysctl in user space to detect CPU
	// features at runtime.
}
