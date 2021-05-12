// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64 && !linux && !freebsd && !android && (!darwin || ios)
// +build arm64
// +build !linux
// +build !freebsd
// +build !android
// +build !darwin ios

package cpu

func osInit() {
	// Other operating systems do not support reading HWCap from auxiliary vector,
	// reading privileged aarch64 system registers or sysctl in user space to detect
	// CPU features at runtime.
}
