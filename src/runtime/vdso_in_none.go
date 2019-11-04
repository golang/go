// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,!386,!amd64,!arm,!arm64,!mips64,!mips64le,!ppc64,!ppc64le !linux

package runtime

// A dummy version of inVDSOPage for targets that don't use a VDSO.

func inVDSOPage(pc uintptr) bool {
	return false
}
