// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (linux && !386 && !amd64 && !arm && !arm64 && !loong64 && !mips64 && !mips64le && !ppc64 && !ppc64le && !riscv64 && !s390x) || !linux

package runtime

// A dummy version of inVDSOPage for targets that don't use a VDSO.

func inVDSOPage(pc uintptr) bool {
	return false
}
