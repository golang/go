// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(amd64 || arm64 || 386 || ppc64le || s390x || riscv64)

package rt

// viewAsAddr copies the leading bytes of b into an aligned temporary
// on architectures without unaligned loads (SPEC T-205, M7-S5: the
// GOARCH=mips leg). The caller checks the length.
func viewAsAddr[T any](b []byte) *T {
	return viewAsCopy[T](b)
}
