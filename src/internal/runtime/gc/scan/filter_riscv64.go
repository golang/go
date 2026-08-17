// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scan

import "internal/cpu"

// FilterNilRVV packs the non-nil values of bufp[0:n] at the front of bufp,
// returning the packed length. It uses the RVV implementation when the V
// extension is available and falls back to the generic FilterNil otherwise.
func FilterNilRVV(bufp *uintptr, n int32) int32 {
	if cpu.RISCV64.HasV {
		return filterNilRVV(bufp, n)
	}
	return FilterNil(bufp, n)
}

//go:noescape
func filterNilRVV(bufp *uintptr, n int32) int32
