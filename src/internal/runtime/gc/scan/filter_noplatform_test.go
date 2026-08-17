// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && !riscv64

package scan_test

func filterNilPlatform() func(*uintptr, int32) int32 {
	return nil
}
