// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build plan9

package runtime

import "unsafe"

var (
	mallocScanTable   []func(size uintptr, typ *_type, needzero bool) unsafe.Pointer
	mallocNoScanTable []func(size uintptr, typ *_type, needzero bool) unsafe.Pointer
)
