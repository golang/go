// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

// Declarations for runtime services implemented in C or assembly.

const PtrSize = 4 << (^uintptr(0) >> 63)           // unsafe.Sizeof(uintptr(0)) but an ideal const
const RegSize = 4 << (^Uintreg(0) >> 63)           // unsafe.Sizeof(uintreg(0)) but an ideal const
const SpAlign = 1*(1-GoarchArm64) + 16*GoarchArm64 // SP alignment: 1 normally, 16 for ARM64
