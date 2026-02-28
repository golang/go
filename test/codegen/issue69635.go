// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func calc(a uint64) uint64 {
	v := a >> 20 & 0x7f
	// amd64: `SHRQ\s\$17, AX$`, `ANDL\s\$1016, AX$`
	return v << 3
}
