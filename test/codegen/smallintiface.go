// asmcheck

package codegen

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

func booliface() interface{} {
	// amd64:`LEAQ\truntime.staticuint64s\+8\(SB\)`
	return true
}

func smallint8iface() interface{} {
	// amd64:`LEAQ\truntime.staticuint64s\+2024\(SB\)`
	return int8(-3)
}

func smalluint8iface() interface{} {
	// amd64:`LEAQ\truntime.staticuint64s\+24\(SB\)`
	return uint8(3)
}
