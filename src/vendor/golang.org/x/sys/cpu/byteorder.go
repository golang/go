// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import (
	"encoding/binary"
	"runtime"
)

// hostByteOrder returns binary.LittleEndian on little-endian machines and
// binary.BigEndian on big-endian machines.
func hostByteOrder() binary.ByteOrder {
	switch runtime.GOARCH {
	case "386", "amd64", "amd64p32",
		"arm", "arm64",
		"mipsle", "mips64le", "mips64p32le",
		"ppc64le",
		"riscv", "riscv64":
		return binary.LittleEndian
	case "armbe", "arm64be",
		"mips", "mips64", "mips64p32",
		"ppc", "ppc64",
		"s390", "s390x",
		"sparc", "sparc64":
		return binary.BigEndian
	}
	panic("unknown architecture")
}
