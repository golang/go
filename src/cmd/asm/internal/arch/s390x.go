// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file encapsulates some of the odd characteristics of the
// s390x instruction set, to minimize its interaction
// with the core of the assembler.

package arch

import (
	"cmd/internal/obj/s390x"
)

func jumpS390x(word string) bool {
	switch word {
	case "BRC",
		"BC",
		"BCL",
		"BEQ",
		"BGE",
		"BGT",
		"BL",
		"BLE",
		"BLEU",
		"BLT",
		"BLTU",
		"BNE",
		"BR",
		"BVC",
		"BVS",
		"BRCT",
		"BRCTG",
		"CMPBEQ",
		"CMPBGE",
		"CMPBGT",
		"CMPBLE",
		"CMPBLT",
		"CMPBNE",
		"CMPUBEQ",
		"CMPUBGE",
		"CMPUBGT",
		"CMPUBLE",
		"CMPUBLT",
		"CMPUBNE",
		"CRJ",
		"CGRJ",
		"CLRJ",
		"CLGRJ",
		"CIJ",
		"CGIJ",
		"CLIJ",
		"CLGIJ",
		"CALL",
		"JMP":
		return true
	}
	return false
}

func s390xRegisterNumber(name string, n int16) (int16, bool) {
	switch name {
	case "AR":
		if 0 <= n && n <= 15 {
			return s390x.REG_AR0 + n, true
		}
	case "F":
		if 0 <= n && n <= 15 {
			return s390x.REG_F0 + n, true
		}
	case "R":
		if 0 <= n && n <= 15 {
			return s390x.REG_R0 + n, true
		}
	case "V":
		if 0 <= n && n <= 31 {
			return s390x.REG_V0 + n, true
		}
	}
	return 0, false
}
