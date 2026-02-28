// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"cmd/internal/obj"
	"fmt"
)

func init() {
	obj.RegisterRegister(obj.RBaseLOONG64, REG_LAST, RegName)
	obj.RegisterOpcode(obj.ABaseLoong64, Anames)
}

func arrange(valid int16) string {
	var regPrefix string
	var arngName string

	// bits 0-4 indicates register: Vn or Xn
	// bits 5-9 indicates arrangement: <T>
	// bits 10 indicates SMID type: 0: LSX, 1: LASX
	simdType := (valid >> EXT_SIMDTYPE_SHIFT) & EXT_SIMDTYPE_MASK
	simdReg := (valid >> EXT_REG_SHIFT) & EXT_REG_MASK
	arngType := (valid >> EXT_TYPE_SHIFT) & EXT_TYPE_MASK

	switch simdType {
	case LSX:
		regPrefix = "V"
	case LASX:
		regPrefix = "X"
	default:
		regPrefix = "#"
	}

	switch arngType {
	case ARNG_32B:
		arngName = "B32"
	case ARNG_16H:
		arngName = "H16"
	case ARNG_8W:
		arngName = "W8"
	case ARNG_4V:
		arngName = "V4"
	case ARNG_2Q:
		arngName = "Q2"
	case ARNG_16B:
		arngName = "B16"
	case ARNG_8H:
		arngName = "H8"
	case ARNG_4W:
		arngName = "W4"
	case ARNG_2V:
		arngName = "V2"
	case ARNG_B:
		arngName = "B"
	case ARNG_H:
		arngName = "H"
	case ARNG_W:
		arngName = "W"
	case ARNG_V:
		arngName = "V"
	case ARNG_BU:
		arngName = "BU"
	case ARNG_HU:
		arngName = "HU"
	case ARNG_WU:
		arngName = "WU"
	case ARNG_VU:
		arngName = "VU"
	default:
		arngName = "ARNG_???"
	}

	return fmt.Sprintf("%s%d.%s", regPrefix, simdReg, arngName)
}

func RegName(r int) string {
	switch {
	case r == 0:
		return "NONE"
	case r == REGG:
		// Special case.
		return "g"
	case REG_R0 <= r && r <= REG_R31:
		return fmt.Sprintf("R%d", r-REG_R0)
	case REG_F0 <= r && r <= REG_F31:
		return fmt.Sprintf("F%d", r-REG_F0)
	case REG_FCSR0 <= r && r <= REG_FCSR31:
		return fmt.Sprintf("FCSR%d", r-REG_FCSR0)
	case REG_FCC0 <= r && r <= REG_FCC31:
		return fmt.Sprintf("FCC%d", r-REG_FCC0)
	case REG_V0 <= r && r <= REG_V31:
		return fmt.Sprintf("V%d", r-REG_V0)
	case REG_X0 <= r && r <= REG_X31:
		return fmt.Sprintf("X%d", r-REG_X0)
	case REG_ARNG <= r && r < REG_ELEM:
		return arrange(int16(r - REG_ARNG))
	case REG_ELEM <= r && r < REG_ELEM_END:
		return arrange(int16(r - REG_ELEM))
	}

	return fmt.Sprintf("badreg(%d)", r-obj.RBaseLOONG64)
}

func DRconv(a int) string {
	s := "C_??"
	if a >= C_NONE && a <= C_NCLASS {
		s = cnames0[a]
	}
	return s
}
