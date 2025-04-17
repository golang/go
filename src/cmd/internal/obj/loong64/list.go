// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"cmd/internal/obj"
	"fmt"
)

func init() {
	obj.RegisterRegister(obj.RBaseLOONG64, REG_LAST, rconv)
	obj.RegisterOpcode(obj.ABaseLoong64, Anames)
}

func arrange(a int16) string {
	switch a {
	case ARNG_32B:
		return "B32"
	case ARNG_16H:
		return "H16"
	case ARNG_8W:
		return "W8"
	case ARNG_4V:
		return "V4"
	case ARNG_2Q:
		return "Q2"
	case ARNG_16B:
		return "B16"
	case ARNG_8H:
		return "H8"
	case ARNG_4W:
		return "W4"
	case ARNG_2V:
		return "V2"
	case ARNG_B:
		return "B"
	case ARNG_H:
		return "H"
	case ARNG_W:
		return "W"
	case ARNG_V:
		return "V"
	case ARNG_BU:
		return "BU"
	case ARNG_HU:
		return "HU"
	case ARNG_WU:
		return "WU"
	case ARNG_VU:
		return "VU"
	default:
		return "ARNG_???"
	}
}

func rconv(r int) string {
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
	}

	// bits 0-4 indicates register: Vn or Xn
	// bits 5-9 indicates arrangement: <T>
	// bits 10 indicates SMID type: 0: LSX, 1: LASX
	simd_type := (int16(r) >> EXT_SIMDTYPE_SHIFT) & EXT_SIMDTYPE_MASK
	reg_num := (int16(r) >> EXT_REG_SHIFT) & EXT_REG_MASK
	arng_type := (int16(r) >> EXT_TYPE_SHIFT) & EXT_TYPE_MASK
	reg_prefix := "#"
	switch simd_type {
	case LSX:
		reg_prefix = "V"
	case LASX:
		reg_prefix = "X"
	}

	switch {
	case REG_ARNG <= r && r < REG_ELEM:
		return fmt.Sprintf("%s%d.%s", reg_prefix, reg_num, arrange(arng_type))

	case REG_ELEM <= r && r < REG_ELEM_END:
		return fmt.Sprintf("%s%d.%s", reg_prefix, reg_num, arrange(arng_type))
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
