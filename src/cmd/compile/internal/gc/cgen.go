// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "cmd/internal/sys"

// hasHMUL64 reports whether the architecture supports 64-bit
// signed and unsigned high multiplication (OHMUL).
func hasHMUL64() bool {
	switch Ctxt.Arch.Family {
	case sys.AMD64, sys.S390X, sys.ARM64:
		return true
	case sys.ARM, sys.I386, sys.MIPS64, sys.PPC64:
		return false
	}
	Fatalf("unknown architecture")
	return false
}

// hasRROTC64 reports whether the architecture supports 64-bit
// rotate through carry instructions (ORROTC).
func hasRROTC64() bool {
	switch Ctxt.Arch.Family {
	case sys.AMD64:
		return true
	case sys.ARM, sys.ARM64, sys.I386, sys.MIPS64, sys.PPC64, sys.S390X:
		return false
	}
	Fatalf("unknown architecture")
	return false
}

func hasRightShiftWithCarry() bool {
	switch Ctxt.Arch.Family {
	case sys.ARM64:
		return true
	case sys.AMD64, sys.ARM, sys.I386, sys.MIPS64, sys.PPC64, sys.S390X:
		return false
	}
	Fatalf("unknown architecture")
	return false
}

func hasAddSetCarry() bool {
	switch Ctxt.Arch.Family {
	case sys.ARM64:
		return true
	case sys.AMD64, sys.ARM, sys.I386, sys.MIPS64, sys.PPC64, sys.S390X:
		return false
	}
	Fatalf("unknown architecture")
	return false
}
