// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/sym"
)

// Temporary dumping around for sym.Symbol version of helper
// functions in elf.go, still being used for some archs/oses.
// FIXME: get rid of this file when dodata() is completely
// converted and the sym.Symbol functions are not needed.

func elfsetstring(s *sym.Symbol, str string, off int) {
	if nelfstr >= len(elfstr) {
		Errorf(s, "too many elf strings")
		errorexit()
	}

	elfstr[nelfstr].s = str
	elfstr[nelfstr].off = off
	nelfstr++
}
