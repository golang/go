// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"internal/abi"
	"strings"

	"cmd/internal/goobj"
	"cmd/link/internal/loader"
)

var logosCloneInfixes = []string{
	abi.LogosCtDisplayInfix,
	abi.LogosCtASCIIInfix,
}

func logosCloneBaseName(name string) string {
	orig := name
	for {
		i := len(name)
		for i > 0 && name[i-1] >= '0' && name[i-1] <= '9' {
			i--
		}
		if i == len(name) {
			return name
		}
		prefix := name[:i]
		var base string
		stripped := false
		for _, infix := range logosCloneInfixes {
			if b, ok := strings.CutSuffix(prefix, infix); ok {
				base, stripped = b, true
				break
			}
		}
		if !stripped {
			return name
		}
		if base == "" {
			return orig
		}
		name = base
	}
}

func logosDisplayName(ldr *loader.Loader, s loader.Sym) string {
	name := ldr.SymName(s)
	if ldr.SymAttr(s)&goobj.SymFlagLogosCtClone != 0 {
		return logosCloneBaseName(name)
	}
	return name
}
