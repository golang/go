// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64asm

import (
	"strings"
)

// GNUSyntax returns the GNU assembler syntax for the instruction, as defined by GNU binutils.
// This form typically matches the syntax defined in the Loong64 Reference Manual. See
// https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html
func GNUSyntax(inst Inst) string {
	return strings.ToLower(inst.String())
}
