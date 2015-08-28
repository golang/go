// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86asm

import (
	"strings"
	"testing"
)

func TestRegString(t *testing.T) {
	for r := Reg(1); r <= regMax; r++ {
		if regNames[r] == "" {
			t.Errorf("regNames[%d] is missing", int(r))
		} else if s := r.String(); strings.Contains(s, "Reg(") {
			t.Errorf("Reg(%d).String() = %s, want proper name", int(r), s)
		}
	}
}
