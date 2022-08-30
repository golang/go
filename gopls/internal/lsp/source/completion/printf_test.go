// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"fmt"
	"testing"
)

func TestFormatOperandKind(t *testing.T) {
	cases := []struct {
		f    string
		idx  int
		kind objKind
	}{
		{"", 1, kindAny},
		{"%", 1, kindAny},
		{"%%%", 1, kindAny},
		{"%[1", 1, kindAny},
		{"%[?%s", 2, kindAny},
		{"%[abc]v", 1, kindAny},

		{"%v", 1, kindAny},
		{"%T", 1, kindAny},
		{"%t", 1, kindBool},
		{"%d", 1, kindInt},
		{"%c", 1, kindInt},
		{"%o", 1, kindInt},
		{"%O", 1, kindInt},
		{"%U", 1, kindInt},
		{"%e", 1, kindFloat | kindComplex},
		{"%E", 1, kindFloat | kindComplex},
		{"%f", 1, kindFloat | kindComplex},
		{"%F", 1, kindFloat | kindComplex},
		{"%g", 1, kindFloat | kindComplex},
		{"%G", 1, kindFloat | kindComplex},
		{"%b", 1, kindInt | kindFloat | kindComplex | kindBytes},
		{"%q", 1, kindString | kindBytes | kindStringer | kindError},
		{"%s", 1, kindString | kindBytes | kindStringer | kindError},
		{"%x", 1, kindString | kindBytes | kindInt | kindFloat | kindComplex},
		{"%X", 1, kindString | kindBytes | kindInt | kindFloat | kindComplex},
		{"%p", 1, kindPtr | kindSlice},
		{"%w", 1, kindError},

		{"%1.2f", 1, kindFloat | kindComplex},
		{"%*f", 1, kindInt},
		{"%*f", 2, kindFloat | kindComplex},
		{"%*.*f", 1, kindInt},
		{"%*.*f", 2, kindInt},
		{"%*.*f", 3, kindFloat | kindComplex},
		{"%[3]*.[2]*[1]f", 1, kindFloat | kindComplex},
		{"%[3]*.[2]*[1]f", 2, kindInt},
		{"%[3]*.[2]*[1]f", 3, kindInt},

		{"foo %% %d", 1, kindInt},
		{"%#-12.34f", 1, kindFloat | kindComplex},
		{"% d", 1, kindInt},

		{"%s %[1]X %d", 1, kindString | kindBytes},
		{"%s %[1]X %d", 2, kindInt},
	}

	for _, c := range cases {
		t.Run(fmt.Sprintf("%q#%d", c.f, c.idx), func(t *testing.T) {
			if got := formatOperandKind(c.f, c.idx); got != c.kind {
				t.Errorf("expected %d (%[1]b), got %d (%[2]b)", c.kind, got)
			}
		})
	}
}
