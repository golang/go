// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span_test

import (
	"testing"

	"golang.org/x/tools/internal/span"
)

// TestUTF16 tests the conversion of column information between the native
// byte offset and the utf16 form.
func TestUTF16(t *testing.T) {
	var input = []byte(`
𐐀23456789
1𐐀3456789
12𐐀456789
123𐐀56789
1234𐐀6789
12345𐐀789
123456𐐀89
1234567𐐀9
12345678𐐀
`[1:])
	c := span.NewContentConverter("test", input)
	for line := 1; line <= 9; line++ {
		runeColumn, runeChr := 0, 0
		for chr := 1; chr <= 9; chr++ {
			switch {
			case chr <= line:
				runeChr = chr
				runeColumn = chr
			case chr == line+1:
				runeChr = chr - 1
				runeColumn = chr - 1
			default:
				runeChr = chr
				runeColumn = chr + 2
			}
			p := span.Point{Line: line, Column: runeColumn}
			// check conversion to utf16 format
			gotChr := span.ToUTF16Column(c, p, input)
			if runeChr != gotChr {
				t.Errorf("ToUTF16Column(%v): expected %v, got %v", p, runeChr, gotChr)
			}
			// we deliberately delay setting the point's offset
			p.Offset = (line-1)*13 + (p.Column - 1)
			offset := c.ToOffset(p.Line, p.Column)
			if p.Offset != offset {
				t.Errorf("ToOffset(%v,%v): expected %v, got %v", p.Line, p.Column, p.Offset, offset)
			}
			// and check the conversion back
			gotPoint := span.FromUTF16Column(c, p.Line, chr, input)
			if p != gotPoint {
				t.Errorf("FromUTF16Column(%v,%v): expected %v, got %v", p.Line, chr, p, gotPoint)
			}
		}
	}
}
