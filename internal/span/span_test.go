// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span_test

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/internal/span"
)

var (
	formats = []string{"%v", "%#v", "%+v"}
	tests   = [][]string{
		{"C:/file_a", "C:/file_a", "file:///C:/file_a:0:0#0-0:0#0"},
		{"C:/file_b:1:2", "C:/file_b:#1", "file:///C:/file_b:1:2#1-1:2#1"},
		{"C:/file_c:1000", "C:/file_c:#9990", "file:///C:/file_c:1000:0#9990-1000:0#9990"},
		{"C:/file_d:14:9", "C:/file_d:#138", "file:///C:/file_d:14:9#138-14:9#138"},
		{"C:/file_e:1:2-7", "C:/file_e:#1-#6", "file:///C:/file_e:1:2#1-1:7#6"},
		{"C:/file_f:500-502", "C:/file_f:#4990-#5010", "file:///C:/file_f:500:0#4990-502:0#5010"},
		{"C:/file_g:3:7-8", "C:/file_g:#26-#27", "file:///C:/file_g:3:7#26-3:8#27"},
		{"C:/file_h:3:7-4:8", "C:/file_h:#26-#37", "file:///C:/file_h:3:7#26-4:8#37"},
	}
)

func TestFormat(t *testing.T) {
	converter := lines(10)
	for _, test := range tests {
		for ti, text := range test {
			spn := span.Parse(text)
			if ti <= 1 {
				// we can check %v produces the same as the input
				expect := toPath(test[ti])
				if got := fmt.Sprintf("%v", spn); got != expect {
					t.Errorf("printing %q got %q expected %q", text, got, expect)
				}
			}
			complete := spn.Clean(converter)
			for fi, format := range []string{"%v", "%#v", "%+v"} {
				expect := toPath(test[fi])
				if got := fmt.Sprintf(format, complete); got != expect {
					t.Errorf("printing completeted %q as %q got %q expected %q [%+v]", text, format, got, expect, spn)
				}
			}
		}
	}
}

func toPath(value string) string {
	if strings.HasPrefix(value, "file://") {
		return value
	}
	return filepath.FromSlash(value)
}

type lines int

func (l lines) ToCoord(offset int) (int, int) {
	return (offset / int(l)) + 1, (offset % int(l)) + 1
}

func (l lines) ToOffset(line, col int) int {
	return (int(l) * (line - 1)) + (col - 1)
}
