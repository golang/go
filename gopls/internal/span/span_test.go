// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span_test

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/span"
)

func TestFormat(t *testing.T) {
	formats := []string{"%v", "%#v", "%+v"}

	// Element 0 is the input, and the elements 0-2 are the expected
	// output in [%v %#v %+v] formats. Thus the first must be in
	// canonical form (invariant under span.Parse + fmt.Sprint).
	// The '#' form displays offsets; the '+' form outputs a URI.
	// If len=4, element 0 is a noncanonical input and 1-3 are expected outputs.
	for _, test := range [][]string{
		{"C:/file_a", "C:/file_a", "file:///C:/file_a:#0"},
		{"C:/file_b:1:2", "C:/file_b:1:2", "file:///C:/file_b:1:2"},
		{"C:/file_c:1000", "C:/file_c:1000", "file:///C:/file_c:1000:1"},
		{"C:/file_d:14:9", "C:/file_d:14:9", "file:///C:/file_d:14:9"},
		{"C:/file_e:1:2-7", "C:/file_e:1:2-7", "file:///C:/file_e:1:2-1:7"},
		{"C:/file_f:500-502", "C:/file_f:500-502", "file:///C:/file_f:500:1-502:1"},
		{"C:/file_g:3:7-8", "C:/file_g:3:7-8", "file:///C:/file_g:3:7-3:8"},
		{"C:/file_h:3:7-4:8", "C:/file_h:3:7-4:8", "file:///C:/file_h:3:7-4:8"},
		{"C:/file_i:#100", "C:/file_i:#100", "file:///C:/file_i:#100"},
		{"C:/file_j:#26-#28", "C:/file_j:#26-#28", "file:///C:/file_j:#26-0#28"}, // 0#28?
		{"C:/file_h:3:7#26-4:8#37", // not canonical
			"C:/file_h:3:7-4:8", "C:/file_h:#26-#37", "file:///C:/file_h:3:7#26-4:8#37"}} {
		input := test[0]
		spn := span.Parse(input)
		wants := test[0:3]
		if len(test) == 4 {
			wants = test[1:4]
		}
		for i, format := range formats {
			want := toPath(wants[i])
			if got := fmt.Sprintf(format, spn); got != want {
				t.Errorf("Sprintf(%q, %q) = %q, want %q", format, input, got, want)
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
