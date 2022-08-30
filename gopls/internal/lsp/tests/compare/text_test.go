// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compare_test

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
)

func TestText(t *testing.T) {
	tests := []struct {
		got, want, wantDiff string
	}{
		{"", "", ""},
		{"equal", "equal", ""},
		{"a", "b", "--- want\n+++ got\n@@ -1 +1 @@\n-b\n+a\n"},
		{"a\nd\nc\n", "a\nb\nc\n", "--- want\n+++ got\n@@ -1,4 +1,4 @@\n a\n-b\n+d\n c\n \n"},
	}

	for _, test := range tests {
		if gotDiff := compare.Text(test.want, test.got); gotDiff != test.wantDiff {
			t.Errorf("compare.Text(%q, %q) =\n%q, want\n%q", test.want, test.got, gotDiff, test.wantDiff)
		}
	}
}
