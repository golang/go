// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "testing"

var parseGoVersionTests = []struct {
	in  string
	out version
}{
	{"go1.21", version{1, 21}},
	{"go1.21.0", version{1, 21}},
	{"go1.21rc2", version{1, 21}},
}

func TestParseGoVersion(t *testing.T) {
	for _, tt := range parseGoVersionTests {
		if out, err := parseGoVersion(tt.in); out != tt.out || err != nil {
			t.Errorf("parseGoVersion(%q) = %v, %v, want %v, nil", tt.in, out, err, tt.out)
		}
	}
}
