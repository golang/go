// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"io"
	"testing"
)

func TestPrefixSuffixSaver(t *testing.T) {
	tests := []struct {
		N      int
		writes []string
		want   string
	}{
		{
			N:      2,
			writes: nil,
			want:   "",
		},
		{
			N:      2,
			writes: []string{"a"},
			want:   "a",
		},
		{
			N:      2,
			writes: []string{"abc", "d"},
			want:   "abcd",
		},
		{
			N:      2,
			writes: []string{"abc", "d", "e"},
			want:   "ab\n... omitting 1 bytes ...\nde",
		},
		{
			N:      2,
			writes: []string{"ab______________________yz"},
			want:   "ab\n... omitting 22 bytes ...\nyz",
		},
		{
			N:      2,
			writes: []string{"ab_______________________y", "z"},
			want:   "ab\n... omitting 23 bytes ...\nyz",
		},
	}
	for i, tt := range tests {
		w := &prefixSuffixSaver{N: tt.N}
		for _, s := range tt.writes {
			n, err := io.WriteString(w, s)
			if err != nil || n != len(s) {
				t.Errorf("%d. WriteString(%q) = %v, %v; want %v, %v", i, s, n, err, len(s), nil)
			}
		}
		if got := string(w.Bytes()); got != tt.want {
			t.Errorf("%d. Bytes = %q; want %q", i, got, tt.want)
		}
	}
}
