// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"testing"
)

func TestForwardCopy(t *testing.T) {
	testCases := []struct {
		dst0, dst1 int
		src0, src1 int
		want       string
	}{
		{0, 9, 0, 9, "012345678"},
		{0, 5, 4, 9, "45678"},
		{4, 9, 0, 5, "01230"},
		{1, 6, 3, 8, "34567"},
		{3, 8, 1, 6, "12121"},
		{0, 9, 3, 6, "345"},
		{3, 6, 0, 9, "012"},
		{1, 6, 0, 9, "00000"},
		{0, 4, 7, 8, "7"},
		{0, 1, 6, 8, "6"},
		{4, 4, 6, 9, ""},
		{2, 8, 6, 6, ""},
		{0, 0, 0, 0, ""},
	}
	for _, tc := range testCases {
		b := []byte("0123456789")
		dst := b[tc.dst0:tc.dst1]
		src := b[tc.src0:tc.src1]
		n := forwardCopy(dst, src)
		got := string(dst[:n])
		if got != tc.want {
			t.Errorf("dst=b[%d:%d], src=b[%d:%d]: got %q, want %q",
				tc.dst0, tc.dst1, tc.src0, tc.src1, got, tc.want)
		}
		// Check that the bytes outside of dst[:n] were not modified.
		for i, x := range b {
			if i >= tc.dst0 && i < tc.dst0+n {
				continue
			}
			if int(x) != '0'+i {
				t.Errorf("dst=b[%d:%d], src=b[%d:%d]: copy overrun at b[%d]: got '%c', want '%c'",
					tc.dst0, tc.dst1, tc.src0, tc.src1, i, x, '0'+i)
			}
		}
	}
}
