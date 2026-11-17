// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"slices"
	"testing"
)

func TestFindVariantContextKeepsValidHostAssignment(t *testing.T) {
	variantTagTestContext(t)
	for _, tt := range []struct {
		expr string
		host []string
	}{
		{"windows && (a || b)", []string{"b"}},
		{"windows && (!a || b)", []string{"b"}},
		{"windows && (a || !b)", []string{"untouched"}},
	} {
		t.Run(tt.expr, func(t *testing.T) {
			dir, name := variantTagTestFile(t, tt.expr)
			goos, goarch, tags, ok := findVariantContextFromLinuxAMD64(dir, name, tt.host)
			if !ok || goos != "windows" || goarch != "amd64" {
				t.Fatalf("context = %s/%s, found=%v, want windows/amd64", goos, goarch, ok)
			}
			if !slices.Equal(tags, tt.host) {
				t.Errorf("tags = %q, want existing satisfying tags %q", tags, tt.host)
			}
			assertVariantTagMatch(t, dir, name, goos, goarch, tags)
		})
	}
}
