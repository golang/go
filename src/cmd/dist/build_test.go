// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/platform"
	"testing"
)

// TestMustLinkExternal verifies that the mustLinkExternal helper
// function matches internal/platform.MustLinkExternal.
func TestMustLinkExternal(t *testing.T) {
	for _, goos := range okgoos {
		for _, goarch := range okgoarch {
			got := mustLinkExternal(goos, goarch)
			want := platform.MustLinkExternal(goos, goarch)
			if got != want {
				t.Errorf("mustLinkExternal(%q, %q) = %v; want %v", goos, goarch, got, want)
			}
		}
	}
}
