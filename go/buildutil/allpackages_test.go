// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Incomplete source tree on Android.

// +build !android

package buildutil_test

import (
	"go/build"
	"testing"

	"golang.org/x/tools/go/buildutil"
)

func TestAllPackages(t *testing.T) {
	all := buildutil.AllPackages(&build.Default)

	set := make(map[string]bool)
	for _, pkg := range all {
		set[pkg] = true
	}

	const wantAtLeast = 250
	if len(all) < wantAtLeast {
		t.Errorf("Found only %d packages, want at least %d", len(all), wantAtLeast)
	}

	for _, want := range []string{"fmt", "crypto/sha256", "golang.org/x/tools/go/buildutil"} {
		if !set[want] {
			t.Errorf("Package %q not found; got %s", want, all)
		}
	}
}
