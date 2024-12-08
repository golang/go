// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/platform"
	"testing"
)

// TestSupported tests that dist and the main tools agree on
// which build modes are supported for a given target. We do things
// this way because the dist tool needs to be buildable directly by
// the bootstrap compiler, and as such can't import internal packages.
func TestSupported(t *testing.T) {
	defer func(a, o string) {
		goarch = a
		goos = o
	}(goarch, goos)

	var modes = []string{
		// we assume that "exe" and "archive" always work
		"pie",
		"c-archive",
		"c-shared",
		"shared",
		"plugin",
	}

	for _, a := range okgoarch {
		goarch = a
		for _, o := range okgoos {
			if _, ok := cgoEnabled[o+"/"+a]; !ok {
				continue
			}
			goos = o
			for _, mode := range modes {
				var dt tester
				dist := dt.supportedBuildmode(mode)
				std := platform.BuildModeSupported("gc", mode, o, a)
				if dist != std {
					t.Errorf("discrepancy for %s-%s %s: dist says %t, standard library says %t", o, a, mode, dist, std)
				}
			}
		}
	}
}
