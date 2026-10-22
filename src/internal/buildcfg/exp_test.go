// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildcfg

import "testing"

func TestARM64EExperiment(t *testing.T) {
	for _, tt := range []struct {
		goos, goarch string
		want         bool
	}{
		{"darwin", "arm64", true},
		{"ios", "arm64", true},
		{"darwin", "amd64", false},
		{"linux", "arm64", false},
	} {
		exp, err := ParseGOEXPERIMENT(tt.goos, tt.goarch, "arm64e")
		if err != nil {
			t.Fatal(err)
		}
		if exp.ARM64E != tt.want {
			t.Errorf("%s/%s: ARM64E = %v, want %v", tt.goos, tt.goarch, exp.ARM64E, tt.want)
		}
	}
}
