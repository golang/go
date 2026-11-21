// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package platform_test

import (
	"internal/platform"
	"testing"
)

func TestMustLinkExternalWithoutHostObj(t *testing.T) {
	for _, tt := range []struct {
		goos, goarch string
		want         bool
	}{
		{"android", "arm64", false},
		{"android", "amd64", true},
		{"android", "386", true},
		{"android", "arm", true},
		{"ios", "arm64", true},
		{"aix", "ppc64", true},
		{"linux", "mips64le", true},
		{"dragonfly", "amd64", true},
		{"linux", "amd64", false},
		{"darwin", "arm64", false},
		{"windows", "amd64", false},
	} {
		if got := platform.MustLinkExternalWithoutHostObj(tt.goos, tt.goarch); got != tt.want {
			t.Errorf("MustLinkExternalWithoutHostObj(%q, %q) = %v, want %v", tt.goos, tt.goarch, got, tt.want)
		}
	}

	// Without host objects, external linking is always required whenever
	// it is required without cgo, is never required where it is not
	// required with host objects, and otherwise only android/arm64 differs.
	for _, p := range platform.List {
		withoutCgo := platform.MustLinkExternal(p.GOOS, p.GOARCH, false)
		withHostObj := platform.MustLinkExternal(p.GOOS, p.GOARCH, true)
		withoutHostObj := platform.MustLinkExternalWithoutHostObj(p.GOOS, p.GOARCH)
		if withoutCgo && !withoutHostObj {
			t.Errorf("%s: MustLinkExternal(withCgo=false) = true, but MustLinkExternalWithoutHostObj = false", p)
		}
		if withoutHostObj && !withHostObj {
			t.Errorf("%s: MustLinkExternalWithoutHostObj = true, but MustLinkExternal(withCgo=true) = false", p)
		}
		if withoutHostObj != withHostObj && p != (platform.OSArch{GOOS: "android", GOARCH: "arm64"}) {
			t.Errorf("%s: MustLinkExternalWithoutHostObj = %v, MustLinkExternal(withCgo=true) = %v", p, withoutHostObj, withHostObj)
		}
	}
}
