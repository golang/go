// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strings"
	"testing"
)

// isLogosSupportPackage reports whether ip is a loGo/s support package
// (logos/rt and its future siblings under src/logos). These are
// fork-owned packages, not Go 1 API surface: logos/rt exports Site, Cold,
// Join, Context, TraceErr, and ErrorTrace for lowered .lgo code, and that
// surface evolves per milestone without a Go proposal. The api check
// (TestCheck, via Check) compares every other standard package against
// api/go1*.txt and api/next, so the support tree is excluded at the
// Walker scan the same way unsafe, vendor/, and internal/ already are.
// Leaving it in would fail the check with +pkg logos/rt lines, while an
// api/next entry would be wrong: next files only load on devel/beta
// toolchains (this tree is release go1.27.2, so they stay empty), every
// next line needs a proposal number that must not be invented, and
// promotion would bless fork API as Go 1 compatibility surface.
func isLogosSupportPackage(ip string) bool {
	seg, _, _ := strings.Cut(ip, "/")
	return seg == "logos"
}

func TestLogosSupportPackageExcluded(t *testing.T) {
	for _, ip := range []string{"logos/rt", "logos/rt/json", "logos/sync"} {
		if !isLogosSupportPackage(ip) {
			t.Errorf("isLogosSupportPackage(%q) = false, want true", ip)
		}
	}
	for _, ip := range []string{"fmt", "net/http", "unsafe", "internal/abi", "vendor/golang.org/x/net/http2/hpack", "logosx", "catalogos/v2"} {
		if isLogosSupportPackage(ip) {
			t.Errorf("isLogosSupportPackage(%q) = true, want false", ip)
		}
	}
}
