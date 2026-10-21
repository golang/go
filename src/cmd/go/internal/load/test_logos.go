// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import "cmd/go/internal/cfg"

// logosTestXTest marks the standard library's external test package of p as
// standard-library provenance. TestPackagesFor builds that package by hand
// and its literal copies Goroot but not Standard, so the object for
// e.g. reflect_test compiles without -std and the linker capabilities
// audit would blame its frames as third-party dependencies. The mark keeps
// the exemption tied to the real path (reflect_test), so a third-party
// package foo_test — including one whose import path ends in _test —
// stays unmarked and audited. Plain go never sees it, and a nil pxtest has
// no external test files and is left alone.
func logosTestXTest(p, pxtest *Package) {
	if !cfg.Logos || pxtest == nil || !p.Standard {
		return
	}
	pxtest.Standard = true
}
