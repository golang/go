// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"testing"
)

func testMatch(t *testing.T, pkgName, fnName, toMatch string, match bool) {
	if matchPkgFn(pkgName, fnName, toMatch) != match {
		t.Errorf("%v != matchPkgFn(%s, %s, %s)", match, pkgName, fnName, toMatch)
	}
}

func TestMatchPkgFn(t *testing.T) {
	// "aFunc" matches "aFunc" (in any package)
	// "aPkg.aFunc" matches "aPkg.aFunc"
	// "aPkg/subPkg.aFunc" matches "subPkg.aFunc"

	match := func(pkgName, fnName, toMatch string) {
		if !matchPkgFn(pkgName, fnName, toMatch) {
			t.Errorf("matchPkgFn(%s, %s, %s) did not match", pkgName, fnName, toMatch)
		}
	}
	match("aPkg", "AFunc", "AFunc")
	match("aPkg", "AFunc", "AFunc")
	match("aPkg", "AFunc", "aPkg.AFunc")
	match("aPkg/sPkg", "AFunc", "aPkg/sPkg.AFunc")
	match("aPkg/sPkg", "AFunc", "sPkg.AFunc")

	notmatch := func(pkgName, fnName, toMatch string) {
		if matchPkgFn(pkgName, fnName, toMatch) {
			t.Errorf("matchPkgFn(%s, %s, %s) should not match", pkgName, fnName, toMatch)
		}
	}
	notmatch("aPkg", "AFunc", "BFunc")
	notmatch("aPkg", "AFunc", "aPkg.BFunc")
	notmatch("aPkg", "AFunc", "bPkg.AFunc")
	notmatch("aPkg", "AFunc", "aPkg_AFunc")
	notmatch("aPkg/sPkg", "AFunc", "aPkg/ssPkg.AFunc")
	notmatch("aPkg/sPkg", "AFunc", "XPkg.AFunc")
}
