// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package govulncheck

import (
	"golang.org/x/mod/semver"
	isem "golang.org/x/tools/gopls/internal/govulncheck/semver"
	"golang.org/x/vuln/osv"
)

// LatestFixed returns the latest fixed version in the list of affected ranges,
// or the empty string if there are no fixed versions.
func LatestFixed(modulePath string, as []osv.Affected) string {
	v := ""
	for _, a := range as {
		if a.Package.Name != modulePath {
			continue
		}
		for _, r := range a.Ranges {
			if r.Type == osv.TypeSemver {
				for _, e := range r.Events {
					if e.Fixed != "" && (v == "" ||
						semver.Compare(isem.CanonicalizeSemverPrefix(e.Fixed), isem.CanonicalizeSemverPrefix(v)) > 0) {
						v = e.Fixed
					}
				}
			}
		}
	}
	return v
}
