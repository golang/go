// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import "strings"

func LinknameBlocked(pkg, name string) bool {
	pkgs, ok := blockedLinknames[name]
	if !ok {
		return false
	}
	for _, p := range pkgs {
		if pkg == p {
			return false
		}
		if strings.HasPrefix(pkg, "crypto/internal/fips140/v") {
			parts := strings.Split(pkg, "/")
			parts = append(parts[:3], parts[4:]...)
			pkg := strings.Join(parts, "/")
			if pkg == p {
				return false
			}
		}
	}
	return true
}

func logosLoaderCheckLinkname(pkg, name string, reportError func()) bool {
	if _, ok := blockedLinknames[name]; ok {
		if LinknameBlocked(pkg, name) {
			reportError()
		}
		return true
	}
	return false
}
