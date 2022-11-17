// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package govulncheck

// Result is the result of vulnerability scanning.
type Result struct {
	// Vulns contains all vulnerabilities that are called or imported by
	// the analyzed module.
	Vulns []*Vuln
}
