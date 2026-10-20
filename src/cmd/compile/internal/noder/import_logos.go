// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"os"

	"cmd/compile/internal/logos/facts"
)

// This file holds the loGo/s facts-sidecar reader (M3-S1). Upstream files
// carry only hook lines; all logic lives here.
//
// readLogosFacts scans the archive for the __.LOGOS member following
// __.PKGDEF (re-seeking from the start, so readExportData keeps its
// signature) and adopts its bytes into the importer overlay registry.
// It reports whether the member was found: archives without it predate
// the sidecar (plain-Go packages) and import as before, and the Go
// union-import check runs only when it is present, so those scopes are
// never resolved. A Go importer (go/importer) reads only __.PKGDEF and
// never sees it.
func readLogosFacts(f *os.File, path, data string) (bool, error) {
	member, err := facts.FindMember(f)
	if err != nil {
		return false, err
	}
	if member == nil {
		return false, nil
	}
	_, err = facts.Import(path, data, member)
	return true, err
}
