// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vulncheck provides an analysis command
// that runs vulnerability analysis using data from
// golang.org/x/vuln/vulncheck.
// This package requires go1.18 or newer.
package vulncheck

import (
	"context"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/source"
)

// With go1.18+, this is swapped with the real implementation.
var Main func(cfg packages.Config, patterns ...string) error = nil

// VulnerablePackages queries the vulndb and reports which vulnerabilities
// apply to this snapshot. The result contains a set of packages,
// grouped by vuln ID and by module.
var VulnerablePackages func(ctx context.Context, snapshot source.Snapshot, modfile source.FileHandle) (*govulncheck.Result, error) = nil
