// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package govulncheck

import "time"

// Result is the result of vulnerability scanning.
type Result struct {
	// Vulns contains all vulnerabilities that are called or imported by
	// the analyzed module.
	Vulns []*Vuln `json:",omitempty"`

	// Mode contains the source of the vulnerability info.
	// Clients of the gopls.fetch_vulncheck_result command may need
	// to interprete the vulnerabilities differently based on the
	// analysis mode. For example, Vuln without callstack traces
	// indicate a vulnerability that is not used if the result was
	// from 'govulncheck' analysis mode. On the other hand, Vuln
	// without callstack traces just implies the package with the
	// vulnerability is known to the workspace and we do not know
	// whether the vulnerable symbols are actually used or not.
	Mode AnalysisMode `json:",omitempty"`

	// AsOf describes when this Result was computed using govulncheck.
	// It is valid only with the govulncheck analysis mode.
	AsOf time.Time `json:",omitempty"`
}

type AnalysisMode string

const (
	ModeInvalid     AnalysisMode = "" // zero value
	ModeGovulncheck AnalysisMode = "govulncheck"
	ModeImports     AnalysisMode = "imports"
)
