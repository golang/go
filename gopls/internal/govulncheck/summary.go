// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package govulncheck

import "golang.org/x/vuln/osv"

// TODO(hyangah): Find a better package for these types
// unless golang.org/x/vuln/exp/govulncheck starts to export these.

// Summary is the govulncheck result.
type Summary struct {
	// Vulnerabilities affecting the analysis target binary or source code.
	Affecting []Vuln
	// Vulnerabilities that may be imported but the vulnerable symbols are
	// not called. For binary analysis, this will be always empty.
	NonAffecting []Vuln
}

// Vuln represents a vulnerability relevant to a (module, package).
type Vuln struct {
	OSV     *osv.Entry
	PkgPath string // Package path.
	ModPath string // Module path.
	FoundIn string // <package path>@<version> if we know when it was introduced. Empty otherwise.
	FixedIn string // <package path>@<version> if fix is available. Empty otherwise.
	// Trace contains a call stack for each affecting symbol.
	// For vulnerabilities found from binary analysis, and vulnerabilities
	// that are reported as Unaffecting ones, this will be always empty.
	Trace []Trace
}

// Trace represents a sample trace for a vulnerable symbol.
type Trace struct {
	Symbol string       // Name of the detected vulnerable function or method.
	Desc   string       // One-line description of the callstack.
	Stack  []StackEntry // Call stack.
	Seen   int          // Number of similar call stacks.
}

// StackEntry represents a call stack entry.
type StackEntry struct {
	FuncName string // Function name is the function name, adjusted to remove pointer annotation.
	CallSite string // Position of the call/reference site. It is one of the formats token.Pos.String() returns or empty if unknown.
}
