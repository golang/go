// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

// Severity is the level a rule reports at. The order is meaningful:
// Allow < Warn < Deny < Forbid.
type Severity uint8

const (
	// Allow is silent.
	Allow Severity = iota
	// Warn prints a diagnostic and never fails the build.
	Warn
	// Deny is an error unless a //logos:allow directive precedes the
	// statement or declaration.
	Deny
	// Forbid is an error that configuration cannot lower and, in a loGo/s
	// package, a directive cannot silence.
	Forbid
)

// String returns the wire name of s: forbid, deny, warn, or allow.
func (s Severity) String() string {
	switch s {
	case Forbid:
		return "forbid"
	case Deny:
		return "deny"
	case Warn:
		return "warn"
	}
	return "allow"
}

// ParseSeverity parses a wire name. It reports false for any other text.
func ParseSeverity(name string) (Severity, bool) {
	switch name {
	case "forbid":
		return Forbid, true
	case "deny":
		return Deny, true
	case "warn":
		return Warn, true
	case "allow":
		return Allow, true
	}
	return Allow, false
}

// Group names the lint group of a rule: the part of its ID before "::".
type Group string

// The lint groups, in the order of the logos.toml [lints] table.
const (
	Correctness Group = "correctness"
	Concurrency Group = "concurrency"
	Slop        Group = "slop"
	Complexity  Group = "complexity"
	Perf        Group = "perf"
	Style       Group = "style"
	Suspicious  Group = "suspicious"
	Portability Group = "portability"
	Pedantic    Group = "pedantic"
)

// Groups lists every lint group.
var Groups = []Group{
	Correctness, Concurrency, Slop, Complexity, Perf, Style, Suspicious, Portability, Pedantic,
}

// DefaultSeverity returns the group severity used when logos.toml has no
// key for g. An unknown group defaults to Deny, the severity of an
// unclassified compiler error.
func DefaultSeverity(g Group) Severity {
	switch g {
	case Correctness, Concurrency:
		return Forbid
	case Slop, Complexity:
		return Deny
	case Perf, Style, Suspicious, Portability:
		return Warn
	case Pedantic:
		return Allow
	}
	return Deny
}

// Kind is the source language of the package under lint.
type Kind uint8

const (
	// KindLgo is a package of loGo/s sources.
	KindLgo Kind = iota
	// KindGo is a package of Go sources, linted only by logos lint.
	KindGo
)
