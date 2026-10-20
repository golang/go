// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package capabilities evaluates the logos.toml [capabilities] policy for
// the loGo/s capabilities auditor (M12).
//
// The auditor denies transitive dependencies that import "unsafe" (this
// package, enforced by cmd/go at load time) and transitive dependencies
// that reach reflect, exec, net, or cgo sinks (enforced by the linker,
// M12-S2). The standard library is exempt and the policy applies to Go and
// loGo/s modules alike under the logos command; plain go runs no audit.
//
// This package stays free of cmd/go internals so its policy core is unit
// testable: callers project their package graph onto Node values and pass
// the parsed severities and allowlists alongside.
package capabilities

import (
	"fmt"
	"path"
	"slices"
	"strings"
)

// Severity names a policy severity: "deny" fails the build,
// "warn" prints a diagnostic and continues, "allow" stays silent.
type Severity string

// Severities of the capabilities policy.
const (
	Deny  Severity = "deny"
	Warn  Severity = "warn"
	Allow Severity = "allow"
)

// ParseSeverity validates a severity value from logos.toml.
func ParseSeverity(s string) (Severity, error) {
	switch Severity(s) {
	case Deny, Warn, Allow:
		return Severity(s), nil
	}
	return "", fmt.Errorf("invalid severity %q (want deny, warn, or allow)", s)
}

// Node is the auditor's view of one loaded package: its import path, the
// resolved import paths it uses, and whether it is standard library or in
// a main module. Main-module packages seed the import-chain search and are
// never blamed; standard-library packages are exempt.
type Node struct {
	ImportPath string
	Imports    []string
	Goroot     bool
	Main       bool
}

// Violation is one transitive dependency importing "unsafe": the blamed
// package path and the shortest import chain from a main-module package,
// ending in "unsafe".
type Violation struct {
	Dep   string
	Chain []string
}

// CheckUnsafe walks the package graph from main-module packages and
// reports every transitive dependency that imports "unsafe" and matches
// no allowlist glob. An "allow" severity skips the walk and reports
// nothing. The standard library, main-module packages, and module-less
// packages (command-line-arguments file lists) are never blamed.
func CheckUnsafe(sev Severity, allowlist []string, all []Node) []Violation {
	if sev == Allow {
		return nil
	}
	byPath := make(map[string]Node, len(all))
	for _, n := range all {
		if _, ok := byPath[n.ImportPath]; !ok {
			byPath[n.ImportPath] = n
		}
	}
	parent := make(map[string]string, len(all))
	var queue []string
	for _, n := range all {
		if n.Main {
			if _, ok := parent[n.ImportPath]; !ok {
				parent[n.ImportPath] = ""
				queue = append(queue, n.ImportPath)
			}
		}
	}
	var out []Violation
	seen := make(map[string]bool)
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		n, ok := byPath[cur]
		if !ok {
			continue
		}
		if !exempt(n) && !seen[n.ImportPath] && slices.Contains(n.Imports, "unsafe") &&
			!Allowlisted(allowlist, n.ImportPath) {
			seen[n.ImportPath] = true
			chain := chainTo(parent, n.ImportPath)
			chain = append(chain, "unsafe")
			out = append(out, Violation{Dep: n.ImportPath, Chain: chain})
		}
		for _, imp := range n.Imports {
			if _, ok := parent[imp]; !ok {
				parent[imp] = cur
				queue = append(queue, imp)
			}
		}
	}
	slices.SortFunc(out, func(a, b Violation) int { return strings.Compare(a.Dep, b.Dep) })
	return out
}

// exempt reports whether n is never blamed: standard library, main-module
// packages (first-party code), and module-less file lists.
func exempt(n Node) bool {
	return n.Goroot || n.Main || n.ImportPath == "" || n.ImportPath == "unsafe"
}

// chainTo rebuilds the shortest import chain from a main-module package
// to dep through the BFS parent map.
func chainTo(parent map[string]string, dep string) []string {
	var rev []string
	for cur := dep; cur != ""; cur = parent[cur] {
		rev = append(rev, cur)
	}
	slices.Reverse(rev)
	return rev
}

// Allowlisted reports whether importPath matches one of the allowlist
// globs. A trailing "/**" matches the prefix itself and everything under
// it; a "**" anywhere else matches any suffix including path separators;
// other patterns match by path.Match (*, ?, [...] classes).
func Allowlisted(allowlist []string, importPath string) bool {
	for _, pattern := range allowlist {
		if matchGlob(pattern, importPath) {
			return true
		}
	}
	return false
}

// matchGlob matches one allowlist glob against one import path. A trailing
// "/**" matches the prefix itself and everything under it; any other "**"
// segment spans zero or more segments; remaining segments match without
// crossing slashes.
func matchGlob(pattern, importPath string) bool {
	if prefix, ok := strings.CutSuffix(pattern, "/**"); ok {
		if importPath == prefix || strings.HasPrefix(importPath, prefix+"/") {
			return true
		}
	}
	if !strings.Contains(pattern, "**") {
		ok, err := path.Match(pattern, importPath)
		return err == nil && ok
	}
	return matchSegPatterns(strings.Split(pattern, "/"), strings.Split(importPath, "/"))
}

// matchSegPatterns matches glob segments against path segments, where a
// "**" segment spans zero or more segments and every other segment matches
// exactly one path segment.
func matchSegPatterns(pats, segs []string) bool {
	if len(pats) == 0 {
		return len(segs) == 0
	}
	if pats[0] == "**" {
		for i := 0; i <= len(segs); i++ {
			if matchSegPatterns(pats[1:], segs[i:]) {
				return true
			}
		}
		return false
	}
	if len(segs) == 0 {
		return false
	}
	ok, err := path.Match(pats[0], segs[0])
	if err != nil || !ok {
		return false
	}
	return matchSegPatterns(pats[1:], segs[1:])
}

// FormatViolation renders one violation in the auditor report format.
func FormatViolation(v Violation) string {
	var b strings.Builder
	fmt.Fprintf(&b, "ERROR [capabilities::unauthorized_unsafe]: Transitive dependency '%s' imports 'unsafe'\n", v.Dep)
	b.WriteString("  --> Import chain (cmd/go capabilities audit):\n")
	for i, frame := range v.Chain {
		if i == 0 {
			fmt.Fprintf(&b, "   %d. %s\n", i+1, frame)
			continue
		}
		fmt.Fprintf(&b, "   %d.   -> %s\n", i+1, frame)
	}
	fmt.Fprintf(&b, " = Help: Remove the dependency or explicitly authorize \"%s/**\" under 'transitive_unsafe_allowlist' in logos.toml.", v.Dep)
	return b.String()
}

// FormatWarning renders the warn-severity diagnostic for one violation.
func FormatWarning(v Violation) string {
	return "go: warning: " + FormatViolation(v)
}
