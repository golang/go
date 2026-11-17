// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package capabilities

import (
	"cmp"
	"encoding/json"
	"fmt"
	"path"
	"slices"
	"strings"
)

type Severity string

const (
	Deny  Severity = "deny"
	Warn  Severity = "warn"
	Allow Severity = "allow"
)

const (
	RuleUnsafe   = "unauthorized_unsafe"
	RuleAssembly = "unclassified_assembly"
)

var ruleDetail = map[string]string{
	RuleUnsafe:   "imports 'unsafe'",
	RuleAssembly: "holds assembly files",
}

type Node struct {
	ImportPath string
	Imports    []string
	Goroot     bool
	Main       bool
	Asm        bool
}

type Violation struct {
	Dep   string
	Rule  string
	Chain []string
}

func Check(sev Severity, allowlist []string, all []Node) []Violation {
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
		if !exempt(n) && !seen[n.ImportPath] && !Allowlisted(allowlist, n.ImportPath) {
			seen[n.ImportPath] = true
			out = appendViolations(out, n, chainTo(parent, n.ImportPath))
		}
		for _, imp := range n.Imports {
			if _, ok := parent[imp]; !ok {
				parent[imp] = cur
				queue = append(queue, imp)
			}
		}
	}
	slices.SortFunc(out, func(a, b Violation) int {
		return cmp.Or(strings.Compare(a.Dep, b.Dep), strings.Compare(a.Rule, b.Rule))
	})
	return out
}

func appendViolations(out []Violation, n Node, chain []string) []Violation {
	if slices.Contains(n.Imports, "unsafe") {
		out = append(out, Violation{Dep: n.ImportPath, Rule: RuleUnsafe, Chain: append(slices.Clone(chain), "unsafe")})
	}
	if n.Asm {
		out = append(out, Violation{Dep: n.ImportPath, Rule: RuleAssembly, Chain: chain})
	}
	return out
}

func exempt(n Node) bool {
	return n.Goroot || n.Main || n.ImportPath == "" || n.ImportPath == "unsafe"
}

func chainTo(parent map[string]string, dep string) []string {
	var rev []string
	for cur := dep; cur != ""; cur = parent[cur] {
		rev = append(rev, cur)
	}
	slices.Reverse(rev)
	return rev
}

func Allowlisted(allowlist []string, importPath string) bool {
	for _, pattern := range allowlist {
		if matchGlob(pattern, importPath) {
			return true
		}
	}
	return false
}

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

func FormatViolation(v Violation) string {
	var b strings.Builder
	fmt.Fprintf(&b, "ERROR [capabilities::%s]: Transitive dependency '%s' %s\n", v.Rule, v.Dep, ruleDetail[v.Rule])
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

func FormatWarning(v Violation) string {
	return "go: warning: " + FormatViolation(v)
}

// jsonRecord mirrors the Sec 14 diagnostics schema emitted by
// cmd/compile under -logos.json, so load-time diagnostics stay
// line-compatible with compiler diagnostics.
type jsonSpan struct {
	StartLine  int `json:"start_line"`
	StartCol   int `json:"start_col"`
	EndLine    int `json:"end_line"`
	EndCol     int `json:"end_col"`
	ByteOffset int `json:"byte_offset"`
	ByteLen    int `json:"byte_len"`
}

type jsonProvenance struct {
	OriginExpr   string `json:"origin_expr"`
	OriginLine   int    `json:"origin_line"`
	InferredType string `json:"inferred_type"`
}

type jsonRecord struct {
	SchemaVersion     string         `json:"schema_version"`
	Severity          string         `json:"severity"`
	RuleID            string         `json:"rule_id"`
	File              string         `json:"file"`
	Span              jsonSpan       `json:"span"`
	Message           string         `json:"message"`
	Note              string         `json:"note,omitzero"`
	Provenance        jsonProvenance `json:"provenance"`
	AutofixCandidates []any          `json:"autofix_candidates"`
}

// FormatJSON renders one violation as a single-line Sec 14 JSON record.
// severity is "deny", "forbid", or "warn". file is the chain-head
// package's source path; an import path is not a document and cannot
// anchor an editor diagnostic.
func FormatJSON(v Violation, severity, file string) string {
	var note strings.Builder
	for i, frame := range v.Chain {
		if i == 0 {
			fmt.Fprintf(&note, "%d. %s\n", i+1, frame)
			continue
		}
		fmt.Fprintf(&note, "%d.   -> %s\n", i+1, frame)
	}
	fmt.Fprintf(&note, " = Help: Remove the dependency or explicitly authorize \"%s/**\" under 'transitive_unsafe_allowlist' in logos.toml.", v.Dep)
	rec := jsonRecord{
		SchemaVersion:     "1.0",
		Severity:          severity,
		RuleID:            "capabilities::" + v.Rule,
		File:              file,
		Message:           fmt.Sprintf("Transitive dependency '%s' %s", v.Dep, ruleDetail[v.Rule]),
		Note:              note.String(),
		AutofixCandidates: []any{},
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return ""
	}
	return string(b)
}
