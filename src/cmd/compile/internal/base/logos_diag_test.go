// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// TestSplitRuleTag checks the rule ID split of a compiler message.
func TestSplitRuleTag(t *testing.T) {
	for _, tc := range []struct{ msg, rule, rest string }{
		{"[correctness::const_func_impure] call is impure", "correctness::const_func_impure", "call is impure"},
		{"undefined: x", "", "undefined: x"},
		{"[not a rule] text", "", "[not a rule] text"},
		{"[slop::discarded_error]", "", "[slop::discarded_error]"},
		{"x [slop::discarded_error] y", "", "x [slop::discarded_error] y"},
	} {
		rule, rest := splitRuleTag(tc.msg)
		if rule != tc.rule || rest != tc.rest {
			t.Errorf("splitRuleTag(%q) = %q, %q, want %q, %q", tc.msg, rule, rest, tc.rule, tc.rest)
		}
	}
}

// TestCompilerSeverity checks the severity of a compiler error that carries
// no resolved lint level.
func TestCompilerSeverity(t *testing.T) {
	for _, tc := range []struct{ rule, want string }{
		{"", "deny"},
		{"slop::discarded_error", "deny"},
		{"correctness::x", "forbid"},
		{"concurrency::x", "forbid"},
	} {
		if got := compilerSeverity(tc.rule); got != tc.want {
			t.Errorf("compilerSeverity(%q) = %q, want %q", tc.rule, got, tc.want)
		}
	}
}

// record decodes one JSON diagnostic line.
func record(t *testing.T, line string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		t.Fatalf("not JSON: %v\n%s", err, line)
	}
	return m
}

// TestLogosRecordFields checks the SPEC Sec 16 field set of a compiler
// diagnostic and of a lint finding.
func TestLogosRecordFields(t *testing.T) {
	wantKeys := []string{"autofix_candidates", "file", "message", "provenance", "rule_id", "schema_version", "severity", "span"}
	compiler := record(t, logosRecord(errorMsg{logos: logosMsg{raw: "undefined: x"}}))
	lint := record(t, logosRecord(errorMsg{logos: logosMsg{raw: "m", diag: &LogosDiag{
		Rule: "slop::discarded_error", Severity: "deny", Message: "m",
		OriginExpr: "f()", OriginLine: 3, InferredType: "error",
		Fixes: []LogosFix{{Description: "d", Replacement: "f()?"}, {Description: "i", Replacement: "x", InsertBeforeLine: 4}},
	}}}))
	for name, m := range map[string]map[string]any{"compiler": compiler, "lint": lint} {
		keys := slices.Sorted(maps.Keys(m))
		if !reflect.DeepEqual(keys, wantKeys) {
			t.Errorf("%s keys = %v, want %v", name, keys, wantKeys)
		}
		if m["schema_version"] != "1.0" {
			t.Errorf("%s schema_version = %v", name, m["schema_version"])
		}
	}
	if compiler["rule_id"] != "" || compiler["severity"] != "deny" {
		t.Errorf("compiler record = %v", compiler)
	}
	fixes := lint["autofix_candidates"].([]any)
	if len(fixes) != 2 {
		t.Fatalf("fixes = %v", fixes)
	}
	if _, has := fixes[0].(map[string]any)["insert_before_line"]; has {
		t.Errorf("replace candidate carries insert_before_line: %v", fixes[0])
	}
	if fixes[1].(map[string]any)["insert_before_line"] != float64(4) {
		t.Errorf("insert candidate = %v", fixes[1])
	}
	prov := lint["provenance"].(map[string]any)
	if prov["origin_expr"] != "f()" || prov["origin_line"] != float64(3) || prov["inferred_type"] != "error" {
		t.Errorf("provenance = %v", prov)
	}
	if got := record(t, logosRecord(errorMsg{logos: logosMsg{raw: "[correctness::x] m"}}))["rule_id"]; got != "correctness::x" {
		t.Errorf("tagged compiler message rule_id = %v", got)
	}
	if got := record(t, logosRecord(errorMsg{logos: logosMsg{raw: "m", warn: true}}))["severity"]; got != "warn" {
		t.Errorf("warning severity = %v", got)
	}
}

// TestLogosRecordEmptyFixes checks that a record with no candidates
// encodes an empty array, never null.
func TestLogosRecordEmptyFixes(t *testing.T) {
	line := logosRecord(errorMsg{logos: logosMsg{raw: "m"}})
	if !strings.Contains(line, `"autofix_candidates":[]`) {
		t.Errorf("no empty array in %s", line)
	}
}

// TestByteOffset checks line and column to byte offset on a real file.
func TestByteOffset(t *testing.T) {
	name := filepath.Join(t.TempDir(), "x.go")
	if err := os.WriteFile(name, []byte("ab\ncd\n\tef\n"), 0o666); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		line, col uint
		want      int
		ok        bool
	}{
		{1, 1, 0, true},
		{2, 2, 4, true},
		{3, 2, 7, true},
		{9, 1, 0, false},
		{1, 0, 0, false},
	} {
		got, ok := byteOffset(name, tc.line, tc.col)
		if got != tc.want || ok != tc.ok {
			t.Errorf("byteOffset(%d, %d) = %d, %v, want %d, %v", tc.line, tc.col, got, ok, tc.want, tc.ok)
		}
	}
	if _, ok := byteOffset(filepath.Join(t.TempDir(), "missing.go"), 1, 1); ok {
		t.Error("byteOffset of an unreadable file succeeded")
	}
}
