// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"internal/testenv"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestLogosJSONDiagnostics checks the --json-diagnostics records of logos
// build and logos lint as decoded JSON. The script tests cover the text
// output and exit statuses; this test pins every structural field the
// scripts used to check with jaq, decoding each stderr line with
// encoding/json so key order cannot matter.
func TestLogosJSONDiagnostics(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	t.Run("Build", func(t *testing.T) {
		dir := logosJSONFixture(t, map[string]string{
			"go.mod":   "module example.com/jsonbuild\n\ngo 1.27\n",
			"main.lgo": "package main\n\nimport \"os\"\n\nfunc main() {\n\tos.Remove(\"missing\")\n}\n",
		})
		out := filepath.Join(dir, "bin")
		if err := os.MkdirAll(out, 0o777); err != nil {
			t.Fatal(err)
		}
		code, records := runLogosJSON(t, dir, "build", "--json-diagnostics", "-o", filepath.Join(out, "jsonbuild"+exeSuffix), ".")
		if code == 0 {
			t.Errorf("logos build exit status 0, want nonzero")
		}
		if len(records) != 1 {
			t.Fatalf("got %d records, want 1: %v", len(records), records)
		}
		m := records[0]
		checkLogosEnvelope(t, m, "main.lgo")
		checkLogosSeverity(t, m, "deny")
		checkLogosRule(t, m, "slop::discarded_error")
		span := checkLogosSpanKeys(t, m)
		if got := span["start_line"]; got != float64(6) {
			t.Errorf("span.start_line = %v, want 6", got)
		}
		if got := span["start_col"]; got != float64(2) {
			t.Errorf("span.start_col = %v, want 2", got)
		}
		if got := logosJSONNumber(t, span, "byte_offset"); got <= 0 {
			t.Errorf("span.byte_offset = %v, want > 0", got)
		}
		prov := checkLogosProvenanceKeys(t, m)
		origin, ok := prov["origin_expr"].(string)
		if !ok {
			t.Fatalf("provenance.origin_expr is not a string: %v", prov["origin_expr"])
		}
		if got := logosJSONNumber(t, span, "byte_len"); got != float64(len(origin)) {
			t.Errorf("span.byte_len = %v, want len(origin_expr) = %d", got, len(origin))
		}
		if got, want := logosJSONNumber(t, span, "end_col"), logosJSONNumber(t, span, "start_col")+logosJSONNumber(t, span, "byte_len"); got != want {
			t.Errorf("span.end_col = %v, want start_col + byte_len = %v", got, want)
		}
		checkLogosMessage(t, m)
		if _, ok := m["autofix_candidates"].([]any); !ok {
			t.Errorf("autofix_candidates is not an array: %v", m["autofix_candidates"])
		}
	})

	t.Run("Lint", func(t *testing.T) {
		dir := logosJSONFixture(t, map[string]string{
			"go.mod":   "module example.com/jsonlint\n\ngo 1.27\n",
			"main.lgo": "package main\n\n//logos:allow(correctness::infinite_recursion)\nfunc loop(n int) int {\n\treturn loop(n)\n}\n\nfunc main() {\n\tprintln(loop(1))\n}\n",
		})
		code, records := runLogosJSON(t, dir, "lint", "--json-diagnostics", "./...")
		if code == 0 {
			t.Errorf("logos lint exit status 0, want nonzero")
		}
		if len(records) != 2 {
			t.Fatalf("got %d records, want 2: %v", len(records), records)
		}
		for _, m := range records {
			checkLogosEnvelope(t, m, "main.lgo")
			if _, ok := m["rule_id"].(string); !ok {
				t.Errorf("rule_id is not a string: %v", m["rule_id"])
			}
		}
		checkLogosAnyRecord(t, records, "correctness::allow_forbidden_rule", "forbid")
		checkLogosAnyRecord(t, records, "correctness::infinite_recursion", "forbid")
	})

	t.Run("Warn", func(t *testing.T) {
		dir := logosJSONFixture(t, map[string]string{
			"go.mod":     "module example.com/jsonwarn\n\ngo 1.27\n",
			"logos.toml": "[lints]\nslop = \"warn\"\n",
			"main.lgo":   "package main\n\nimport \"os\"\n\nfunc main() {\n\tos.Remove(\"missing\")\n}\n",
		})
		code, records := runLogosJSON(t, dir, "lint", "--json-diagnostics", "./...")
		if code != 0 {
			t.Errorf("logos lint exit status %d, want 0", code)
		}
		if len(records) != 1 {
			t.Fatalf("got %d records, want 1: %v", len(records), records)
		}
		checkLogosEnvelope(t, records[0], "main.lgo")
		checkLogosSeverity(t, records[0], "warn")
		checkLogosRule(t, records[0], "slop::discarded_error")
	})

	t.Run("GoTypeError", func(t *testing.T) {
		dir := logosJSONFixture(t, map[string]string{
			"go.mod":  "module example.com/gotypeerr\n\ngo 1.27\n",
			"main.go": "package main\n\nfunc main() {\n\tundefinedFn()\n}\n",
		})
		code, records := runLogosJSON(t, dir, "build", "--json-diagnostics", "./...")
		if code == 0 {
			t.Errorf("logos build exit status 0, want nonzero")
		}
		if len(records) != 1 {
			t.Fatalf("got %d records, want 1: %v", len(records), records)
		}
		m := records[0]
		checkLogosEnvelope(t, m, ".go")
		checkLogosSeverity(t, m, "deny")
		checkLogosRule(t, m, "")
		msg, ok := m["message"].(string)
		if !ok || !strings.Contains(msg, "undefined") {
			t.Errorf("message = %v, want a string containing %q", m["message"], "undefined")
		}
	})

	t.Run("GoLint", func(t *testing.T) {
		dir := logosJSONFixture(t, map[string]string{
			"go.mod":     "module example.com/dropped\n\ngo 1.27\n",
			"main.go":    "package main\n\nimport (\n\t\"os\"\n\n\t\"example.com/dropped/lib\"\n)\n\nfunc run() {\n\tos.Remove(\"missing\")\n\tgo work()\n}\n\nfunc work() {}\n\nfunc main() {\n\trun()\n\tprintln(lib.Name())\n}\n",
			"lib/lib.go": "package lib\n\nfunc Name() string { return \"lib\" }\n",
		})
		code, records := runLogosJSON(t, dir, "lint", "--json-diagnostics", "./...")
		if code == 0 {
			t.Errorf("logos lint exit status 0, want nonzero")
		}
		if len(records) != 1 {
			t.Fatalf("got %d records, want 1: %v", len(records), records)
		}
		m := records[0]
		checkLogosEnvelope(t, m, "main.go")
		checkLogosSeverity(t, m, "deny")
		checkLogosRule(t, m, "slop::discarded_error")
		span := logosJSONSpan(t, m)
		if got := span["start_line"]; got != float64(10) {
			t.Errorf("span.start_line = %v, want 10", got)
		}
		prov, ok := m["provenance"].(map[string]any)
		if !ok {
			t.Fatalf("provenance is not an object: %v", m["provenance"])
		}
		origin, ok := prov["origin_expr"].(string)
		if !ok {
			t.Fatalf("provenance.origin_expr is not a string: %v", prov["origin_expr"])
		}
		if got := logosJSONNumber(t, span, "byte_len"); got != float64(len(origin)) {
			t.Errorf("span.byte_len = %v, want len(origin_expr) = %d", got, len(origin))
		}
	})

	t.Run("GoCappedForbid", func(t *testing.T) {
		dir := logosJSONFixture(t, map[string]string{
			"go.mod":  "module example.com/recursive\n\ngo 1.27\n",
			"main.go": "package main\n\nfunc loop(n int) int {\n\treturn loop(n)\n}\n\nfunc main() {\n\tprintln(loop(1))\n}\n",
		})
		code, records := runLogosJSON(t, dir, "lint", "--json-diagnostics", "./...")
		if code == 0 {
			t.Errorf("logos lint exit status 0, want nonzero")
		}
		if len(records) != 1 {
			t.Fatalf("got %d records, want 1: %v", len(records), records)
		}
		checkLogosEnvelope(t, records[0], "main.go")
		checkLogosSeverity(t, records[0], "deny")
		checkLogosRule(t, records[0], "correctness::infinite_recursion")
	})

	t.Run("Pedantic", func(t *testing.T) {
		dir := logosJSONFixture(t, map[string]string{
			"go.mod":   "module example.com/union\n\ngo 1.27\n",
			"main.lgo": "package main\n\ntype Shape interface {\n\tCircle | Square\n}\n\ntype Circle struct{ r int }\n\ntype Square struct{ s int }\n\nfunc area(sh Shape) int {\n\tswitch v := sh.(type) {\n\tcase Circle:\n\t\treturn v.r\n\tdefault:\n\t\treturn 0\n\t}\n}\n\nfunc main() {\n\tprintln(area(Circle{1}))\n}\n",
		})
		code, records := runLogosJSON(t, dir, "lint", "--pedantic", "--json-diagnostics", "./...")
		if code != 0 {
			t.Errorf("logos lint exit status %d, want 0", code)
		}
		if len(records) != 1 {
			t.Fatalf("got %d records, want 1: %v", len(records), records)
		}
		checkLogosEnvelope(t, records[0], "main.lgo")
		checkLogosSeverity(t, records[0], "warn")
		checkLogosRule(t, records[0], "pedantic::exhaustive_union_switch")
		if got := logosJSONSpan(t, records[0])["start_line"]; got != float64(15) {
			t.Errorf("span.start_line = %v, want 15", got)
		}
	})
}

// logosJSONFixture writes one fixture module into a fresh directory.
func logosJSONFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// runLogosJSON runs the test logos binary with args in dir and decodes
// every stderr line as one JSON diagnostic object.
func runLogosJSON(t *testing.T, dir string, args ...string) (int, []map[string]any) {
	t.Helper()
	cmd := exec.Command(testLogos, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOTOOLCHAIN=local")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	code := 0
	if err := cmd.Run(); err != nil {
		exit, ok := errors.AsType[*exec.ExitError](err)
		if !ok {
			t.Fatalf("running logos %v: %v\nstderr:\n%s", args, err, stderr.String())
		}
		code = exit.ExitCode()
	}
	text := strings.TrimSpace(stderr.String())
	if text == "" {
		t.Fatalf("logos %v printed no diagnostics on stderr (stdout: %q)", args, stdout.String())
	}
	var records []map[string]any
	for line := range strings.SplitSeq(text, "\n") {
		var m map[string]any
		dec := json.NewDecoder(strings.NewReader(line))
		if err := dec.Decode(&m); err != nil {
			t.Fatalf("stderr line is not one JSON object: %v\nline: %s\nfull stderr:\n%s", err, line, stderr.String())
		}
		var extra any
		if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
			t.Fatalf("stderr line holds more than one JSON object:\n%s", line)
		}
		records = append(records, m)
	}
	return code, records
}

// checkLogosEnvelope checks the record keys, schema version, file name
// suffix, and that the message is a non-empty string.
func checkLogosEnvelope(t *testing.T, m map[string]any, suffix string) {
	t.Helper()
	checkLogosEnvelopeSuffix(t, m, suffix)
	if m["schema_version"] != "1.0" {
		t.Errorf("schema_version = %v, want %q", m["schema_version"], "1.0")
	}
}

// checkLogosEnvelopeSuffix checks the record keys, file name suffix,
// message text, and the nested span, provenance, and autofix shapes.
func checkLogosEnvelopeSuffix(t *testing.T, m map[string]any, suffix string) {
	t.Helper()
	checkLogosKeys(t, m, "diagnostic", "autofix_candidates", "file", "message", "provenance", "rule_id", "schema_version", "severity", "span")
	file, ok := m["file"].(string)
	if !ok || !strings.HasSuffix(file, suffix) {
		t.Errorf("file = %v, want a string ending in %q", m["file"], suffix)
	}
	msg, ok := m["message"].(string)
	if !ok || msg == "" {
		t.Errorf("message = %v, want a non-empty string", m["message"])
	}
	span := logosJSONSpan(t, m)
	checkLogosKeys(t, span, "span", "byte_len", "byte_offset", "end_col", "end_line", "start_col", "start_line")
	for _, key := range []string{"byte_len", "byte_offset", "end_col", "end_line", "start_col", "start_line"} {
		if _, ok := span[key].(float64); !ok {
			t.Errorf("span.%s = %v, want a number", key, span[key])
		}
	}
	prov, ok := m["provenance"].(map[string]any)
	if !ok {
		t.Fatalf("provenance is not an object: %v", m["provenance"])
	}
	checkLogosKeys(t, prov, "provenance", "inferred_type", "origin_expr", "origin_line")
	if _, ok := prov["origin_expr"].(string); !ok {
		t.Errorf("provenance.origin_expr = %v, want a string", prov["origin_expr"])
	}
	if _, ok := prov["origin_line"].(float64); !ok {
		t.Errorf("provenance.origin_line = %v, want a number", prov["origin_line"])
	}
	if _, ok := prov["inferred_type"].(string); !ok {
		t.Errorf("provenance.inferred_type = %v, want a string", prov["inferred_type"])
	}
	if _, ok := m["autofix_candidates"].([]any); !ok {
		t.Errorf("autofix_candidates is not an array: %v", m["autofix_candidates"])
	}
}

// checkLogosSeverity checks the severity field of one record.
func checkLogosSeverity(t *testing.T, m map[string]any, want string) {
	t.Helper()
	if m["severity"] != want {
		t.Errorf("severity = %v, want %q", m["severity"], want)
	}
}

// checkLogosRule checks the rule_id field of one record.
func checkLogosRule(t *testing.T, m map[string]any, want string) {
	t.Helper()
	if m["rule_id"] != want {
		t.Errorf("rule_id = %v, want %q", m["rule_id"], want)
	}
}

// checkLogosMessage checks that the message field is a non-empty string.
func checkLogosMessage(t *testing.T, m map[string]any) {
	t.Helper()
	msg, ok := m["message"].(string)
	if !ok || msg == "" {
		t.Errorf("message = %v, want a non-empty string", m["message"])
	}
}

// checkLogosSpanKeys checks the exact key set of the span object.
func checkLogosSpanKeys(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	span := logosJSONSpan(t, m)
	checkLogosKeys(t, span, "span", "byte_len", "byte_offset", "end_col", "end_line", "start_col", "start_line")
	return span
}

// checkLogosProvenanceKeys checks the exact key set of the provenance object.
func checkLogosProvenanceKeys(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	prov, ok := m["provenance"].(map[string]any)
	if !ok {
		t.Fatalf("provenance is not an object: %v", m["provenance"])
	}
	checkLogosKeys(t, prov, "provenance", "inferred_type", "origin_expr", "origin_line")
	return prov
}

// checkLogosAnyRecord checks that some record carries rule and severity.
func checkLogosAnyRecord(t *testing.T, records []map[string]any, rule, severity string) {
	t.Helper()
	for _, m := range records {
		if m["rule_id"] == rule && m["severity"] == severity {
			return
		}
	}
	t.Errorf("no record with rule_id %q and severity %q in %v", rule, severity, records)
}

// checkLogosKeys checks that m holds exactly the wanted keys.
func checkLogosKeys(t *testing.T, m map[string]any, what string, want ...string) {
	t.Helper()
	for _, k := range want {
		if _, ok := m[k]; !ok {
			t.Errorf("%s is missing key %q (keys: %s)", what, k, logosJSONKeyList(m))
		}
	}
	if len(m) != len(want) {
		t.Errorf("%s has %d keys (%s), want %d", what, len(m), logosJSONKeyList(m), len(want))
	}
}

// logosJSONSpan returns the span object of one record.
func logosJSONSpan(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	span, ok := m["span"].(map[string]any)
	if !ok {
		t.Fatalf("span is not an object: %v", m["span"])
	}
	return span
}

// logosJSONNumber returns one numeric field as a float64.
func logosJSONNumber(t *testing.T, m map[string]any, key string) float64 {
	t.Helper()
	v, ok := m[key].(float64)
	if !ok {
		t.Fatalf("%s is not a number: %v", key, m[key])
	}
	return v
}

// logosJSONKeyList lists the keys of m for failure messages.
func logosJSONKeyList(m map[string]any) string {
	return strings.Join(slices.Sorted(maps.Keys(m)), ",")
}
