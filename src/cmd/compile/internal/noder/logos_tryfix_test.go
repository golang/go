// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmp"
	"encoding/json"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

type diagSpan struct {
	ByteOffset int `json:"byte_offset"`
	ByteLen    int `json:"byte_len"`
}

type diagFix struct {
	Replacement      string `json:"replacement"`
	InsertBeforeLine int    `json:"insert_before_line"`
}

type diagProvenance struct {
	OriginExpr   string `json:"origin_expr"`
	OriginLine   int    `json:"origin_line"`
	InferredType string `json:"inferred_type"`
}

type diagRecord struct {
	RuleID            string         `json:"rule_id"`
	Severity          string         `json:"severity"`
	Span              diagSpan       `json:"span"`
	Note              string         `json:"note"`
	Provenance        diagProvenance `json:"provenance"`
	AutofixCandidates []diagFix      `json:"autofix_candidates"`
}

const (
	misplacedTryRuleID = "correctness::misplaced_try"
	severityForbid     = "forbid"
)

func logosToolPath(t *testing.T) string {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	goTool := testenv.GoToolPath(t)
	name := "logos"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	p := filepath.Join(filepath.Dir(goTool), name)
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("logos binary not found at %s: %v", p, err)
	}
	return p
}

func compileLgoJSON(t *testing.T, src string) ([]diagRecord, int) {
	t.Helper()
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()
	srcPath := filepath.Join(dir, "x.lgo")
	policyPath := filepath.Join(dir, "policy.json")
	objPath := filepath.Join(dir, "out.a")

	if err := os.WriteFile(srcPath, []byte(src), 0o666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(policyPath, []byte(`{"version":1}`), 0o666); err != nil {
		t.Fatal(err)
	}

	args := []string{
		"tool", "compile",
		"-p", "main",
		"-logos.cfg=" + policyPath,
		"-logos.json",
		"-o", objPath,
		srcPath,
	}
	cmd := testenv.Command(t, testenv.GoToolPath(t), args...)
	out, err := cmd.CombinedOutput()

	exitCode := 0
	if ee, ok := err.(*exec.ExitError); ok {
		exitCode = ee.ExitCode()
	} else if err != nil {
		t.Fatalf("compile failed: %v\noutput: %s", err, out)
	}

	var diags []diagRecord
	for line := range strings.SplitSeq(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var rec diagRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("failed to decode JSON diagnostic: %v\nline: %s", err, line)
		}
		diags = append(diags, rec)
	}
	return diags, exitCode
}

func applyFix(t *testing.T, src string, span diagSpan, fix diagFix) string {
	t.Helper()
	if fix.InsertBeforeLine > 0 {
		lines := strings.Split(src, "\n")
		idx := fix.InsertBeforeLine - 1
		if idx < 0 || idx > len(lines) {
			t.Fatalf("invalid InsertBeforeLine %d for %d lines", fix.InsertBeforeLine, len(lines))
		}
		var b strings.Builder
		for i, l := range lines {
			if i == idx {
				b.WriteString(fix.Replacement)
			}
			b.WriteString(l)
			b.WriteByte('\n')
		}
		return strings.TrimSuffix(b.String(), "\n")
	}

	start := span.ByteOffset
	end := span.ByteOffset + span.ByteLen
	if start < 0 || end > len(src) || start > end {
		t.Fatalf("invalid byte span [%d:%d] for src length %d", start, end, len(src))
	}
	return src[:start] + fix.Replacement + src[end:]
}

func runLgoSourceOut(t *testing.T, src string) string {
	t.Helper()
	testenv.MustHaveGoRun(t)

	dir := t.TempDir()
	srcPath := filepath.Join(dir, "main.lgo")
	modPath := filepath.Join(dir, "go.mod")

	if err := os.WriteFile(srcPath, []byte(src), 0o666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modPath, []byte("module testprog\n\ngo 1.27\n"), 0o666); err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, logosToolPath(t), "run", ".")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("execution failed: %v\noutput:\n%s\nsource:\n%s", err, out, src)
	}
	return string(out)
}

func misplacedTryDiag(t *testing.T, diags []diagRecord) *diagRecord {
	t.Helper()
	var found *diagRecord
	for i := range diags {
		if diags[i].RuleID == misplacedTryRuleID {
			if found != nil {
				t.Fatalf("expected one %s diagnostic, got several", misplacedTryRuleID)
			}
			found = &diags[i]
		}
	}
	if found == nil {
		t.Fatalf("expected diagnostic with rule_id %q, got: %+v", misplacedTryRuleID, diags)
	}
	return found
}

func runLgoSource(t *testing.T, src string) {
	t.Helper()
	testenv.MustHaveGoRun(t)

	dir := t.TempDir()
	srcPath := filepath.Join(dir, "main.lgo")
	modPath := filepath.Join(dir, "go.mod")

	if err := os.WriteFile(srcPath, []byte(src), 0o666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modPath, []byte("module testprog\n\ngo 1.27\n"), 0o666); err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, logosToolPath(t), "run", ".")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("execution failed: %v\noutput:\n%s\nsource:\n%s", err, out, src)
	}
}

func TestMisplacedTryAutofix(t *testing.T) {
	const safeSrc = `package main

type myError struct{}

func (myError) Error() string { return "step error" }

var (
	stepCalls int
	stepFail  bool
)

func step() (int, error) {
	stepCalls++
	if stepFail {
		return 0, myError{}
	}
	return 42, nil
}

func compute() (int, error) {
	v := step()? + 1
	return v, nil
}

func main() {
	// Verify success path: result value 43 and exactly-once evaluation
	stepCalls = 0
	stepFail = false
	val, err := compute()
	if err != nil {
		panic("unexpected error on success path")
	}
	if stepCalls != 1 {
		panic("expected callcount 1")
	}
	if val != 43 {
		panic("expected value 43")
	}

	// Verify error path: error propagation and exactly-once evaluation
	stepCalls = 0
	stepFail = true
	_, err = compute()
	if err == nil {
		panic("expected error on error path, got nil")
	}
	if err.Error() != "step error" {
		panic("unexpected error message on error path")
	}
	if stepCalls != 1 {
		panic("expected callcount 1 on error path")
	}
}
`

	diags, exitCode := compileLgoJSON(t, safeSrc)
	if exitCode == 0 {
		t.Fatalf("expected compile failure on misplaced '?', got exit code 0")
	}

	var targetDiag *diagRecord
	for i := range diags {
		if diags[i].RuleID == misplacedTryRuleID {
			targetDiag = &diags[i]
			break
		}
	}

	if targetDiag == nil {
		t.Fatalf("expected diagnostic with rule_id %q, got: %+v", misplacedTryRuleID, diags)
	}
	if targetDiag.Severity != severityForbid {
		t.Errorf("expected severity %q, got %q", severityForbid, targetDiag.Severity)
	}

	if len(targetDiag.AutofixCandidates) == 0 {
		t.Fatalf("missing autofix candidate for %s: got 0 candidates, want split statement autofix", misplacedTryRuleID)
	}

	for i, fix := range targetDiag.AutofixCandidates {
		fixedSrc := applyFix(t, safeSrc, targetDiag.Span, fix)
		runLgoSource(t, fixedSrc)
		t.Logf("candidate %d compiled and executed with verified runtime assertions", i)
	}
}

func TestMisplacedTryAutofixOrdering(t *testing.T) {
	const unsafeSrc = `package main

type myError struct{}

func (myError) Error() string { return "step error" }

var (
	trace     []string
	nextCalls int
	stepCalls int
	stepFail  bool
)

func next() int {
	nextCalls++
	trace = append(trace, "next")
	return 1
}

func step() (int, error) {
	stepCalls++
	trace = append(trace, "step")
	if stepFail {
		return 0, myError{}
	}
	return 42, nil
}

func computeUnsafe() (int, error) {
	v := next() + step()?
	return v, nil
}

func main() {
	// Verify success path: left-to-right evaluation order (next before step)
	trace = nil
	nextCalls = 0
	stepCalls = 0
	stepFail = false
	val, err := computeUnsafe()
	if err != nil {
		panic("unexpected error on success path")
	}
	if val != 43 {
		panic("expected value 43")
	}
	if nextCalls != 1 || stepCalls != 1 {
		panic("expected nextCalls=1 stepCalls=1")
	}
	if len(trace) != 2 || trace[0] != "next" || trace[1] != "step" {
		panic("evaluation order violated")
	}

	// Verify error path: next() must execute before step() fails and returns
	trace = nil
	nextCalls = 0
	stepCalls = 0
	stepFail = true
	_, err = computeUnsafe()
	if err == nil {
		panic("expected error on error path, got nil")
	}
	if err.Error() != "step error" {
		panic("unexpected error message on error path")
	}
	if nextCalls != 1 || stepCalls != 1 {
		panic("expected nextCalls=1 stepCalls=1 on error path")
	}
	if len(trace) != 2 || trace[0] != "next" || trace[1] != "step" {
		panic("evaluation order violated on error path")
	}
}
`

	diags, exitCode := compileLgoJSON(t, unsafeSrc)
	if exitCode == 0 {
		t.Fatalf("expected compile failure on unsafe misplaced '?', got exit code 0")
	}

	var targetDiag *diagRecord
	for i := range diags {
		if diags[i].RuleID == misplacedTryRuleID {
			targetDiag = &diags[i]
			break
		}
	}
	if targetDiag == nil {
		t.Fatalf("expected diagnostic with rule_id %q, got: %+v", misplacedTryRuleID, diags)
	}
	if targetDiag.Severity != severityForbid {
		t.Errorf("expected severity %q, got %q", severityForbid, targetDiag.Severity)
	}

	if len(targetDiag.AutofixCandidates) == 0 {
		t.Logf("conservative refusal: no autofix candidate offered for order-sensitive expression")
		return
	}

	for i, fix := range targetDiag.AutofixCandidates {
		fixedSrc := applyFix(t, unsafeSrc, targetDiag.Span, fix)
		runLgoSource(t, fixedSrc)
		t.Logf("candidate %d preserved evaluation order and error behavior", i)
	}
}

func TestMisplacedTrySplitOffered(t *testing.T) {
	const stepHelpers = `type myErr string

func (e myErr) Error() string { return string(e) }

var (
	calls int
	fail  bool
)

func step() (int, error) {
	calls++
	if fail {
		return 0, myErr("boom")
	}
	return 42, nil
}
`

	const mainSrc = `func main() {
	calls, fail = 0, false
	v, err := compute()
	if err != nil || v != %d {
		println("bad")
	} else {
		println("ok")
	}
	calls, fail = 0, true
	v, err = compute()
	if err == nil || err.Error() != "boom" {
		println("bad")
	} else {
		println("err")
	}
	if calls != 1 {
		println("bad")
	} else {
		println("once")
	}
}
`

	tests := []struct {
		name    string
		decls   string
		compute string
		wantVal int
	}{
		{
			name: "call_argument",
			decls: `var got int

func g(x int) error {
	got = x
	return nil
}
`,
			compute: `func compute() (int, error) {
	err := g(step()?)
	return got, err
}
`,
			wantVal: 42,
		},
		{
			name: "binary_operand",
			compute: `func compute() (int, error) {
	a := 2 + step()?
	return a, nil
}
`,
			wantVal: 44,
		},
		{
			name: "chain_argument",
			decls: `type S struct{}

var gotM int

func (S) M(x int) S {
	gotM = x
	return S{}
}

func (S) N() int {
	return gotM
}
`,
			compute: `func compute() (int, error) {
	v := S{}.M(step()?).N()
	return v, nil
}
`,
			wantVal: 42,
		},
		{
			name: "if_condition",
			compute: `func compute() (int, error) {
	if step()? == 42 {
		return 43, nil
	}
	return 0, nil
}
`,
			wantVal: 43,
		},
		{
			name: "return_value",
			compute: `func compute() (int, error) {
	return step()?, nil
}
`,
			wantVal: 42,
		},
		{
			name: "composite_literal",
			compute: `func compute() (int, error) {
	xs := []int{step()?, 1}
	return xs[0] + xs[1], nil
}
`,
			wantVal: 43,
		},
		{
			name: "multivalue_argument",
			decls: `var (
	gotN int
	gotB bool
)

func step2() (int, bool, error) {
	calls++
	if fail {
		return 0, false, myErr("boom")
	}
	return 42, true, nil
}

func g2(n int, b bool) error {
	gotN, gotB = n, b
	return nil
}
`,
			compute: `func compute() (int, error) {
	err := g2(step2()?)
	return gotN, err
}
`,
			wantVal: 42,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := "package main\n\n" + tc.decls + "\n" + stepHelpers + tc.compute + fmt.Sprintf(mainSrc, tc.wantVal)
			diags, exitCode := compileLgoJSON(t, src)
			if exitCode == 0 {
				t.Fatalf("expected compile failure on misplaced '?', got exit code 0")
			}
			d := misplacedTryDiag(t, diags)
			if d.Severity != severityForbid {
				t.Errorf("expected severity %q, got %q", severityForbid, d.Severity)
			}
			if len(d.AutofixCandidates) != 1 {
				t.Fatalf("expected exactly one autofix candidate for %s position, got %d", tc.name, len(d.AutofixCandidates))
			}
			fix := d.AutofixCandidates[0]
			if strings.Contains(fix.Replacement, "_logos_") || strings.Contains(fix.Replacement, "Logos_") {
				t.Fatalf("candidate replacement uses a reserved prefix: %q", fix.Replacement)
			}
			fixedSrc := applyFix(t, src, d.Span, fix)
			got := runLgoSourceOut(t, fixedSrc)
			const wantOut = "ok\nerr\nonce\n"
			if got != wantOut {
				t.Fatalf("applied candidate output = %q, want %q\nfixed source:\n%s", got, wantOut, fixedSrc)
			}
		})
	}
}

func TestMisplacedTrySplitRefused(t *testing.T) {
	const stepHelpers = `type myErr string

func (e myErr) Error() string { return string(e) }

func step() (int, error) {
	if fail {
		return 0, myErr("boom")
	}
	return 42, nil
}

var fail bool

func g() int { return 5 }
`

	tests := []struct {
		name    string
		compute string
	}{
		{
			name: "right_of_call",
			compute: `func compute() (int, error) {
	a := g() + step()?
	return a, nil
}
`,
		},
		{
			name: "short_circuit",
			compute: `func compute() (int, error) {
	if fail && step()? > 0 {
		return 43, nil
	}
	return 0, nil
}
`,
		},
		{
			name: "loop_condition",
			compute: `func compute() (int, error) {
	for step()? > 100 {
		break
	}
	return 0, nil
}
`,
		},
		{
			name: "case_expression",
			compute: `func compute() (int, error) {
	switch step()? {
	case 42:
		return 43, nil
	}
	return 0, nil
}
`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := "package main\n\n" + stepHelpers + tc.compute
			diags, exitCode := compileLgoJSON(t, src)
			if exitCode == 0 {
				t.Fatalf("expected compile failure on misplaced '?', got exit code 0")
			}
			d := misplacedTryDiag(t, diags)
			if len(d.AutofixCandidates) != 0 {
				t.Fatalf("expected no autofix candidate for %s position, got %d: %+v", tc.name, len(d.AutofixCandidates), d.AutofixCandidates)
			}
		})
	}
}
func TestMisplacedTrySplitPositions(t *testing.T) {
	const stepHelpers = `type myErr string

func (e myErr) Error() string { return string(e) }

var (
	calls int
	fail bool
	got int
	gotM int
)

func step() (int, error) {
	calls++
	if fail {
		return 0, myErr("boom")
	}
	return 42, nil
}

func step2() (int, bool, error) {
	calls++
	if fail {
		return 0, false, myErr("boom")
	}
	return 42, true, nil
}

func g(x int) error {
	got = x
	return nil
}

func gInt(x int) int { return x }
`
	const stdMain = `func main() {
	calls, fail = 0, false
	v, err := compute()
	if err != nil || v != %d {
		println("bad")
	} else {
		println("ok")
	}
	calls, fail = 0, true
	v, err = compute()
	if err == nil || err.Error() != "boom" {
		println("bad")
	} else {
		println("err")
	}
	if calls != 1 {
		println("bad")
	} else {
		println("once")
	}
}
`
	tests := []struct {
		name      string
		decls     string
		misplaced string
		hand      string
		mainSrc   string
		wantVal   int
		wantOffer bool
	}{
		{
			name:      "call_argument",
			misplaced: "func compute() (int, error) {\n\terr := g(step()?)\n\treturn got, err\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\terr := g(_v)\n\treturn got, err\n}\n",
			wantVal:   42,
			wantOffer: true,
		},
		{
			name:      "binary_operand",
			misplaced: "func compute() (int, error) {\n\ta := 2 + step()?\n\treturn a, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\ta := 2 + _v\n\treturn a, nil\n}\n",
			wantVal:   44,
			wantOffer: true,
		},
		{
			name:      "chain_argument",
			decls:     "type S struct{}\n\nfunc (S) M(x int) S {\n\tgotM = x\n\treturn S{}\n}\n\nfunc (S) N() int {\n\treturn gotM\n}\n",
			misplaced: "func compute() (int, error) {\n\tv := S{}.M(step()?).N()\n\treturn v, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\tv := S{}.M(_v).N()\n\treturn v, nil\n}\n",
			wantVal:   42,
			wantOffer: true,
		},
		{
			name:      "composite_literal",
			misplaced: "func compute() (int, error) {\n\txs := []int{step()?, 1}\n\treturn xs[0] + xs[1], nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\txs := []int{_v, 1}\n\treturn xs[0] + xs[1], nil\n}\n",
			wantVal:   43,
			wantOffer: true,
		},
		{
			name:      "if_condition",
			misplaced: "func compute() (int, error) {\n\tif step()? == 42 {\n\t\treturn 43, nil\n\t}\n\treturn 0, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\tif _v == 42 {\n\t\treturn 43, nil\n\t}\n\treturn 0, nil\n}\n",
			wantVal:   43,
			wantOffer: true,
		},
		{
			name:      "return_partial",
			misplaced: "func compute() (int, error) {\n\treturn step()?, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\treturn _v, nil\n}\n",
			wantVal:   42,
			wantOffer: true,
		},
		{
			name:      "return_all_single",
			misplaced: "func compute() (int, error) {\n\treturn step()?\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\treturn _v, nil\n}\n",
			wantVal:   42,
			wantOffer: true,
		},
		{
			name:      "return_all_multi",
			misplaced: "func compute() (int, bool, error) {\n\treturn step2()?\n}\n",
			hand:      "func compute() (int, bool, error) {\n\t_a, _b, _e := step2()\n\tif _e != nil {\n\t\treturn 0, false, _e\n\t}\n\treturn _a, _b, nil\n}\n",
			mainSrc:   "func main() {\n\tcalls, fail = 0, false\n\tv, b, err := compute()\n\tif err != nil || v != 42 || !b {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"ok\")\n\t}\n\tcalls, fail = 0, true\n\tv, b, err = compute()\n\tif err == nil || err.Error() != \"boom\" {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"err\")\n\t}\n\tif calls != 1 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"once\")\n\t}\n}\n",
			wantOffer: true,
		},
		{
			name:      "return_nested_single",
			misplaced: "func compute() error {\n\treturn g(step()?)\n}\n",
			hand:      "func compute() error {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn _e\n\t}\n\treturn g(_v)\n}\n",
			mainSrc:   "func main() {\n\tcalls, fail = 0, false\n\tif err := compute(); err != nil || got != 42 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"ok\")\n\t}\n\tcalls, fail = 0, true\n\tif err := compute(); err == nil || err.Error() != \"boom\" {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"err\")\n\t}\n\tif calls != 1 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"once\")\n\t}\n}\n",
			wantOffer: true,
		},
		{
			name:      "closure",
			misplaced: "func compute() (int, error) {\n\tf := func() (int, error) {\n\t\tv := step()? + 1\n\t\treturn v, nil\n\t}\n\treturn f()\n}\n",
			hand:      "func compute() (int, error) {\n\tf := func() (int, error) {\n\t\t_v, _e := step()\n\t\tif _e != nil {\n\t\t\treturn 0, _e\n\t\t}\n\t\tv := _v + 1\n\t\treturn v, nil\n\t}\n\treturn f()\n}\n",
			wantVal:   43,
			wantOffer: true,
		},
		{
			name:      "generic",
			misplaced: "func compute[T any](x T) (int, error) {\n\tv := step()? + 1\n\t_ = x\n\treturn v, nil\n}\n",
			hand:      "func compute[T any](x T) (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\tv := _v + 1\n\t_ = x\n\treturn v, nil\n}\n",
			mainSrc:   "func main() {\n\tcalls, fail = 0, false\n\tv, err := compute(0)\n\tif err != nil || v != 43 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"ok\")\n\t}\n\tcalls, fail = 0, true\n\tv, err = compute(0)\n\tif err == nil || err.Error() != \"boom\" {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"err\")\n\t}\n\tif calls != 1 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"once\")\n\t}\n}\n",
			wantOffer: true,
		},
		{
			name:      "for_init",
			misplaced: "func compute() (int, error) {\n\tfor i := gInt(step()?); i < 100; i++ {\n\t\treturn i, nil\n\t}\n\treturn 0, nil\n}\n",
			wantOffer: false,
		},
		{
			name:      "return_mismatch",
			misplaced: "func compute() error {\n\treturn step()?\n}\n",
			wantOffer: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mainSrc := cmp.Or(tc.mainSrc, fmt.Sprintf(stdMain, tc.wantVal))
			src := "package main\n\n" + stepHelpers + tc.decls + "\n" + tc.misplaced + mainSrc
			diags, exitCode := compileLgoJSON(t, src)
			if exitCode == 0 {
				t.Fatalf("expected compile failure on misplaced '?', got exit code 0")
			}
			d := misplacedTryDiag(t, diags)
			if !tc.wantOffer {
				if len(d.AutofixCandidates) != 0 {
					t.Fatalf("expected no autofix candidate for %s, got %d: %+v", tc.name, len(d.AutofixCandidates), d.AutofixCandidates)
				}
				return
			}
			if len(d.AutofixCandidates) != 1 {
				t.Fatalf("expected exactly one autofix candidate for %s, got %d", tc.name, len(d.AutofixCandidates))
			}
			fix := d.AutofixCandidates[0]
			if strings.Contains(fix.Replacement, "_logos_") || strings.Contains(fix.Replacement, "Logos_") {
				t.Fatalf("candidate replacement uses a reserved prefix: %q", fix.Replacement)
			}
			fixedSrc := applyFix(t, src, d.Span, fix)
			got := runLgoSourceOut(t, fixedSrc)
			handSrc := "package main\n\n" + stepHelpers + tc.decls + "\n" + tc.hand + mainSrc
			want := runLgoSourceOut(t, handSrc)
			if got != want {
				t.Fatalf("applied candidate output = %q, hand-written = %q\nfixed source:\n%s", got, want, fixedSrc)
			}
			const wantOut = "ok\nerr\nonce\n"
			if got != wantOut {
				t.Fatalf("applied candidate output = %q, want %q", got, wantOut)
			}
		})
	}
}
func misplacedTryDiags(t *testing.T, diags []diagRecord) []diagRecord {
	t.Helper()
	var out []diagRecord
	for _, d := range diags {
		if d.RuleID == misplacedTryRuleID {
			out = append(out, d)
		}
	}
	if len(out) == 0 {
		t.Fatalf("expected %s diagnostics, got none: %+v", misplacedTryRuleID, diags)
	}
	return out
}

func TestMisplacedTrySplitMultiple(t *testing.T) {
	const helpers = `type myErr string

func (e myErr) Error() string { return string(e) }

var (
	calls int
	fail bool
	flag bool
	gotAB int
)

func step() (int, error) {
	calls++
	if fail {
		return 0, myErr("boom")
	}
	return 42, nil
}

func okStep() (bool, error) {
	calls++
	if fail {
		return false, myErr("boom")
	}
	return true, nil
}

func add2(a, b int) error {
	gotAB = a + b
	return nil
}

func f1(x int) (int, error) {
	if fail {
		return 0, myErr("boom")
	}
	return x + 1, nil
}

func g1(x int) (int, error) {
	if fail {
		return 0, myErr("boom")
	}
	return x + 1, nil
}
`
	refused := []struct {
		name      string
		compute   string
		wantDiags int
		wantCand  int
		wantNote  bool
	}{
		{
			name:      "two_tries_binary",
			compute:   "func compute() (int, error) {\n\tv := step()? + step()?\n\treturn v, nil\n}\n",
			wantDiags: 2,
			wantCand:  1,
		},
		{
			name:      "three_tries_binary",
			compute:   "func compute() (int, error) {\n\tv := step()? + step()? + step()?\n\treturn v, nil\n}\n",
			wantDiags: 3,
			wantCand:  1,
		},
		{
			name:      "two_tries_call_args",
			compute:   "func compute() (int, error) {\n\terr := add2(step()?, step()?)\n\treturn gotAB, err\n}\n",
			wantDiags: 2,
			wantCand:  1,
		},
		{
			name:      "three_tries_composite",
			compute:   "func compute() (int, error) {\n\txs := []int{step()?, step()?, step()?}\n\treturn xs[0] + xs[1] + xs[2], nil\n}\n",
			wantDiags: 3,
			wantCand:  1,
		},
		{
			name:      "two_tries_if_cond",
			compute:   "func compute() (int, error) {\n\tif step()? == step()? {\n\t\treturn 1, nil\n\t}\n\treturn 0, nil\n}\n",
			wantDiags: 2,
			wantCand:  1,
		},
		{
			name:      "three_tries_return",
			compute:   "func compute() (int, int, int, error) {\n\treturn step()?, step()?, step()?\n}\n",
			wantDiags: 3,
			wantCand:  1,
		},
		{
			name:      "nested_try",
			compute:   "func compute() (int, error) {\n\tv := f1(g1(step()?)?)\n\treturn v, nil\n}\n",
			wantDiags: 2,
			wantCand:  0,
		},
		{
			name:      "and_both",
			compute:   "func compute() (bool, error) {\n\tv := okStep()? && okStep()?\n\treturn v, nil\n}\n",
			wantDiags: 2,
			wantCand:  0,
			wantNote:  true,
		},
		{
			name:      "or_both",
			compute:   "func compute() (bool, error) {\n\tv := okStep()? || okStep()?\n\treturn v, nil\n}\n",
			wantDiags: 2,
			wantCand:  0,
			wantNote:  true,
		},
		{
			name:      "coalesce_right",
			compute:   "func compute() (int, error) {\n\tvar opt int?\n\tv := opt ?? step()?\n\treturn v, nil\n}\n",
			wantDiags: 1,
			wantCand:  0,
			wantNote:  true,
		},
		{
			name:      "coalesce_right_beside_left_try",
			compute:   "func compute() (int, error) {\n\tvar opt int?\n\tv := step()? + (opt ?? step()?)\n\treturn v, nil\n}\n",
			wantDiags: 2,
			wantCand:  0,
			wantNote:  true,
		},
	}
	for _, tc := range refused {
		t.Run(tc.name, func(t *testing.T) {
			src := "package main\n\n" + helpers + tc.compute
			diags, exitCode := compileLgoJSON(t, src)
			if exitCode == 0 {
				t.Fatalf("expected compile failure on misplaced '?', got exit code 0")
			}
			got := misplacedTryDiags(t, diags)
			if len(got) != tc.wantDiags {
				t.Fatalf("got %d %s diagnostics, want %d: %+v", len(got), misplacedTryRuleID, tc.wantDiags, got)
			}
			total := 0
			for i := range got {
				total += len(got[i].AutofixCandidates)
				if got[i].Span.ByteLen == 0 {
					t.Fatalf("expected non-empty span for %s diag %d, got %+v", tc.name, i, got[i].Span)
				}
				if got[i].Provenance.OriginExpr == "" {
					t.Fatalf("expected provenance for %s diag %d, got %+v", tc.name, i, got[i].Provenance)
				}
			}
			if total != tc.wantCand {
				t.Fatalf("expected %d autofix candidates for %s, got %d: %+v", tc.wantCand, tc.name, total, got)
			}
			if tc.wantNote {
				for i := range got {
					if got[i].Note == "" {
						t.Fatalf("expected refusal note for %s diag %d", tc.name, i)
					}
				}
			}
		})
	}
	offered := []struct {
		name      string
		decls     string
		misplaced string
		hand      string
		mainSrc   string
	}{
		{
			name:      "and_left",
			misplaced: "func compute() (bool, error) {\n\tv := okStep()? && flag\n\treturn v, nil\n}\n",
			hand:      "func compute() (bool, error) {\n\t_v, _e := okStep()\n\tif _e != nil {\n\t\treturn false, _e\n\t}\n\tv := _v && flag\n\treturn v, nil\n}\n",
			mainSrc:   "func main() {\n\tcalls, fail, flag = 0, false, true\n\tv, err := compute()\n\tif err != nil || !v {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"ok\")\n\t}\n\tcalls, fail, flag = 0, true, true\n\tv, err = compute()\n\tif err == nil || err.Error() != \"boom\" {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"err\")\n\t}\n\tif calls != 1 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"once\")\n\t}\n}\n",
		},
		{
			name:      "or_left",
			misplaced: "func compute() (bool, error) {\n\tv := okStep()? || flag\n\treturn v, nil\n}\n",
			hand:      "func compute() (bool, error) {\n\t_v, _e := okStep()\n\tif _e != nil {\n\t\treturn false, _e\n\t}\n\tv := _v || flag\n\treturn v, nil\n}\n",
			mainSrc:   "func main() {\n\tcalls, fail, flag = 0, false, false\n\tv, err := compute()\n\tif err != nil || !v {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"ok\")\n\t}\n\tcalls, fail, flag = 0, true, false\n\tv, err = compute()\n\tif err == nil || err.Error() != \"boom\" {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"err\")\n\t}\n\tif calls != 1 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"once\")\n\t}\n}\n",
		},
		{
			name:      "temp_collision",
			misplaced: "func compute() (int, error) {\n\t_step := 7\n\tv := step()? + _step\n\treturn v, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_step := 7\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\tv := _v + _step\n\treturn v, nil\n}\n",
			mainSrc:   "func main() {\n\tcalls, fail = 0, false\n\tv, err := compute()\n\tif err != nil || v != 49 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"ok\")\n\t}\n\tcalls, fail = 0, true\n\tv, err = compute()\n\tif err == nil || err.Error() != \"boom\" {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"err\")\n\t}\n\tif calls != 1 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"once\")\n\t}\n}\n",
		},
		{
			name:      "coalesce_left",
			decls:     "func optStep() (int?, error) {\n\tcalls++\n\tif fail {\n\t\treturn nil, myErr(\"boom\")\n\t}\n\treturn 7, nil\n}\n\n",
			misplaced: "func compute() (int, error) {\n\tv := optStep()? ?? 0\n\treturn v, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := optStep()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\tv := _v ?? 0\n\treturn v, nil\n}\n",
			mainSrc:   "func main() {\n\tcalls, fail = 0, false\n\tv, err := compute()\n\tif err != nil || v != 7 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"ok\")\n\t}\n\tcalls, fail = 0, true\n\tv, err = compute()\n\tif err == nil || err.Error() != \"boom\" {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"err\")\n\t}\n\tif calls != 1 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"once\")\n\t}\n}\n",
		},
	}
	for _, tc := range offered {
		t.Run(tc.name, func(t *testing.T) {
			src := "package main\n\n" + helpers + tc.decls + tc.misplaced + tc.mainSrc
			diags, exitCode := compileLgoJSON(t, src)
			if exitCode == 0 {
				t.Fatalf("expected compile failure on misplaced '?', got exit code 0")
			}
			d := misplacedTryDiag(t, diags)
			if len(d.AutofixCandidates) != 1 {
				t.Fatalf("expected exactly one autofix candidate for %s, got %d", tc.name, len(d.AutofixCandidates))
			}
			fixedSrc := applyFix(t, src, d.Span, d.AutofixCandidates[0])
			got := runLgoSourceOut(t, fixedSrc)
			handSrc := "package main\n\n" + helpers + tc.decls + tc.hand + tc.mainSrc
			want := runLgoSourceOut(t, handSrc)
			if got != want {
				t.Fatalf("applied candidate output = %q, hand-written = %q\nfixed source:\n%s", got, want, fixedSrc)
			}
			const wantOut = "ok\nerr\nonce\n"
			if got != wantOut {
				t.Fatalf("applied candidate output = %q, want %q", got, wantOut)
			}
		})
	}
}

func TestMisplacedTrySplitMultiFix(t *testing.T) {
	const helpers = `type myErr string

func (e myErr) Error() string { return string(e) }

var (
	calls int
	fail bool
	gotA int
	gotB int
	gotAB int
	gotS string
)

func step() (int, error) {
	calls++
	if fail {
		return 0, myErr("boom")
	}
	return 42, nil
}

func name() (string, error) {
	calls++
	if fail {
		return "", myErr("boom")
	}
	return "hi", nil
}

func add3(a, b, c int) error {
	gotA, gotB = a, b
	gotAB = a + b + c
	return nil
}

func outer(x int) (int, error) {
	if fail {
		return 0, myErr("boom")
	}
	return x + 1, nil
}

func inner(x int) (int, error) {
	if fail {
		return 0, myErr("boom")
	}
	return x + 10, nil
}

func next() (int, error) {
	calls++
	if fail {
		return 0, myErr("boom")
	}
	return calls, nil
}

func sub(a, b int) int {
	return a*10 + b
}
`
	const stdMain = `func main() {
	calls, fail = 0, false
	v, err := compute()
	if err != nil || v != %d || calls != %d {
		println("bad")
	} else {
		println("ok")
	}
	calls, fail = 0, true
	v, err = compute()
	if err == nil || err.Error() != "boom" || calls != 1 {
		println("bad")
	} else {
		println("err")
	}
	println("once")
}
`
	tests := []struct {
		name      string
		misplaced string
		hand      string
		mainSrc   string
		wantVal   int
		calls     int
	}{
		{
			name:      "return_multi",
			misplaced: "func compute() (int, int, error) {\n\treturn step()?, step()?\n}\n",
			hand:      "func compute() (int, int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, 0, _e\n\t}\n\t_step, _e1 := step()\n\tif _e1 != nil {\n\t\treturn 0, 0, _e1\n\t}\n\treturn _v, _step, nil\n}\n",
			mainSrc:   "func main() {\n\tcalls, fail = 0, false\n\ta, b, err := compute()\n\tif err != nil || a != 42 || b != 42 || calls != 2 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"ok\")\n\t}\n\tcalls, fail = 0, true\n\ta, b, err = compute()\n\tif err == nil || err.Error() != \"boom\" || calls != 1 {\n\t\tprintln(\"bad\")\n\t} else {\n\t\tprintln(\"err\")\n\t}\n\tprintln(\"once\")\n}\n",
			wantVal:   42,
			calls:     2,
		},
		{
			name:      "two_call_args",
			misplaced: "func compute() (int, error) {\n\terr := add3(step()?, step()?, 0)\n\treturn gotAB, err\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\t_step, _e1 := step()\n\tif _e1 != nil {\n\t\treturn 0, _e1\n\t}\n\terr := add3(_v, _step, 0)\n\treturn gotAB, err\n}\n",
			wantVal:   84,
			calls:     2,
		},
		{
			name:      "nested_try_wrap",
			misplaced: "func compute() (int, error) {\n\tv := outer(inner(step()?)?)?\n\treturn v, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\t_inner, _e1 := inner(_v)\n\tif _e1 != nil {\n\t\treturn 0, _e1\n\t}\n\t_out, _e2 := outer(_inner)\n\tif _e2 != nil {\n\t\treturn 0, _e2\n\t}\n\treturn _out, nil\n}\n",
			wantVal:   53,
			calls:     1,
		},
		{
			name:      "binary_sides",
			misplaced: "func compute() (int, error) {\n\tv := step()? + step()?\n\treturn v, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\t_step, _e1 := step()\n\tif _e1 != nil {\n\t\treturn 0, _e1\n\t}\n\tv := _v + _step\n\treturn v, nil\n}\n",
			wantVal:   84,
			calls:     2,
		},
		{
			name:      "three_chain",
			misplaced: "func compute() (int, error) {\n\tv := step()? + step()? + step()?\n\treturn v, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\t_step, _e1 := step()\n\tif _e1 != nil {\n\t\treturn 0, _e1\n\t}\n\t_step1, _e2 := step()\n\tif _e2 != nil {\n\t\treturn 0, _e2\n\t}\n\tv := _v + _step + _step1\n\treturn v, nil\n}\n",
			wantVal:   126,
			calls:     3,
		},
		{
			name:      "and_left",
			misplaced: "func compute() (int, error) {\n\tb := name()? == \"hi\" && true\n\tif b {\n\t\treturn 1, nil\n\t}\n\treturn 0, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := name()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\tb := _v == \"hi\" && true\n\tif b {\n\t\treturn 1, nil\n\t}\n\treturn 0, nil\n}\n",
			wantVal:   1,
			calls:     1,
		},
		{
			name:      "if_cond_both",
			misplaced: "func compute() (int, error) {\n\tif step()? == step()? {\n\t\treturn 1, nil\n\t}\n\treturn 0, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := step()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\t_step, _e1 := step()\n\tif _e1 != nil {\n\t\treturn 0, _e1\n\t}\n\tif _v == _step {\n\t\treturn 1, nil\n\t}\n\treturn 0, nil\n}\n",
			wantVal:   1,
			calls:     2,
		},
		{
			name:      "args_in_order",
			misplaced: "func compute() (int, error) {\n\tv := sub(next()?, next()?)\n\treturn v, nil\n}\n",
			hand:      "func compute() (int, error) {\n\t_v, _e := next()\n\tif _e != nil {\n\t\treturn 0, _e\n\t}\n\t_next, _e1 := next()\n\tif _e1 != nil {\n\t\treturn 0, _e1\n\t}\n\tv := sub(_v, _next)\n\treturn v, nil\n}\n",
			wantVal:   12,
			calls:     2,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mainSrc := cmp.Or(tc.mainSrc, fmt.Sprintf(stdMain, tc.wantVal, tc.calls))
			src := "package main\n\n" + helpers + tc.misplaced + mainSrc
			diags, exitCode := compileLgoJSON(t, src)
			if exitCode == 0 {
				t.Fatalf("expected compile failure on misplaced '?', got exit code 0")
			}
			var fix *diagFix
			var span diagSpan
			for i := range diags {
				if diags[i].RuleID != misplacedTryRuleID {
					continue
				}
				if diags[i].Span.ByteLen == 0 {
					t.Fatalf("expected non-empty span on diag %d, got %+v", i, diags[i].Span)
				}
				switch n := len(diags[i].AutofixCandidates); {
				case n == 1 && fix == nil:
					fix = &diags[i].AutofixCandidates[0]
					span = diags[i].Span
				case n > 0:
					t.Fatalf("expected at most one candidate across diagnostics, diag %d adds more", i)
				}
			}
			if fix == nil {
				t.Fatalf("expected one autofix candidate for %s, got none", tc.name)
			}
			if strings.Contains(fix.Replacement, "_logos_") || strings.Contains(fix.Replacement, "Logos_") {
				t.Fatalf("candidate replacement uses a reserved prefix: %q", fix.Replacement)
			}
			fixedSrc := applyFix(t, src, span, *fix)
			fixedDiags, _ := compileLgoJSON(t, fixedSrc)
			for _, fd := range fixedDiags {
				if fd.RuleID == misplacedTryRuleID {
					t.Fatalf("fixed source still reports misplaced_try: %+v\nfixed source:\n%s", fd, fixedSrc)
				}
			}
			got := runLgoSourceOut(t, fixedSrc)
			handSrc := "package main\n\n" + helpers + tc.hand + mainSrc
			want := runLgoSourceOut(t, handSrc)
			if got != want {
				t.Fatalf("applied candidate output = %q, hand-written = %q\nfixed source:\n%s", got, want, fixedSrc)
			}
			const wantOut = "ok\nerr\nonce\n"
			if got != wantOut {
				t.Fatalf("applied candidate output = %q, want %q", got, wantOut)
			}
		})
	}
}

func TestMisplacedTryShortCircuitRefused(t *testing.T) {
	const helpers = `type myErr string

func (e myErr) Error() string { return string(e) }

var (
	calls int
	fail bool
	flag bool
)

func okStep() (bool, error) {
	calls++
	if fail {
		return false, myErr("boom")
	}
	return true, nil
}
`
	src := "package main\n\n" + helpers + "func compute() (bool, error) {\n\tv := flag && okStep()?\n\treturn v, nil\n}\n"
	diags, exitCode := compileLgoJSON(t, src)
	if exitCode == 0 {
		t.Fatalf("expected compile failure on misplaced '?', got exit code 0")
	}
	d := misplacedTryDiag(t, diags)
	if len(d.AutofixCandidates) != 0 {
		t.Fatalf("expected no candidate for short-circuit right operand, got %+v", d.AutofixCandidates)
	}
}
