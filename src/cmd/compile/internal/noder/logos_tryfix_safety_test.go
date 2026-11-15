// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"testing"
)

func TestMisplacedTryAutofixShortCircuitAnd(t *testing.T) {
	const src = `package main

type myError struct{}

func (myError) Error() string { return "step error" }

var (
	stepCalls int
	stepFail  bool
)

func step() (bool, error) {
	stepCalls++
	if stepFail {
		return false, myError{}
	}
	return true, nil
}

func computeAnd(cond bool) (bool, error) {
	v := cond && step()?
	return v, nil
}

func main() {
	// && with cond=false: step must NOT be called even if it would fail.
	stepCalls = 0
	stepFail = true
	val, err := computeAnd(false)
	if err != nil {
		panic("computeAnd(false) must not propagate error from unreached step()")
	}
	if val != false {
		panic("computeAnd(false) must be false")
	}
	if stepCalls != 0 {
		panic("computeAnd(false) must short-circuit and not execute step()")
	}

	// && with cond=true, stepFail=false: step called exactly once.
	stepCalls = 0
	stepFail = false
	val, err = computeAnd(true)
	if err != nil || val != true || stepCalls != 1 {
		panic("computeAnd(true) failed success assertions")
	}
}
`

	diags, exitCode := compileLgoJSON(t, src)
	if exitCode == 0 {
		t.Fatalf("expected compile failure on misplaced '?' on '&&' RHS, got exit code 0")
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
		t.Logf("conservative refusal: safely offered no autofix candidate on '&&' RHS")
		return
	}

	for i, fix := range targetDiag.AutofixCandidates {
		fixedSrc := applyFix(t, src, targetDiag.Span, fix)
		runLgoSource(t, fixedSrc)
		t.Logf("candidate %d preserved '&&' short-circuit evaluation semantics", i)
	}
}

func TestMisplacedTryAutofixShortCircuitOr(t *testing.T) {
	const src = `package main

type myError struct{}

func (myError) Error() string { return "step error" }

var (
	stepCalls int
	stepFail  bool
)

func step() (bool, error) {
	stepCalls++
	if stepFail {
		return false, myError{}
	}
	return true, nil
}

func computeOr(cond bool) (bool, error) {
	v := cond || step()?
	return v, nil
}

func main() {
	// || with cond=true: step must NOT be called even if it would fail.
	stepCalls = 0
	stepFail = true
	val, err := computeOr(true)
	if err != nil {
		panic("computeOr(true) must not propagate error from unreached step()")
	}
	if val != true {
		panic("computeOr(true) must be true")
	}
	if stepCalls != 0 {
		panic("computeOr(true) must short-circuit and not execute step()")
	}

	// || with cond=false, stepFail=false: step called exactly once.
	stepCalls = 0
	stepFail = false
	val, err = computeOr(false)
	if err != nil || val != true || stepCalls != 1 {
		panic("computeOr(false) failed success assertions")
	}
}
`

	diags, exitCode := compileLgoJSON(t, src)
	if exitCode == 0 {
		t.Fatalf("expected compile failure on misplaced '?' on '||' RHS, got exit code 0")
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
		t.Logf("conservative refusal: safely offered no autofix candidate on '||' RHS")
		return
	}

	for i, fix := range targetDiag.AutofixCandidates {
		fixedSrc := applyFix(t, src, targetDiag.Span, fix)
		runLgoSource(t, fixedSrc)
		t.Logf("candidate %d preserved '||' short-circuit evaluation semantics", i)
	}
}

func TestMisplacedTryAutofixShortCircuitCoalesce(t *testing.T) {
	const src = `package main

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

func computeCoalesce(opt int?) (int, error) {
	v := opt ?? step()?
	return v, nil
}

func main() {
	// opt is some (42): step must NOT be called even if it would fail.
	stepCalls = 0
	stepFail = true
	var some int? = 42
	val, err := computeCoalesce(some)
	if err != nil {
		panic("computeCoalesce(some) must not propagate error from unreached fallback")
	}
	if val != 42 {
		panic("computeCoalesce(some) must yield 42")
	}
	if stepCalls != 0 {
		panic("computeCoalesce(some) must short-circuit and not execute fallback step()")
	}

	// opt is none (nil), stepFail=false: fallback step called exactly once.
	stepCalls = 0
	stepFail = false
	var none int? = nil
	val, err = computeCoalesce(none)
	if err != nil || val != 42 || stepCalls != 1 {
		panic("computeCoalesce(none) failed success assertions")
	}
}
`

	diags, exitCode := compileLgoJSON(t, src)
	if exitCode == 0 {
		t.Fatalf("expected compile failure on misplaced '?' on '??' RHS, got exit code 0")
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
		t.Logf("conservative refusal: safely offered no autofix candidate on '??' RHS")
		return
	}

	for i, fix := range targetDiag.AutofixCandidates {
		fixedSrc := applyFix(t, src, targetDiag.Span, fix)
		runLgoSource(t, fixedSrc)
		t.Logf("candidate %d preserved '??' short-circuit fallback semantics", i)
	}
}

func TestMisplacedTryAutofixSameLinePrefix(t *testing.T) {
	const caseSrc = `package main

type myError struct{}

func (myError) Error() string { return "step error" }

func step() (int, error) {
	return 42, nil
}

func computeCase() (int, error) {
	switch 1 {
	case 1: v := step()? + 1; return v, nil
	default: return 0, nil
	}
}

func main() {
	val, err := computeCase()
	if err != nil || val != 43 {
		panic("unexpected result from computeCase")
	}
}
`

	diags, exitCode := compileLgoJSON(t, caseSrc)
	if exitCode == 0 {
		t.Fatalf("expected compile failure on same-line 'case ...: stmt', got exit code 0")
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
		t.Logf("conservative refusal: safely offered no autofix candidate for same-line case clause")
		return
	}

	for i, fix := range targetDiag.AutofixCandidates {
		fixedSrc := applyFix(t, caseSrc, targetDiag.Span, fix)
		runLgoSource(t, fixedSrc)
		t.Logf("candidate %d compiled and executed without prefix duplication", i)
	}
}

func TestMisplacedTryAutofixSameLineBrace(t *testing.T) {
	const braceSrc = `package main

type myError struct{}

func (myError) Error() string { return "step error" }

func step() (int, error) {
	return 42, nil
}

func computeBrace() (int, error) {
	{ v := step()? + 1; return v, nil }
}

func main() {
	val, err := computeBrace()
	if err != nil || val != 43 {
		panic("unexpected result from computeBrace")
	}
}
`

	diags, exitCode := compileLgoJSON(t, braceSrc)
	if exitCode == 0 {
		t.Fatalf("expected compile failure on same-line '{ stmt }', got exit code 0")
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
		t.Logf("conservative refusal: safely offered no autofix candidate for inline block")
		return
	}

	for i, fix := range targetDiag.AutofixCandidates {
		fixedSrc := applyFix(t, braceSrc, targetDiag.Span, fix)
		runLgoSource(t, fixedSrc)
		t.Logf("candidate %d compiled and executed without brace duplication", i)
	}
}

func TestMisplacedTryAutofixCommHeader(t *testing.T) {
	const headerSrc = `package main

type myError struct{}

func (myError) Error() string { return "step error" }

func getChan() (chan int, error) {
	return make(chan int, 1), nil
}

func selectHeader() (int, error) {
	select {
	case v := <-getChan()?:
		return v, nil
	default:
		return 0, nil
	}
}

func main() {
	val, err := selectHeader()
	if err != nil {
		panic(err)
	}
	_ = val
}
`

	diags, exitCode := compileLgoJSON(t, headerSrc)
	if exitCode == 0 {
		t.Fatalf("expected compile failure on select-header misplaced '?', got exit code 0")
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

	if len(targetDiag.AutofixCandidates) != 0 {
		t.Fatalf("must refuse autofix for select-header Comm: got %d candidate(s) with replacement %q, want 0 (multiline split before colon has no valid statement context)", len(targetDiag.AutofixCandidates), targetDiag.AutofixCandidates[0].Replacement)
	}

	const bodySrc = `package main

type myError struct{}

func (myError) Error() string { return "step error" }

var (
	trace     []string
	stepCalls int
	stepFail  bool
)

func step() (int, error) {
	stepCalls++
	trace = append(trace, "step")
	if stepFail {
		return 0, myError{}
	}
	return 42, nil
}

func transform(a, b int) int {
	trace = append(trace, "transform")
	return a + b
}

func computeBody() (int, error) {
	ch := make(chan int, 1)
	ch <- 10
	select {
	case v := <-ch:
		w := transform(v, step()?)
		return w, nil
	default:
		return -1, nil
	}
}

func main() {
	trace = nil
	stepCalls = 0
	stepFail = false
	val, err := computeBody()
	if err != nil {
		panic("unexpected error on success path")
	}
	if val != 52 {
		panic("expected value 52 on success path")
	}
	if stepCalls != 1 {
		panic("expected exactly-once step() evaluation on success path")
	}
	if len(trace) != 2 || trace[0] != "step" || trace[1] != "transform" {
		panic("evaluation order violated on success path")
	}

	trace = nil
	stepCalls = 0
	stepFail = true
	_, err = computeBody()
	if err == nil {
		panic("expected error on error path, got nil")
	}
	if err.Error() != "step error" {
		panic("unexpected error message on error path")
	}
	if stepCalls != 1 {
		panic("expected exactly-once step() evaluation on error path")
	}
	if len(trace) != 1 || trace[0] != "step" {
		panic("transform must not execute after step() failure on error path")
	}
}
`

	bodyDiags, bodyExit := compileLgoJSON(t, bodySrc)
	if bodyExit == 0 {
		t.Fatalf("expected compile failure on clause-body misplaced '?', got exit code 0")
	}

	var bodyTarget *diagRecord
	for i := range bodyDiags {
		if bodyDiags[i].RuleID == misplacedTryRuleID {
			bodyTarget = &bodyDiags[i]
			break
		}
	}
	if bodyTarget == nil {
		t.Fatalf("expected diagnostic with rule_id %q for clause-body control, got: %+v", misplacedTryRuleID, bodyDiags)
	}
	if bodyTarget.Severity != severityForbid {
		t.Errorf("expected severity %q, got %q", severityForbid, bodyTarget.Severity)
	}
	if len(bodyTarget.AutofixCandidates) == 0 {
		t.Fatalf("expected autofix candidate for clause-body nested try, got 0 candidates")
	}

	for i, fix := range bodyTarget.AutofixCandidates {
		fixedSrc := applyFix(t, bodySrc, bodyTarget.Span, fix)
		runLgoSource(t, fixedSrc)
		t.Logf("candidate %d preserved clause-body exactly-once result and error propagation", i)
	}
}
