// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// compileLgo runs go tool compile on one loGo/s file x.lgo. A non-empty
// policy is written to a file named by -logos.cfg; an empty one passes no
// policy, as a direct compiler run does.
func compileLgo(t *testing.T, src, policy string, flags ...string) compileResult {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "x.lgo")
	object := filepath.Join(dir, "out.a")
	if err := os.WriteFile(srcPath, []byte(src), 0o666); err != nil {
		t.Fatal(err)
	}
	args := []string{"tool", "compile", "-p", "p", "-o", object}
	if policy != "" {
		policyPath := filepath.Join(dir, "policy.json")
		if err := os.WriteFile(policyPath, []byte(policy), 0o666); err != nil {
			t.Fatal(err)
		}
		args = append(args, "-logos.cfg="+policyPath)
	}
	args = append(append(args, flags...), srcPath)
	out, err := exec.Command(testenv.GoToolPath(t), args...).CombinedOutput()
	res := compileResult{output: string(out)}
	if ee, ok := err.(*exec.ExitError); ok {
		res.exitCode = ee.ExitCode()
	} else if err != nil {
		t.Fatal(err)
	}
	_, statErr := os.Stat(object)
	res.objectOK = statErr == nil
	return res
}

const routedSrc = `package p

type Cfg struct {
	Port int32 = 8080
}

func f() {
	var c Cfg
	_ = c
}
`

const routedAllowedSrc = `package p

type Cfg struct {
	Port int32 = 8080
}

//logos:allow(suspicious::unapplied_struct_default)
func f() {
	var c Cfg
	_ = c
}
`

const (
	structDefaultText = "x.lgo:8:8: warning: [suspicious::unapplied_struct_default] zero declaration of Cfg with field defaults"
	structDefaultDeny = "x.lgo:8:8: [suspicious::unapplied_struct_default] zero declaration of Cfg with field defaults"
)

// TestCheckerWarningsFollowSeverity checks that a warning of types2 takes
// the group severity of the policy: warn prints it and writes the object,
// deny turns it into an error without the warning prefix, and allow is
// silent.
func TestCheckerWarningsFollowSeverity(t *testing.T) {
	res := compileLgo(t, routedSrc, `{"version":1}`)
	if res.exitCode != 0 || !res.objectOK || !strings.Contains(res.output, structDefaultText) {
		t.Errorf("default: exit %d, object %v, output %q", res.exitCode, res.objectOK, res.output)
	}
	res = compileLgo(t, routedSrc, `{"version":1,"groups":{"suspicious":"deny"}}`)
	if res.exitCode != 2 || res.objectOK || strings.Contains(res.output, "warning:") || !strings.Contains(res.output, structDefaultDeny) {
		t.Errorf("deny: exit %d, object %v, output %q", res.exitCode, res.objectOK, res.output)
	}
	res = compileLgo(t, routedSrc, `{"version":1,"groups":{"suspicious":"allow"}}`)
	if res.exitCode != 0 || !res.objectOK || res.output != "" {
		t.Errorf("allow: exit %d, object %v, output %q", res.exitCode, res.objectOK, res.output)
	}
}

// TestCheckerWarningsFollowDirective checks that //logos:allow silences a
// warning of types2 under warn and under deny, and that forbid ignores it
// and reports the directive.
func TestCheckerWarningsFollowDirective(t *testing.T) {
	for _, level := range []string{"warn", "deny"} {
		res := compileLgo(t, routedAllowedSrc, `{"version":1,"groups":{"suspicious":"`+level+`"}}`)
		if res.exitCode != 0 || !res.objectOK || res.output != "" {
			t.Errorf("%s with directive: exit %d, object %v, output %q", level, res.exitCode, res.objectOK, res.output)
		}
	}
	res := compileLgo(t, routedAllowedSrc, `{"version":1,"groups":{"suspicious":"forbid"}}`)
	for _, want := range []string{
		"x.lgo:7:1: [correctness::allow_forbidden_rule]",
		"x.lgo:9:8: [suspicious::unapplied_struct_default]",
	} {
		if !strings.Contains(res.output, want) {
			t.Errorf("forbid with directive: output lacks %q: %q", want, res.output)
		}
	}
	if res.exitCode != 2 || res.objectOK {
		t.Errorf("forbid with directive: exit %d, object %v", res.exitCode, res.objectOK)
	}
}

// TestCheckerWarningsWithoutPolicy checks a compile with no -logos.cfg: the
// warning prints at the default severity, and a directive silences it.
func TestCheckerWarningsWithoutPolicy(t *testing.T) {
	res := compileLgo(t, routedSrc, "")
	if res.exitCode != 0 || !res.objectOK || !strings.Contains(res.output, structDefaultText) {
		t.Errorf("no policy: exit %d, object %v, output %q", res.exitCode, res.objectOK, res.output)
	}
	res = compileLgo(t, routedAllowedSrc, "")
	if res.exitCode != 0 || !res.objectOK || res.output != "" {
		t.Errorf("no policy with directive: exit %d, object %v, output %q", res.exitCode, res.objectOK, res.output)
	}
}

// TestCheckerWarningsBesideTypeError checks that a package that fails to
// type-check still prints its resolved warnings beside the error, and that
// a policy that allows the rule drops them.
func TestCheckerWarningsBesideTypeError(t *testing.T) {
	src := routedSrc + "\nfunc h() { undefined() }\n"
	res := compileLgo(t, src, `{"version":1}`)
	if res.exitCode != 2 || !strings.Contains(res.output, "undefined: undefined") || !strings.Contains(res.output, structDefaultText) {
		t.Errorf("default beside error: exit %d, output %q", res.exitCode, res.output)
	}
	res = compileLgo(t, src, `{"version":1,"groups":{"suspicious":"allow"}}`)
	if res.exitCode != 2 || strings.Contains(res.output, "suspicious::") {
		t.Errorf("allow beside error: exit %d, output %q", res.exitCode, res.output)
	}
	res = compileLgo(t, src, `{"version":1,"groups":{"suspicious":"deny"}}`)
	if res.exitCode != 2 || strings.Contains(res.output, "warning:") || !strings.Contains(res.output, structDefaultDeny) {
		t.Errorf("deny beside error: exit %d, output %q", res.exitCode, res.output)
	}
}

// TestCheckerWarningsJSON checks the JSON record of a routed warning: the
// severity of the policy and the rule ID.
func TestCheckerWarningsJSON(t *testing.T) {
	res := compileLgo(t, routedSrc, `{"version":1}`, "-logos.json")
	if res.exitCode != 0 || !strings.Contains(res.output, `"severity":"warn","rule_id":"suspicious::unapplied_struct_default"`) {
		t.Errorf("JSON warn: exit %d, output %q", res.exitCode, res.output)
	}
	res = compileLgo(t, routedSrc, `{"version":1,"groups":{"suspicious":"deny"}}`, "-logos.json")
	if res.exitCode != 2 || !strings.Contains(res.output, `"severity":"deny","rule_id":"suspicious::unapplied_struct_default"`) {
		t.Errorf("JSON deny: exit %d, output %q", res.exitCode, res.output)
	}
}
