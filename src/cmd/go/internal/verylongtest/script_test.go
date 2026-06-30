// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package verylongtest

import (
	"cmd/internal/script"
	"cmd/internal/script/scripttest"
	"internal/testenv"
	"os/exec"
	"runtime"
	"testing"
)

func TestScript(t *testing.T) {
	if testing.Short() {
		// Don't bother setting up the script engine. None of these are short tests.
		t.Skip()
	}
	testenv.MustHaveGoBuild(t)
	testenv.SkipIfShortAndSlow(t)

	engine, env := scripttest.NewEngine(t, nil)
	env = append(env, "GOROOT="+runtime.GOROOT())
	engine.Conds["net"] = script.PrefixCondition("can connect to external network host <suffix>", hasNet)
	engine.Conds["git"] = script.OnceCondition("the 'git' executable exists and provides the standard CLI", hasWorkingGit)
	scripttest.RunTests(t, t.Context(), engine, env, "testdata/script/*.txt")
}

func hasNet(*script.State, string) (bool, error) {
	return testenv.HasExternalNetwork(), nil
}

func hasWorkingGit() (bool, error) {
	if runtime.GOOS == "plan9" {
		// The Git command is usually not the real Git on Plan 9.
		// See https://golang.org/issues/29640.
		return false, nil
	}
	_, err := exec.LookPath("git")
	return err == nil, nil
}
