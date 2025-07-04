// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package scripttest adapts the script engine for use in tests.
package scripttest

import (
	"bytes"
	"cmd/internal/script"
	"context"
	"fmt"
	"internal/testenv"
	"internal/txtar"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// ToolReplacement records the name of a tool to replace
// within a given GOROOT for script testing purposes.
type ToolReplacement struct {
	ToolName        string // e.g. compile, link, addr2line, etc
	ReplacementPath string // path to replacement tool exe
	EnvVar          string // env var setting (e.g. "FOO=BAR")
}

// RunToolScriptTest kicks off a set of script tests runs for
// a tool of some sort (compiler, linker, etc). The expectation
// is that we'll be called from the top level cmd/X dir for tool X,
// and that instead of executing the install tool X we'll use the
// test binary instead.
func RunToolScriptTest(t *testing.T, repls []ToolReplacement, scriptsdir string, fixReadme bool) {
	// Nearly all script tests involve doing builds, so don't
	// bother here if we don't have "go build".
	testenv.MustHaveGoBuild(t)

	// Skip this path on plan9, which doesn't support symbolic
	// links (we would have to copy too much).
	if runtime.GOOS == "plan9" {
		t.Skipf("no symlinks on plan9")
	}

	// Locate our Go tool.
	gotool, err := testenv.GoTool()
	if err != nil {
		t.Fatalf("locating go tool: %v", err)
	}

	goEnv := func(name string) string {
		out, err := exec.Command(gotool, "env", name).CombinedOutput()
		if err != nil {
			t.Fatalf("go env %s: %v\n%s", name, err, out)
		}
		return strings.TrimSpace(string(out))
	}

	// Construct an initial set of commands + conditions to make available
	// to the script tests.
	cmds := DefaultCmds()
	conds := DefaultConds()

	addcmd := func(name string, cmd script.Cmd) {
		if _, ok := cmds[name]; ok {
			panic(fmt.Sprintf("command %q is already registered", name))
		}
		cmds[name] = cmd
	}

	prependToPath := func(env []string, dir string) {
		found := false
		for k := range env {
			ev := env[k]
			oldpath, cut := strings.CutPrefix(ev, "PATH=")
			if !cut {
				continue
			}
			env[k] = "PATH=" + dir + string(filepath.ListSeparator) + oldpath
			found = true
			break
		}
		if !found {
			t.Fatalf("could not update PATH")
		}
	}

	setenv := func(env []string, varname, val string) []string {
		pref := varname + "="
		found := false
		for k := range env {
			if !strings.HasPrefix(env[k], pref) {
				continue
			}
			env[k] = pref + val
			found = true
			break
		}
		if !found {
			env = append(env, varname+"="+val)
		}
		return env
	}

	interrupt := func(cmd *exec.Cmd) error {
		return cmd.Process.Signal(os.Interrupt)
	}
	gracePeriod := 60 * time.Second // arbitrary

	// Set up an alternate go root for running script tests, since it
	// is possible that we might want to replace one of the installed
	// tools with a unit test executable.
	goroot := goEnv("GOROOT")
	tmpdir := t.TempDir()
	tgr := SetupTestGoRoot(t, tmpdir, goroot)

	// Replace tools if appropriate
	for _, repl := range repls {
		ReplaceGoToolInTestGoRoot(t, tgr, repl.ToolName, repl.ReplacementPath)
	}

	// Add in commands for "go" and "cc".
	testgo := filepath.Join(tgr, "bin", "go")
	gocmd := script.Program(testgo, interrupt, gracePeriod)
	addcmd("go", gocmd)
	cmdExec := cmds["exec"]
	addcmd("cc", scriptCC(cmdExec, goEnv("CC")))

	// Add various helpful conditions related to builds and toolchain use.
	goHostOS, goHostArch := goEnv("GOHOSTOS"), goEnv("GOHOSTARCH")
	AddToolChainScriptConditions(t, conds, goHostOS, goHostArch)

	// Environment setup.
	env := os.Environ()
	prependToPath(env, filepath.Join(tgr, "bin"))
	env = setenv(env, "GOROOT", tgr)
	for _, repl := range repls {
		// consistency check
		chunks := strings.Split(repl.EnvVar, "=")
		if len(chunks) != 2 {
			t.Fatalf("malformed env var setting: %s", repl.EnvVar)
		}
		env = append(env, repl.EnvVar)
	}

	// Manufacture engine...
	engine := &script.Engine{
		Conds: conds,
		Cmds:  cmds,
		Quiet: !testing.Verbose(),
	}

	t.Run("README", func(t *testing.T) {
		checkScriptReadme(t, engine, env, scriptsdir, gotool, fixReadme)
	})

	// ... and kick off tests.
	ctx := context.Background()
	pattern := filepath.Join(scriptsdir, "*.txt")
	RunTests(t, ctx, engine, env, pattern)
}

// RunTests kicks off one or more script-based tests using the
// specified engine, running all test files that match pattern.
// This function adapted from Russ's rsc.io/script/scripttest#Run
// function, which was in turn forked off cmd/go's runner.
func RunTests(t *testing.T, ctx context.Context, engine *script.Engine, env []string, pattern string) {
	gracePeriod := 100 * time.Millisecond
	if deadline, ok := t.Deadline(); ok {
		timeout := time.Until(deadline)

		// If time allows, increase the termination grace period to 5% of the
		// remaining time.
		if gp := timeout / 20; gp > gracePeriod {
			gracePeriod = gp
		}

		// When we run commands that execute subprocesses, we want to
		// reserve two grace periods to clean up. We will send the
		// first termination signal when the context expires, then
		// wait one grace period for the process to produce whatever
		// useful output it can (such as a stack trace). After the
		// first grace period expires, we'll escalate to os.Kill,
		// leaving the second grace period for the test function to
		// record its output before the test process itself
		// terminates.
		timeout -= 2 * gracePeriod

		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		t.Cleanup(cancel)
	}

	files, _ := filepath.Glob(pattern)
	if len(files) == 0 {
		t.Fatal("no testdata")
	}
	for _, file := range files {
		file := file
		name := strings.TrimSuffix(filepath.Base(file), ".txt")
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			workdir := t.TempDir()
			s, err := script.NewState(ctx, workdir, env)
			if err != nil {
				t.Fatal(err)
			}

			// Unpack archive.
			a, err := txtar.ParseFile(file)
			if err != nil {
				t.Fatal(err)
			}
			initScriptDirs(t, s)
			if err := s.ExtractFiles(a); err != nil {
				t.Fatal(err)
			}

			t.Log(time.Now().UTC().Format(time.RFC3339))
			work, _ := s.LookupEnv("WORK")
			t.Logf("$WORK=%s", work)

			// Note: Do not use filepath.Base(file) here:
			// editors that can jump to file:line references in the output
			// will work better seeing the full path relative to the
			// directory containing the command being tested
			// (e.g. where "go test" command is usually run).
			Run(t, engine, s, file, bytes.NewReader(a.Comment))
		})
	}
}

func initScriptDirs(t testing.TB, s *script.State) {
	must := func(err error) {
		if err != nil {
			t.Helper()
			t.Fatal(err)
		}
	}

	work := s.Getwd()
	must(s.Setenv("WORK", work))
	must(os.MkdirAll(filepath.Join(work, "tmp"), 0777))
	must(s.Setenv(tempEnvName(), filepath.Join(work, "tmp")))
}

func tempEnvName() string {
	switch runtime.GOOS {
	case "windows":
		return "TMP"
	case "plan9":
		return "TMPDIR" // actually plan 9 doesn't have one at all but this is fine
	default:
		return "TMPDIR"
	}
}

// scriptCC runs the platform C compiler.
func scriptCC(cmdExec script.Cmd, ccexe string) script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "run the platform C compiler",
			Args:    "args...",
		},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			return cmdExec.Run(s, append([]string{ccexe}, args...)...)
		})
}
