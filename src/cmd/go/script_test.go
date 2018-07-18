// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Script-driven tests.
// See testdata/script/README for an overview.

package main_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"

	"cmd/go/internal/imports"
	"cmd/go/internal/par"
	"cmd/go/internal/txtar"
)

// TestScript runs the tests in testdata/script/*.txt.
func TestScript(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if skipExternal {
		t.Skipf("skipping external tests on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	files, err := filepath.Glob("testdata/script/*.txt")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		file := file
		name := strings.TrimSuffix(filepath.Base(file), ".txt")
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			ts := &testScript{t: t, name: name, file: file}
			ts.setup()
			if !*testWork {
				defer removeAll(ts.workdir)
			}
			ts.run()
		})
	}
}

// A testScript holds execution state for a single test script.
type testScript struct {
	t       *testing.T
	workdir string            // temporary work dir ($WORK)
	log     bytes.Buffer      // test execution log (printed at end of test)
	mark    int               // offset of next log truncation
	cd      string            // current directory during test execution; initially $WORK/gopath/src
	name    string            // short name of test ("foo")
	file    string            // full file name ("testdata/script/foo.txt")
	lineno  int               // line number currently executing
	line    string            // line currently executing
	env     []string          // environment list (for os/exec)
	envMap  map[string]string // environment mapping (matches env)
	stdout  string            // standard output from last 'go' command; for 'stdout' command
	stderr  string            // standard error from last 'go' command; for 'stderr' command
	stopped bool              // test wants to stop early
	start   time.Time         // time phase started
}

// setup sets up the test execution temporary directory and environment.
func (ts *testScript) setup() {
	StartProxy()
	ts.workdir = filepath.Join(testTmpDir, "script-"+ts.name)
	ts.check(os.MkdirAll(filepath.Join(ts.workdir, "tmp"), 0777))
	ts.check(os.MkdirAll(filepath.Join(ts.workdir, "gopath/src"), 0777))
	ts.cd = filepath.Join(ts.workdir, "gopath/src")
	ts.env = []string{
		"WORK=" + ts.workdir, // must be first for ts.abbrev
		"PATH=" + os.Getenv("PATH"),
		homeEnvName() + "=/no-home",
		"GOARCH=" + runtime.GOARCH,
		"GOCACHE=" + testGOCACHE,
		"GOOS=" + runtime.GOOS,
		"GOPATH=" + filepath.Join(ts.workdir, "gopath"),
		"GOPROXY=" + proxyURL,
		"GOROOT=" + testGOROOT,
		tempEnvName() + "=" + filepath.Join(ts.workdir, "tmp"),
		"devnull=" + os.DevNull,
	}
	if runtime.GOOS == "windows" {
		ts.env = append(ts.env, "exe=.exe")
	} else {
		ts.env = append(ts.env, "exe=")
	}
	ts.envMap = make(map[string]string)
	for _, kv := range ts.env {
		if i := strings.Index(kv, "="); i >= 0 {
			ts.envMap[kv[:i]] = kv[i+1:]
		}
	}
}

var execCache par.Cache

// run runs the test script.
func (ts *testScript) run() {
	// Truncate log at end of last phase marker,
	// discarding details of successful phase.
	rewind := func() {
		if !testing.Verbose() {
			ts.log.Truncate(ts.mark)
		}
	}

	// Insert elapsed time for phase at end of phase marker
	markTime := func() {
		if ts.mark > 0 && !ts.start.IsZero() {
			afterMark := append([]byte{}, ts.log.Bytes()[ts.mark:]...)
			ts.log.Truncate(ts.mark - 1) // cut \n and afterMark
			fmt.Fprintf(&ts.log, " (%.3fs)\n", time.Since(ts.start).Seconds())
			ts.log.Write(afterMark)
		}
		ts.start = time.Time{}
	}

	defer func() {
		markTime()
		// Flush testScript log to testing.T log.
		ts.t.Log("\n" + ts.abbrev(ts.log.String()))
	}()

	// Unpack archive.
	a, err := txtar.ParseFile(ts.file)
	ts.check(err)
	for _, f := range a.Files {
		name := ts.mkabs(ts.expand(f.Name))
		ts.check(os.MkdirAll(filepath.Dir(name), 0777))
		ts.check(ioutil.WriteFile(name, f.Data, 0666))
	}

	// With -v or -testwork, start log with full environment.
	if *testWork || testing.Verbose() {
		// Display environment.
		ts.cmdEnv(false, nil)
		fmt.Fprintf(&ts.log, "\n")
		ts.mark = ts.log.Len()
	}

	// Run script.
	// See testdata/script/README for documentation of script form.
	script := string(a.Comment)
Script:
	for script != "" {
		// Extract next line.
		ts.lineno++
		var line string
		if i := strings.Index(script, "\n"); i >= 0 {
			line, script = script[:i], script[i+1:]
		} else {
			line, script = script, ""
		}

		// # is a comment indicating the start of new phase.
		if strings.HasPrefix(line, "#") {
			// If there was a previous phase, it succeeded,
			// so rewind the log to delete its details (unless -v is in use).
			// If nothing has happened at all since the mark,
			// rewinding is a no-op and adding elapsed time
			// for doing nothing is meaningless, so don't.
			if ts.log.Len() > ts.mark {
				rewind()
				markTime()
			}
			// Print phase heading and mark start of phase output.
			fmt.Fprintf(&ts.log, "%s\n", line)
			ts.mark = ts.log.Len()
			ts.start = time.Now()
			continue
		}

		// Parse input line. Ignore blanks entirely.
		args := ts.parse(line)
		if len(args) == 0 {
			continue
		}

		// Echo command to log.
		fmt.Fprintf(&ts.log, "> %s\n", line)

		// Command prefix [cond] means only run this command if cond is satisfied.
		for strings.HasPrefix(args[0], "[") && strings.HasSuffix(args[0], "]") {
			cond := args[0]
			cond = cond[1 : len(cond)-1]
			cond = strings.TrimSpace(cond)
			args = args[1:]
			if len(args) == 0 {
				ts.fatalf("missing command after condition")
			}
			want := true
			if strings.HasPrefix(cond, "!") {
				want = false
				cond = strings.TrimSpace(cond[1:])
			}
			// Known conds are: $GOOS, $GOARCH, runtime.Compiler, and 'short' (for testing.Short).
			//
			// NOTE: If you make changes here, update testdata/script/README too!
			//
			ok := false
			switch cond {
			case runtime.GOOS, runtime.GOARCH, runtime.Compiler:
				ok = true
			case "short":
				ok = testing.Short()
			case "cgo":
				ok = canCgo
			case "msan":
				ok = canMSan
			case "race":
				ok = canRace
			case "net":
				ok = testenv.HasExternalNetwork()
			case "link":
				ok = testenv.HasLink()
			case "symlink":
				ok = testenv.HasSymlink()
			default:
				if strings.HasPrefix(cond, "exec:") {
					prog := cond[len("exec:"):]
					ok = execCache.Do(prog, func() interface{} {
						_, err := exec.LookPath(prog)
						return err == nil
					}).(bool)
					break
				}
				if !imports.KnownArch[cond] && !imports.KnownOS[cond] && cond != "gc" && cond != "gccgo" {
					ts.fatalf("unknown condition %q", cond)
				}
			}
			if ok != want {
				// Don't run rest of line.
				continue Script
			}
		}

		// Command prefix ! means negate the expectations about this command:
		// go command should fail, match should not be found, etc.
		neg := false
		if args[0] == "!" {
			neg = true
			args = args[1:]
			if len(args) == 0 {
				ts.fatalf("! on line by itself")
			}
		}

		// Run command.
		cmd := scriptCmds[args[0]]
		if cmd == nil {
			ts.fatalf("unknown command %q", args[0])
		}
		cmd(ts, neg, args[1:])

		// Command can ask script to stop early.
		if ts.stopped {
			return
		}
	}

	// Final phase ended.
	rewind()
	markTime()
	fmt.Fprintf(&ts.log, "PASS\n")
}

// scriptCmds are the script command implementations.
// Keep list and the implementations below sorted by name.
//
// NOTE: If you make changes here, update testdata/script/README too!
//
var scriptCmds = map[string]func(*testScript, bool, []string){
	"cd":     (*testScript).cmdCd,
	"cp":     (*testScript).cmdCp,
	"env":    (*testScript).cmdEnv,
	"exec":   (*testScript).cmdExec,
	"exists": (*testScript).cmdExists,
	"go":     (*testScript).cmdGo,
	"mkdir":  (*testScript).cmdMkdir,
	"rm":     (*testScript).cmdRm,
	"skip":   (*testScript).cmdSkip,
	"stale":  (*testScript).cmdStale,
	"stderr": (*testScript).cmdStderr,
	"stdout": (*testScript).cmdStdout,
	"stop":   (*testScript).cmdStop,
}

// cd changes to a different directory.
func (ts *testScript) cmdCd(neg bool, args []string) {
	if neg {
		ts.fatalf("unsupported: ! cd")
	}
	if len(args) != 1 {
		ts.fatalf("usage: cd dir")
	}

	dir := args[0]
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(ts.cd, dir)
	}
	info, err := os.Stat(dir)
	if os.IsNotExist(err) {
		ts.fatalf("directory %s does not exist", dir)
	}
	ts.check(err)
	if !info.IsDir() {
		ts.fatalf("%s is not a directory", dir)
	}
	ts.cd = dir
	fmt.Fprintf(&ts.log, "%s\n", ts.cd)
}

// cp copies files, maybe eventually directories.
func (ts *testScript) cmdCp(neg bool, args []string) {
	if neg {
		ts.fatalf("unsupported: ! cp")
	}
	if len(args) < 2 {
		ts.fatalf("usage: cp src... dst")
	}

	dst := ts.mkabs(args[len(args)-1])
	info, err := os.Stat(dst)
	dstDir := err == nil && info.IsDir()
	if len(args) > 2 && !dstDir {
		ts.fatalf("cp: destination %s is not a directory", dst)
	}

	for _, arg := range args[:len(args)-1] {
		src := ts.mkabs(arg)
		info, err := os.Stat(src)
		ts.check(err)
		data, err := ioutil.ReadFile(src)
		ts.check(err)
		targ := dst
		if dstDir {
			targ = filepath.Join(dst, filepath.Base(src))
		}
		ts.check(ioutil.WriteFile(targ, data, info.Mode()&0777))
	}
}

// env displays or adds to the environment.
func (ts *testScript) cmdEnv(neg bool, args []string) {
	if neg {
		ts.fatalf("unsupported: ! env")
	}
	if len(args) == 0 {
		printed := make(map[string]bool) // env list can have duplicates; only print effective value (from envMap) once
		for _, kv := range ts.env {
			k := kv[:strings.Index(kv, "=")]
			if !printed[k] {
				fmt.Fprintf(&ts.log, "%s=%s\n", k, ts.envMap[k])
			}
		}
		return
	}
	for _, env := range args {
		i := strings.Index(env, "=")
		if i < 0 {
			// Display value instead of setting it.
			fmt.Fprintf(&ts.log, "%s=%s\n", env, ts.envMap[env])
			continue
		}
		ts.env = append(ts.env, env)
		ts.envMap[env[:i]] = env[i+1:]
	}
}

// exec runs the given command.
func (ts *testScript) cmdExec(neg bool, args []string) {
	if len(args) < 1 {
		ts.fatalf("usage: exec program [args...]")
	}
	var err error
	ts.stdout, ts.stderr, err = ts.exec(args[0], args[1:]...)
	if ts.stdout != "" {
		fmt.Fprintf(&ts.log, "[stdout]\n%s", ts.stdout)
	}
	if ts.stderr != "" {
		fmt.Fprintf(&ts.log, "[stderr]\n%s", ts.stderr)
	}
	if err != nil {
		fmt.Fprintf(&ts.log, "[%v]\n", err)
		if !neg {
			ts.fatalf("unexpected command failure")
		}
	} else {
		if neg {
			ts.fatalf("unexpected command success")
		}
	}
}

// exists checks that the list of files exists.
func (ts *testScript) cmdExists(neg bool, args []string) {
	if len(args) == 0 {
		ts.fatalf("usage: exists file...")
	}

	for _, file := range args {
		file = ts.mkabs(file)
		info, err := os.Stat(file)
		if err == nil && neg {
			what := "file"
			if info.IsDir() {
				what = "directory"
			}
			ts.fatalf("%s %s unexpectedly exists", what, file)
		}
		if err != nil && !neg {
			ts.fatalf("%s does not exist", file)
		}
	}
}

// go runs the go command.
func (ts *testScript) cmdGo(neg bool, args []string) {
	ts.cmdExec(neg, append([]string{testGo}, args...))
}

// mkdir creates directories.
func (ts *testScript) cmdMkdir(neg bool, args []string) {
	if neg {
		ts.fatalf("unsupported: ! mkdir")
	}
	if len(args) < 1 {
		ts.fatalf("usage: mkdir dir...")
	}
	for _, arg := range args {
		ts.check(os.MkdirAll(ts.mkabs(arg), 0777))
	}
}

// rm removes files or directories.
func (ts *testScript) cmdRm(neg bool, args []string) {
	if neg {
		ts.fatalf("unsupported: ! rm")
	}
	if len(args) < 1 {
		ts.fatalf("usage: rm file...")
	}
	for _, arg := range args {
		file := ts.mkabs(arg)
		removeAll(file)              // does chmod and then attempts rm
		ts.check(os.RemoveAll(file)) // report error
	}
}

// skip marks the test skipped.
func (ts *testScript) cmdSkip(neg bool, args []string) {
	if len(args) > 1 {
		ts.fatalf("usage: skip [msg]")
	}
	if neg {
		ts.fatalf("unsupported: ! skip")
	}
	if len(args) == 1 {
		ts.t.Skip(args[0])
	}
	ts.t.Skip()
}

// stale checks that the named build targets are stale.
func (ts *testScript) cmdStale(neg bool, args []string) {
	if len(args) == 0 {
		ts.fatalf("usage: stale target...")
	}
	tmpl := "{{if .Error}}{{.ImportPath}}: {{.Error.Err}}{else}}"
	if neg {
		tmpl += "{{if .Stale}}{{.ImportPath}} is unexpectedly stale{{end}}"
	} else {
		tmpl += "{{if not .Stale}}{{.ImportPath}} is unexpectedly NOT stale{{end}}"
	}
	tmpl += "{{end}}"
	goArgs := append([]string{"list", "-e", "-f=" + tmpl}, args...)
	stdout, stderr, err := ts.exec(testGo, goArgs...)
	if err != nil {
		ts.fatalf("go list: %v\n%s%s", err, stdout, stderr)
	}
	if stdout != "" {
		ts.fatalf("%s", stdout)
	}
}

// stop stops execution of the test (marking it passed).
func (ts *testScript) cmdStop(neg bool, args []string) {
	if neg {
		ts.fatalf("unsupported: ! stop")
	}
	if len(args) > 1 {
		ts.fatalf("usage: stop [msg]")
	}
	if len(args) == 1 {
		fmt.Fprintf(&ts.log, "stop: %s\n", args[0])
	} else {
		fmt.Fprintf(&ts.log, "stop\n")
	}
	ts.stopped = true
}

// stdout checks that the last go command standard output matches a regexp.
func (ts *testScript) cmdStdout(neg bool, args []string) {
	scriptMatch(ts, neg, args, ts.stdout, "stdout")
}

// stderr checks that the last go command standard output matches a regexp.
func (ts *testScript) cmdStderr(neg bool, args []string) {
	scriptMatch(ts, neg, args, ts.stderr, "stderr")
}

// scriptMatch implements both stdout and stderr.
func scriptMatch(ts *testScript, neg bool, args []string, text, name string) {
	if len(args) != 1 {
		ts.fatalf("usage: %s 'pattern' (%q)", name, args)
	}
	re, err := regexp.Compile(`(?m)` + args[0])
	ts.check(err)
	if neg {
		if re.MatchString(text) {
			ts.fatalf("unexpected match for %#q found in %s: %s %q", args[0], name, text, re.FindString(text))
		}
	} else {
		if !re.MatchString(text) {
			ts.fatalf("no match for %#q found in %s", args[0], name)
		}
	}
}

// Helpers for command implementations.

// abbrev abbreviates the actual work directory in the string s to the literal string "$WORK".
func (ts *testScript) abbrev(s string) string {
	s = strings.Replace(s, ts.workdir, "$WORK", -1)
	if *testWork {
		// Expose actual $WORK value in environment dump on first line of work script,
		// so that the user can find out what directory -testwork left behind.
		s = "WORK=" + ts.workdir + "\n" + strings.TrimPrefix(s, "WORK=$WORK\n")
	}
	return s
}

// check calls ts.fatalf if err != nil.
func (ts *testScript) check(err error) {
	if err != nil {
		ts.fatalf("%v", err)
	}
}

// exec runs the given command line (an actual subprocess, not simulated)
// in ts.cd with environment ts.env and then returns collected standard output and standard error.
func (ts *testScript) exec(command string, args ...string) (stdout, stderr string, err error) {
	cmd := exec.Command(testGo, args...)
	cmd.Dir = ts.cd
	cmd.Env = append(ts.env, "PWD="+ts.cd)
	var stdoutBuf, stderrBuf strings.Builder
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf
	err = cmd.Run()
	return stdoutBuf.String(), stderrBuf.String(), err
}

// expand applies environment variable expansion to the string s.
func (ts *testScript) expand(s string) string {
	return os.Expand(s, func(key string) string { return ts.envMap[key] })
}

// fatalf aborts the test with the given failure message.
func (ts *testScript) fatalf(format string, args ...interface{}) {
	fmt.Fprintf(&ts.log, "FAIL: %s:%d: %s\n", ts.file, ts.lineno, fmt.Sprintf(format, args...))
	ts.t.FailNow()
}

// mkabs interprets file relative to the test script's current directory
// and returns the corresponding absolute path.
func (ts *testScript) mkabs(file string) string {
	if filepath.IsAbs(file) {
		return file
	}
	return filepath.Join(ts.cd, file)
}

// parse parses a single line as a list of space-separated arguments
// subject to environment variable expansion (but not resplitting).
// Single quotes around text disable splitting and expansion.
// To embed a single quote, double it: 'Don''t communicate by sharing memory.'
func (ts *testScript) parse(line string) []string {
	ts.line = line

	var (
		args   []string
		arg    string  // text of current arg so far (need to add line[start:i])
		start  = -1    // if >= 0, position where current arg text chunk starts
		quoted = false // currently processing quoted text
	)
	for i := 0; ; i++ {
		if !quoted && (i >= len(line) || line[i] == ' ' || line[i] == '\t' || line[i] == '\r' || line[i] == '#') {
			// Found arg-separating space.
			if start >= 0 {
				arg += ts.expand(line[start:i])
				args = append(args, arg)
				start = -1
				arg = ""
			}
			if i >= len(line) || line[i] == '#' {
				break
			}
			continue
		}
		if i >= len(line) {
			ts.fatalf("unterminated quoted argument")
		}
		if line[i] == '\'' {
			if !quoted {
				// starting a quoted chunk
				if start >= 0 {
					arg += ts.expand(line[start:i])
				}
				start = i + 1
				quoted = true
				continue
			}
			// 'foo''bar' means foo'bar, like in rc shell and Pascal.
			if i+1 < len(line) && line[i+1] == '\'' {
				arg += line[start:i]
				start = i + 1
				i++ // skip over second ' before next iteration
				continue
			}
			// ending a quoted chunk
			arg += line[start:i]
			start = i + 1
			quoted = false
			continue
		}
		// found character worth saving; make sure we're saving
		if start < 0 {
			start = i
		}
	}
	return args
}
