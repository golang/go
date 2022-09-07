// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Script-driven tests.
// See testdata/script/README for an overview.

package main_test

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"internal/testenv"
	"internal/txtar"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"cmd/go/internal/cfg"
	"cmd/go/internal/imports"
	"cmd/go/internal/par"
	"cmd/go/internal/robustio"
	"cmd/go/internal/work"
	"cmd/internal/sys"
)

var testSum = flag.String("testsum", "", `may be tidy, listm, or listall. If set, TestScript generates a go.sum file at the beginning of each test and updates test files if they pass.`)

// TestScript runs the tests in testdata/script/*.txt.
func TestScript(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.SkipIfShortAndSlow(t)

	var (
		ctx         = context.Background()
		gracePeriod = 100 * time.Millisecond
	)
	if deadline, ok := t.Deadline(); ok {
		timeout := time.Until(deadline)

		// If time allows, increase the termination grace period to 5% of the
		// remaining time.
		if gp := timeout / 20; gp > gracePeriod {
			gracePeriod = gp
		}

		// When we run commands that execute subprocesses, we want to reserve two
		// grace periods to clean up. We will send the first termination signal when
		// the context expires, then wait one grace period for the process to
		// produce whatever useful output it can (such as a stack trace). After the
		// first grace period expires, we'll escalate to os.Kill, leaving the second
		// grace period for the test function to record its output before the test
		// process itself terminates.
		timeout -= 2 * gracePeriod

		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		t.Cleanup(cancel)
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
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			ts := &testScript{
				t:           t,
				ctx:         ctx,
				cancel:      cancel,
				gracePeriod: gracePeriod,
				name:        name,
				file:        file,
			}
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
	t           testing.TB
	ctx         context.Context
	cancel      context.CancelFunc
	gracePeriod time.Duration
	workdir     string            // temporary work dir ($WORK)
	log         bytes.Buffer      // test execution log (printed at end of test)
	mark        int               // offset of next log truncation
	cd          string            // current directory during test execution; initially $WORK/gopath/src
	name        string            // short name of test ("foo")
	file        string            // full file name ("testdata/script/foo.txt")
	lineno      int               // line number currently executing
	line        string            // line currently executing
	env         []string          // environment list (for os/exec)
	envMap      map[string]string // environment mapping (matches env)
	stdout      string            // standard output from last 'go' command; for 'stdout' command
	stderr      string            // standard error from last 'go' command; for 'stderr' command
	stopped     bool              // test wants to stop early
	start       time.Time         // time phase started
	background  []*backgroundCmd  // backgrounded 'exec' and 'go' commands
}

type backgroundCmd struct {
	want           simpleStatus
	args           []string
	done           <-chan struct{}
	err            error
	stdout, stderr strings.Builder
}

type simpleStatus string

const (
	success          simpleStatus = ""
	failure          simpleStatus = "!"
	successOrFailure simpleStatus = "?"
)

var extraEnvKeys = []string{
	"SYSTEMROOT",         // must be preserved on Windows to find DLLs; golang.org/issue/25210
	"WINDIR",             // must be preserved on Windows to be able to run PowerShell command; golang.org/issue/30711
	"LD_LIBRARY_PATH",    // must be preserved on Unix systems to find shared libraries
	"LIBRARY_PATH",       // allow override of non-standard static library paths
	"C_INCLUDE_PATH",     // allow override non-standard include paths
	"CC",                 // don't lose user settings when invoking cgo
	"GO_TESTING_GOTOOLS", // for gccgo testing
	"GCCGO",              // for gccgo testing
	"GCCGOTOOLDIR",       // for gccgo testing
}

// setup sets up the test execution temporary directory and environment.
func (ts *testScript) setup() {
	if err := ts.ctx.Err(); err != nil {
		ts.t.Fatalf("test interrupted during setup: %v", err)
	}

	StartProxy()
	ts.workdir = filepath.Join(testTmpDir, "script-"+ts.name)
	ts.check(os.MkdirAll(filepath.Join(ts.workdir, "tmp"), 0777))
	ts.check(os.MkdirAll(filepath.Join(ts.workdir, "gopath/src"), 0777))
	ts.cd = filepath.Join(ts.workdir, "gopath/src")
	version, err := goVersion()
	if err != nil {
		ts.t.Fatal(err)
	}
	ts.env = []string{
		"WORK=" + ts.workdir, // must be first for ts.abbrev
		pathEnvName() + "=" + testBin + string(filepath.ListSeparator) + os.Getenv(pathEnvName()),
		homeEnvName() + "=/no-home",
		"CCACHE_DISABLE=1", // ccache breaks with non-existent HOME
		"GOARCH=" + runtime.GOARCH,
		"TESTGO_GOHOSTARCH=" + goHostArch,
		"GOCACHE=" + testGOCACHE,
		"GODEBUG=" + os.Getenv("GODEBUG"),
		"GOEXE=" + cfg.ExeSuffix,
		"GOEXPERIMENT=" + os.Getenv("GOEXPERIMENT"),
		"GOOS=" + runtime.GOOS,
		"TESTGO_GOHOSTOS=" + goHostOS,
		"GOPATH=" + filepath.Join(ts.workdir, "gopath"),
		"GOPROXY=" + proxyURL,
		"GOPRIVATE=",
		"GOROOT=" + testGOROOT,
		"GOROOT_FINAL=" + testGOROOT_FINAL, // causes spurious rebuilds and breaks the "stale" built-in if not propagated
		"GOTRACEBACK=system",
		"TESTGO_GOROOT=" + testGOROOT,
		"GOSUMDB=" + testSumDBVerifierKey,
		"GONOPROXY=",
		"GONOSUMDB=",
		"GOVCS=*:all",
		"PWD=" + ts.cd,
		tempEnvName() + "=" + filepath.Join(ts.workdir, "tmp"),
		"devnull=" + os.DevNull,
		"goversion=" + version,
		"CMDGO_TEST_RUN_MAIN=true",
	}
	if testenv.Builder() != "" || os.Getenv("GIT_TRACE_CURL") == "1" {
		// To help diagnose https://go.dev/issue/52545,
		// enable tracing for Git HTTPS requests.
		ts.env = append(ts.env,
			"GIT_TRACE_CURL=1",
			"GIT_TRACE_CURL_NO_DATA=1",
			"GIT_REDACT_COOKIES=o,SSO,GSSO_Uberproxy")
	}
	if !testenv.HasExternalNetwork() {
		ts.env = append(ts.env, "TESTGONETWORK=panic", "TESTGOVCS=panic")
	}
	if os.Getenv("CGO_ENABLED") != "" || runtime.GOOS != goHostOS || runtime.GOARCH != goHostArch {
		// If the actual CGO_ENABLED might not match the cmd/go default, set it
		// explicitly in the environment. Otherwise, leave it unset so that we also
		// cover the default behaviors.
		ts.env = append(ts.env, "CGO_ENABLED="+cgoEnabled)
	}

	for _, key := range extraEnvKeys {
		if val := os.Getenv(key); val != "" {
			ts.env = append(ts.env, key+"="+val)
		}
	}

	ts.envMap = make(map[string]string)
	for _, kv := range ts.env {
		if i := strings.Index(kv, "="); i >= 0 {
			ts.envMap[kv[:i]] = kv[i+1:]
		}
	}
	// Add entries for ${:} and ${/} to make it easier to write platform-independent
	// environment variables.
	ts.envMap["/"] = string(os.PathSeparator)
	ts.envMap[":"] = string(os.PathListSeparator)

	fmt.Fprintf(&ts.log, "# (%s)\n", time.Now().UTC().Format(time.RFC3339))
	ts.mark = ts.log.Len()
}

// goVersion returns the current Go version.
func goVersion() (string, error) {
	tags := build.Default.ReleaseTags
	version := tags[len(tags)-1]
	if !regexp.MustCompile(`^go([1-9][0-9]*)\.(0|[1-9][0-9]*)$`).MatchString(version) {
		return "", fmt.Errorf("invalid go version %q", version)
	}
	return version[2:], nil
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
		// On a normal exit from the test loop, background processes are cleaned up
		// before we print PASS. If we return early (e.g., due to a test failure),
		// don't print anything about the processes that were still running.
		ts.cancel()
		for _, bg := range ts.background {
			<-bg.done
		}
		ts.background = nil

		markTime()
		// Flush testScript log to testing.T log.
		ts.t.Log("\n" + ts.abbrev(ts.log.String()))
	}()

	// Unpack archive.
	a, err := txtar.ParseFile(ts.file)
	ts.check(err)
	for _, f := range a.Files {
		name := ts.mkabs(ts.expand(f.Name, false))
		ts.check(os.MkdirAll(filepath.Dir(name), 0777))
		ts.check(os.WriteFile(name, f.Data, 0666))
	}

	// With -v or -testwork, start log with full environment.
	if *testWork || testing.Verbose() {
		// Display environment.
		ts.cmdEnv(success, nil)
		fmt.Fprintf(&ts.log, "\n")
		ts.mark = ts.log.Len()
	}

	// With -testsum, if a go.mod file is present in the test's initial
	// working directory, run 'go mod tidy'.
	if *testSum != "" {
		if ts.updateSum(a) {
			defer func() {
				if ts.t.Failed() {
					return
				}
				data := txtar.Format(a)
				if err := os.WriteFile(ts.file, data, 0666); err != nil {
					ts.t.Errorf("rewriting test file: %v", err)
				}
			}()
		}
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
		parsed := ts.parse(line)
		if parsed.name == "" {
			if parsed.want != "" || len(parsed.conds) > 0 {
				ts.fatalf("missing command")
			}
			continue
		}

		// Echo command to log.
		fmt.Fprintf(&ts.log, "> %s\n", line)

		for _, cond := range parsed.conds {
			if err := ts.ctx.Err(); err != nil {
				ts.fatalf("test interrupted: %v", err)
			}

			// Known conds are: $GOOS, $GOARCH, runtime.Compiler, and 'short' (for testing.Short).
			//
			// NOTE: If you make changes here, update testdata/script/README too!
			//
			ok := false
			switch cond.tag {
			case runtime.GOOS, runtime.GOARCH, runtime.Compiler:
				ok = true
			case "cross":
				ok = goHostOS != runtime.GOOS || goHostArch != runtime.GOARCH
			case "short":
				ok = testing.Short()
			case "cgo":
				ok = canCgo
			case "msan":
				ok = canMSan
			case "asan":
				ok = canASan
			case "race":
				ok = canRace
			case "fuzz":
				ok = canFuzz
			case "fuzz-instrumented":
				ok = fuzzInstrumented
			case "net":
				ok = testenv.HasExternalNetwork()
			case "link":
				ok = testenv.HasLink()
			case "root":
				ok = os.Geteuid() == 0
			case "symlink":
				ok = testenv.HasSymlink()
			case "case-sensitive":
				ok, err = isCaseSensitive()
				if err != nil {
					ts.fatalf("%v", err)
				}
			case "trimpath":
				if info, _ := debug.ReadBuildInfo(); info == nil {
					ts.fatalf("missing build info")
				} else {
					for _, s := range info.Settings {
						if s.Key == "-trimpath" && s.Value == "true" {
							ok = true
							break
						}
					}
				}
			case "mismatched-goroot":
				ok = testGOROOT_FINAL != "" && testGOROOT_FINAL != testGOROOT
			default:
				if strings.HasPrefix(cond.tag, "exec:") {
					prog := cond.tag[len("exec:"):]
					ok = execCache.Do(prog, func() any {
						if runtime.GOOS == "plan9" && prog == "git" {
							// The Git command is usually not the real Git on Plan 9.
							// See https://golang.org/issues/29640.
							return false
						}
						_, err := exec.LookPath(prog)
						return err == nil
					}).(bool)
					break
				}
				if strings.HasPrefix(cond.tag, "GODEBUG:") {
					value := strings.TrimPrefix(cond.tag, "GODEBUG:")
					parts := strings.Split(os.Getenv("GODEBUG"), ",")
					for _, p := range parts {
						if strings.TrimSpace(p) == value {
							ok = true
							break
						}
					}
					break
				}
				if strings.HasPrefix(cond.tag, "buildmode:") {
					value := strings.TrimPrefix(cond.tag, "buildmode:")
					ok = sys.BuildModeSupported(runtime.Compiler, value, runtime.GOOS, runtime.GOARCH)
					break
				}
				if !imports.KnownArch[cond.tag] && !imports.KnownOS[cond.tag] && cond.tag != "gc" && cond.tag != "gccgo" {
					ts.fatalf("unknown condition %q", cond.tag)
				}
			}
			if ok != cond.want {
				// Don't run rest of line.
				continue Script
			}
		}

		// Run command.
		cmd := scriptCmds[parsed.name]
		if cmd == nil {
			ts.fatalf("unknown command %q", parsed.name)
		}
		cmd(ts, parsed.want, parsed.args)

		// Command can ask script to stop early.
		if ts.stopped {
			// Break instead of returning, so that we check the status of any
			// background processes and print PASS.
			break
		}
	}

	ts.cancel()
	ts.cmdWait(success, nil)

	// Final phase ended.
	rewind()
	markTime()
	if !ts.stopped {
		fmt.Fprintf(&ts.log, "PASS\n")
	}
}

var (
	onceCaseSensitive sync.Once
	caseSensitive     bool
	caseSensitiveErr  error
)

func isCaseSensitive() (bool, error) {
	onceCaseSensitive.Do(func() {
		tmpdir, err := os.MkdirTemp("", "case-sensitive")
		if err != nil {
			caseSensitiveErr = fmt.Errorf("failed to create directory to determine case-sensitivity: %w", err)
			return
		}
		defer os.RemoveAll(tmpdir)

		fcap := filepath.Join(tmpdir, "FILE")
		if err := os.WriteFile(fcap, []byte{}, 0644); err != nil {
			caseSensitiveErr = fmt.Errorf("error writing file to determine case-sensitivity: %w", err)
			return
		}

		flow := filepath.Join(tmpdir, "file")
		_, err = os.ReadFile(flow)
		switch {
		case err == nil:
			caseSensitive = false
			return
		case os.IsNotExist(err):
			caseSensitive = true
			return
		default:
			caseSensitiveErr = fmt.Errorf("unexpected error reading file when determining case-sensitivity: %w", err)
		}
	})

	return caseSensitive, caseSensitiveErr
}

// scriptCmds are the script command implementations.
// Keep list and the implementations below sorted by name.
//
// NOTE: If you make changes here, update testdata/script/README too!
var scriptCmds = map[string]func(*testScript, simpleStatus, []string){
	"addcrlf": (*testScript).cmdAddcrlf,
	"cc":      (*testScript).cmdCc,
	"cd":      (*testScript).cmdCd,
	"chmod":   (*testScript).cmdChmod,
	"cmp":     (*testScript).cmdCmp,
	"cmpenv":  (*testScript).cmdCmpenv,
	"cp":      (*testScript).cmdCp,
	"env":     (*testScript).cmdEnv,
	"exec":    (*testScript).cmdExec,
	"exists":  (*testScript).cmdExists,
	"go":      (*testScript).cmdGo,
	"grep":    (*testScript).cmdGrep,
	"mkdir":   (*testScript).cmdMkdir,
	"mv":      (*testScript).cmdMv,
	"rm":      (*testScript).cmdRm,
	"skip":    (*testScript).cmdSkip,
	"sleep":   (*testScript).cmdSleep,
	"stale":   (*testScript).cmdStale,
	"stderr":  (*testScript).cmdStderr,
	"stdout":  (*testScript).cmdStdout,
	"stop":    (*testScript).cmdStop,
	"symlink": (*testScript).cmdSymlink,
	"wait":    (*testScript).cmdWait,
}

// When expanding shell variables for these commands, we apply regexp quoting to
// expanded strings within the first argument.
var regexpCmd = map[string]bool{
	"grep":   true,
	"stderr": true,
	"stdout": true,
}

// addcrlf adds CRLF line endings to the named files.
func (ts *testScript) cmdAddcrlf(want simpleStatus, args []string) {
	if len(args) == 0 {
		ts.fatalf("usage: addcrlf file...")
	}

	for _, file := range args {
		file = ts.mkabs(file)
		data, err := os.ReadFile(file)
		ts.check(err)
		ts.check(os.WriteFile(file, bytes.ReplaceAll(data, []byte("\n"), []byte("\r\n")), 0666))
	}
}

// cc runs the C compiler along with platform specific options.
func (ts *testScript) cmdCc(want simpleStatus, args []string) {
	if len(args) < 1 || (len(args) == 1 && args[0] == "&") {
		ts.fatalf("usage: cc args... [&]")
	}

	b := work.NewBuilder(ts.workdir)
	defer func() {
		if err := b.Close(); err != nil {
			ts.fatalf("%v", err)
		}
	}()
	ts.cmdExec(want, append(b.GccCmd(".", ""), args...))
}

// cd changes to a different directory.
func (ts *testScript) cmdCd(want simpleStatus, args []string) {
	if want != success {
		ts.fatalf("unsupported: %v cd", want)
	}
	if len(args) != 1 {
		ts.fatalf("usage: cd dir")
	}

	dir := filepath.FromSlash(args[0])
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
	ts.envMap["PWD"] = dir
	fmt.Fprintf(&ts.log, "%s\n", ts.cd)
}

// chmod changes permissions for a file or directory.
func (ts *testScript) cmdChmod(want simpleStatus, args []string) {
	if want != success {
		ts.fatalf("unsupported: %v chmod", want)
	}
	if len(args) < 2 {
		ts.fatalf("usage: chmod perm paths...")
	}
	perm, err := strconv.ParseUint(args[0], 0, 32)
	if err != nil || perm&uint64(fs.ModePerm) != perm {
		ts.fatalf("invalid mode: %s", args[0])
	}
	for _, arg := range args[1:] {
		path := arg
		if !filepath.IsAbs(path) {
			path = filepath.Join(ts.cd, arg)
		}
		err := os.Chmod(path, fs.FileMode(perm))
		ts.check(err)
	}
}

// cmp compares two files.
func (ts *testScript) cmdCmp(want simpleStatus, args []string) {
	quiet := false
	if len(args) > 0 && args[0] == "-q" {
		quiet = true
		args = args[1:]
	}
	if len(args) != 2 {
		ts.fatalf("usage: cmp file1 file2")
	}
	ts.doCmdCmp(want, args, false, quiet)
}

// cmpenv compares two files with environment variable substitution.
func (ts *testScript) cmdCmpenv(want simpleStatus, args []string) {
	quiet := false
	if len(args) > 0 && args[0] == "-q" {
		quiet = true
		args = args[1:]
	}
	if len(args) != 2 {
		ts.fatalf("usage: cmpenv file1 file2")
	}
	ts.doCmdCmp(want, args, true, quiet)
}

func (ts *testScript) doCmdCmp(want simpleStatus, args []string, env, quiet bool) {
	name1, name2 := args[0], args[1]
	var text1, text2 string
	switch name1 {
	case "stdout":
		text1 = ts.stdout
	case "stderr":
		text1 = ts.stderr
	default:
		data, err := os.ReadFile(ts.mkabs(name1))
		ts.check(err)
		text1 = string(data)
	}

	data, err := os.ReadFile(ts.mkabs(name2))
	ts.check(err)
	text2 = string(data)

	if env {
		text1 = ts.expand(text1, false)
		text2 = ts.expand(text2, false)
	}

	eq := text1 == text2
	if !eq && !quiet && want != failure {
		fmt.Fprintf(&ts.log, "[diff -%s +%s]\n%s\n", name1, name2, diff(text1, text2))
	}
	switch want {
	case failure:
		if eq {
			ts.fatalf("%s and %s do not differ", name1, name2)
		}
	case success:
		if !eq {
			ts.fatalf("%s and %s differ", name1, name2)
		}
	case successOrFailure:
		if eq {
			fmt.Fprintf(&ts.log, "%s and %s do not differ\n", name1, name2)
		} else {
			fmt.Fprintf(&ts.log, "%s and %s differ\n", name1, name2)
		}
	default:
		ts.fatalf("unsupported: %v cmp", want)
	}
}

// cp copies files, maybe eventually directories.
func (ts *testScript) cmdCp(want simpleStatus, args []string) {
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
		var (
			src  string
			data []byte
			mode fs.FileMode
		)
		switch arg {
		case "stdout":
			src = arg
			data = []byte(ts.stdout)
			mode = 0666
		case "stderr":
			src = arg
			data = []byte(ts.stderr)
			mode = 0666
		default:
			src = ts.mkabs(arg)
			info, err := os.Stat(src)
			ts.check(err)
			mode = info.Mode() & 0777
			data, err = os.ReadFile(src)
			ts.check(err)
		}
		targ := dst
		if dstDir {
			targ = filepath.Join(dst, filepath.Base(src))
		}
		err := os.WriteFile(targ, data, mode)
		switch want {
		case failure:
			if err == nil {
				ts.fatalf("unexpected command success")
			}
		case success:
			ts.check(err)
		}
	}
}

// env displays or adds to the environment.
func (ts *testScript) cmdEnv(want simpleStatus, args []string) {
	if want != success {
		ts.fatalf("unsupported: %v env", want)
	}

	conv := func(s string) string { return s }
	if len(args) > 0 && args[0] == "-r" {
		conv = regexp.QuoteMeta
		args = args[1:]
	}

	var out strings.Builder
	if len(args) == 0 {
		printed := make(map[string]bool) // env list can have duplicates; only print effective value (from envMap) once
		for _, kv := range ts.env {
			k := kv[:strings.Index(kv, "=")]
			if !printed[k] {
				fmt.Fprintf(&out, "%s=%s\n", k, ts.envMap[k])
			}
		}
	} else {
		for _, env := range args {
			i := strings.Index(env, "=")
			if i < 0 {
				// Display value instead of setting it.
				fmt.Fprintf(&out, "%s=%s\n", env, ts.envMap[env])
				continue
			}
			key, val := env[:i], conv(env[i+1:])
			ts.env = append(ts.env, key+"="+val)
			ts.envMap[key] = val
		}
	}
	if out.Len() > 0 || len(args) > 0 {
		ts.stdout = out.String()
		ts.log.WriteString(out.String())
	}
}

// exec runs the given command.
func (ts *testScript) cmdExec(want simpleStatus, args []string) {
	if len(args) < 1 || (len(args) == 1 && args[0] == "&") {
		ts.fatalf("usage: exec program [args...] [&]")
	}

	background := false
	if len(args) > 0 && args[len(args)-1] == "&" {
		background = true
		args = args[:len(args)-1]
	}

	bg, err := ts.startBackground(want, args[0], args[1:]...)
	if err != nil {
		ts.fatalf("unexpected error starting command: %v", err)
	}
	if background {
		ts.stdout, ts.stderr = "", ""
		ts.background = append(ts.background, bg)
		return
	}

	<-bg.done
	ts.stdout = bg.stdout.String()
	ts.stderr = bg.stderr.String()
	if ts.stdout != "" {
		fmt.Fprintf(&ts.log, "[stdout]\n%s", ts.stdout)
	}
	if ts.stderr != "" {
		fmt.Fprintf(&ts.log, "[stderr]\n%s", ts.stderr)
	}
	if bg.err != nil {
		fmt.Fprintf(&ts.log, "[%v]\n", bg.err)
	}
	ts.checkCmd(bg)
}

// exists checks that the list of files exists.
func (ts *testScript) cmdExists(want simpleStatus, args []string) {
	if want == successOrFailure {
		ts.fatalf("unsupported: %v exists", want)
	}
	var readonly, exec bool
loop:
	for len(args) > 0 {
		switch args[0] {
		case "-readonly":
			readonly = true
			args = args[1:]
		case "-exec":
			exec = true
			args = args[1:]
		default:
			break loop
		}
	}
	if len(args) == 0 {
		ts.fatalf("usage: exists [-readonly] [-exec] file...")
	}

	for _, file := range args {
		file = ts.mkabs(file)
		info, err := os.Stat(file)
		if err == nil && want == failure {
			what := "file"
			if info.IsDir() {
				what = "directory"
			}
			ts.fatalf("%s %s unexpectedly exists", what, file)
		}
		if err != nil && want == success {
			ts.fatalf("%s does not exist", file)
		}
		if err == nil && want == success && readonly && info.Mode()&0222 != 0 {
			ts.fatalf("%s exists but is writable", file)
		}
		if err == nil && want == success && exec && runtime.GOOS != "windows" && info.Mode()&0111 == 0 {
			ts.fatalf("%s exists but is not executable", file)
		}
	}
}

// go runs the go command.
func (ts *testScript) cmdGo(want simpleStatus, args []string) {
	ts.cmdExec(want, append([]string{testGo}, args...))
}

// mkdir creates directories.
func (ts *testScript) cmdMkdir(want simpleStatus, args []string) {
	if want != success {
		ts.fatalf("unsupported: %v mkdir", want)
	}
	if len(args) < 1 {
		ts.fatalf("usage: mkdir dir...")
	}
	for _, arg := range args {
		ts.check(os.MkdirAll(ts.mkabs(arg), 0777))
	}
}

func (ts *testScript) cmdMv(want simpleStatus, args []string) {
	if want != success {
		ts.fatalf("unsupported: %v mv", want)
	}
	if len(args) != 2 {
		ts.fatalf("usage: mv old new")
	}
	ts.check(os.Rename(ts.mkabs(args[0]), ts.mkabs(args[1])))
}

// rm removes files or directories.
func (ts *testScript) cmdRm(want simpleStatus, args []string) {
	if want != success {
		ts.fatalf("unsupported: %v rm", want)
	}
	if len(args) < 1 {
		ts.fatalf("usage: rm file...")
	}
	for _, arg := range args {
		file := ts.mkabs(arg)
		removeAll(file)                    // does chmod and then attempts rm
		ts.check(robustio.RemoveAll(file)) // report error
	}
}

// skip marks the test skipped.
func (ts *testScript) cmdSkip(want simpleStatus, args []string) {
	if len(args) > 1 {
		ts.fatalf("usage: skip [msg]")
	}
	if want != success {
		ts.fatalf("unsupported: %v skip", want)
	}

	// Before we mark the test as skipped, shut down any background processes and
	// make sure they have returned the correct status.
	ts.cancel()
	ts.cmdWait(success, nil)

	if len(args) == 1 {
		ts.t.Skip(args[0])
	}
	ts.t.Skip()
}

// sleep sleeps for the given duration
func (ts *testScript) cmdSleep(want simpleStatus, args []string) {
	if len(args) != 1 {
		ts.fatalf("usage: sleep duration")
	}
	d, err := time.ParseDuration(args[0])
	if err != nil {
		ts.fatalf("sleep: %v", err)
	}
	if want != success {
		ts.fatalf("unsupported: %v sleep", want)
	}
	time.Sleep(d)
}

// stale checks that the named build targets are stale.
func (ts *testScript) cmdStale(want simpleStatus, args []string) {
	if len(args) == 0 {
		ts.fatalf("usage: stale target...")
	}
	tmpl := "{{if .Error}}{{.ImportPath}}: {{.Error.Err}}{{else}}"
	switch want {
	case failure:
		tmpl += `{{if .Stale}}{{.ImportPath}} ({{.Target}}) is unexpectedly stale:{{"\n\t"}}{{.StaleReason}}{{end}}`
	case success:
		tmpl += "{{if not .Stale}}{{.ImportPath}} ({{.Target}}) is unexpectedly NOT stale{{end}}"
	default:
		ts.fatalf("unsupported: %v stale", want)
	}
	tmpl += "{{end}}"
	goArgs := append([]string{"list", "-e", "-f=" + tmpl}, args...)
	stdout, stderr, err := ts.exec(testGo, goArgs...)
	if err != nil {
		// Print stdout before stderr, because stderr may explain the error
		// independent of whatever we may have printed to stdout.
		ts.fatalf("go list: %v\n%s%s", err, stdout, stderr)
	}
	if stdout != "" {
		// Print stderr before stdout, because stderr may contain verbose
		// debugging info (for example, if GODEBUG=gocachehash=1 is set)
		// and we know that stdout contains a useful summary.
		ts.fatalf("%s%s", stderr, stdout)
	}
}

// stdout checks that the last go command standard output matches a regexp.
func (ts *testScript) cmdStdout(want simpleStatus, args []string) {
	scriptMatch(ts, want, args, ts.stdout, "stdout")
}

// stderr checks that the last go command standard output matches a regexp.
func (ts *testScript) cmdStderr(want simpleStatus, args []string) {
	scriptMatch(ts, want, args, ts.stderr, "stderr")
}

// grep checks that file content matches a regexp.
// Like stdout/stderr and unlike Unix grep, it accepts Go regexp syntax.
func (ts *testScript) cmdGrep(want simpleStatus, args []string) {
	scriptMatch(ts, want, args, "", "grep")
}

// scriptMatch implements both stdout and stderr.
func scriptMatch(ts *testScript, want simpleStatus, args []string, text, name string) {
	if want == successOrFailure {
		ts.fatalf("unsupported: %v %s", want, name)
	}

	n := 0
	if len(args) >= 1 && strings.HasPrefix(args[0], "-count=") {
		if want == failure {
			ts.fatalf("cannot use -count= with negated match")
		}
		var err error
		n, err = strconv.Atoi(args[0][len("-count="):])
		if err != nil {
			ts.fatalf("bad -count=: %v", err)
		}
		if n < 1 {
			ts.fatalf("bad -count=: must be at least 1")
		}
		args = args[1:]
	}
	quiet := false
	if len(args) >= 1 && args[0] == "-q" {
		quiet = true
		args = args[1:]
	}

	extraUsage := ""
	wantArgs := 1
	if name == "grep" {
		extraUsage = " file"
		wantArgs = 2
	}
	if len(args) != wantArgs {
		ts.fatalf("usage: %s [-count=N] 'pattern'%s", name, extraUsage)
	}

	pattern := `(?m)` + args[0]
	re, err := regexp.Compile(pattern)
	if err != nil {
		ts.fatalf("regexp.Compile(%q): %v", pattern, err)
	}

	isGrep := name == "grep"
	if isGrep {
		name = args[1] // for error messages
		data, err := os.ReadFile(ts.mkabs(args[1]))
		ts.check(err)
		text = string(data)
	}

	// Matching against workdir would be misleading.
	text = strings.ReplaceAll(text, ts.workdir, "$WORK")

	switch want {
	case failure:
		if re.MatchString(text) {
			if isGrep && !quiet {
				fmt.Fprintf(&ts.log, "[%s]\n%s\n", name, text)
			}
			ts.fatalf("unexpected match for %#q found in %s: %s", pattern, name, re.FindString(text))
		}

	case success:
		if !re.MatchString(text) {
			if isGrep && !quiet {
				fmt.Fprintf(&ts.log, "[%s]\n%s\n", name, text)
			}
			ts.fatalf("no match for %#q found in %s", pattern, name)
		}
		if n > 0 {
			count := len(re.FindAllString(text, -1))
			if count != n {
				if isGrep && !quiet {
					fmt.Fprintf(&ts.log, "[%s]\n%s\n", name, text)
				}
				ts.fatalf("have %d matches for %#q, want %d", count, pattern, n)
			}
		}
	}
}

// stop stops execution of the test (marking it passed).
func (ts *testScript) cmdStop(want simpleStatus, args []string) {
	if want != success {
		ts.fatalf("unsupported: %v stop", want)
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

// symlink creates a symbolic link.
func (ts *testScript) cmdSymlink(want simpleStatus, args []string) {
	if want != success {
		ts.fatalf("unsupported: %v symlink", want)
	}
	if len(args) != 3 || args[1] != "->" {
		ts.fatalf("usage: symlink file -> target")
	}
	// Note that the link target args[2] is not interpreted with mkabs:
	// it will be interpreted relative to the directory file is in.
	ts.check(os.Symlink(args[2], ts.mkabs(args[0])))
}

// wait waits for background commands to exit, setting stderr and stdout to their result.
func (ts *testScript) cmdWait(want simpleStatus, args []string) {
	if want != success {
		ts.fatalf("unsupported: %v wait", want)
	}
	if len(args) > 0 {
		ts.fatalf("usage: wait")
	}

	var stdouts, stderrs []string
	for _, bg := range ts.background {
		<-bg.done

		args := append([]string{filepath.Base(bg.args[0])}, bg.args[1:]...)
		fmt.Fprintf(&ts.log, "[background] %s: %v\n", strings.Join(args, " "), bg.err)

		cmdStdout := bg.stdout.String()
		if cmdStdout != "" {
			fmt.Fprintf(&ts.log, "[stdout]\n%s", cmdStdout)
			stdouts = append(stdouts, cmdStdout)
		}

		cmdStderr := bg.stderr.String()
		if cmdStderr != "" {
			fmt.Fprintf(&ts.log, "[stderr]\n%s", cmdStderr)
			stderrs = append(stderrs, cmdStderr)
		}

		ts.checkCmd(bg)
	}

	ts.stdout = strings.Join(stdouts, "")
	ts.stderr = strings.Join(stderrs, "")
	ts.background = nil
}

// Helpers for command implementations.

// abbrev abbreviates the actual work directory in the string s to the literal string "$WORK".
func (ts *testScript) abbrev(s string) string {
	s = strings.ReplaceAll(s, ts.workdir, "$WORK")
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

func (ts *testScript) checkCmd(bg *backgroundCmd) {
	select {
	case <-bg.done:
	default:
		panic("checkCmd called when not done")
	}

	if bg.err == nil {
		if bg.want == failure {
			ts.fatalf("unexpected command success")
		}
		return
	}

	if errors.Is(bg.err, context.DeadlineExceeded) {
		ts.fatalf("test timed out while running command")
	}

	if errors.Is(bg.err, context.Canceled) {
		// The process was still running at the end of the test.
		// The test must not depend on its exit status.
		if bg.want != successOrFailure {
			ts.fatalf("unexpected background command remaining at test end")
		}
		return
	}

	if bg.want == success {
		ts.fatalf("unexpected command failure")
	}
}

// exec runs the given command line (an actual subprocess, not simulated)
// in ts.cd with environment ts.env and then returns collected standard output and standard error.
func (ts *testScript) exec(command string, args ...string) (stdout, stderr string, err error) {
	bg, err := ts.startBackground(success, command, args...)
	if err != nil {
		return "", "", err
	}
	<-bg.done
	return bg.stdout.String(), bg.stderr.String(), bg.err
}

// startBackground starts the given command line (an actual subprocess, not simulated)
// in ts.cd with environment ts.env.
func (ts *testScript) startBackground(want simpleStatus, command string, args ...string) (*backgroundCmd, error) {
	done := make(chan struct{})
	bg := &backgroundCmd{
		want: want,
		args: append([]string{command}, args...),
		done: done,
	}

	// Use the script's PATH to look up the command if it contains a separator
	// instead of the test process's PATH (see lookPath).
	// Don't use filepath.Clean, since that changes "./foo" to "foo".
	command = filepath.FromSlash(command)
	if !strings.Contains(command, string(filepath.Separator)) {
		var err error
		command, err = ts.lookPath(command)
		if err != nil {
			return nil, err
		}
	}
	cmd := exec.Command(command, args...)
	cmd.Dir = ts.cd
	cmd.Env = append(ts.env, "PWD="+ts.cd)
	cmd.Stdout = &bg.stdout
	cmd.Stderr = &bg.stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}

	go func() {
		bg.err = waitOrStop(ts.ctx, cmd, quitSignal(), ts.gracePeriod)
		close(done)
	}()
	return bg, nil
}

// lookPath is (roughly) like exec.LookPath, but it uses the test script's PATH
// instead of the test process's PATH to find the executable. We don't change
// the test process's PATH since it may run scripts in parallel.
func (ts *testScript) lookPath(command string) (string, error) {
	var strEqual func(string, string) bool
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" {
		// Using GOOS as a proxy for case-insensitive file system.
		strEqual = strings.EqualFold
	} else {
		strEqual = func(a, b string) bool { return a == b }
	}

	var pathExt []string
	var searchExt bool
	var isExecutable func(os.FileInfo) bool
	if runtime.GOOS == "windows" {
		// Use the test process's PathExt instead of the script's.
		// If PathExt is set in the command's environment, cmd.Start fails with
		// "parameter is invalid". Not sure why.
		// If the command already has an extension in PathExt (like "cmd.exe")
		// don't search for other extensions (not "cmd.bat.exe").
		pathExt = strings.Split(os.Getenv("PathExt"), string(filepath.ListSeparator))
		searchExt = true
		cmdExt := filepath.Ext(command)
		for _, ext := range pathExt {
			if strEqual(cmdExt, ext) {
				searchExt = false
				break
			}
		}
		isExecutable = func(fi os.FileInfo) bool {
			return fi.Mode().IsRegular()
		}
	} else {
		isExecutable = func(fi os.FileInfo) bool {
			return fi.Mode().IsRegular() && fi.Mode().Perm()&0111 != 0
		}
	}

	for _, dir := range strings.Split(ts.envMap[pathEnvName()], string(filepath.ListSeparator)) {
		if searchExt {
			ents, err := os.ReadDir(dir)
			if err != nil {
				continue
			}
			for _, ent := range ents {
				for _, ext := range pathExt {
					if !ent.IsDir() && strEqual(ent.Name(), command+ext) {
						return dir + string(filepath.Separator) + ent.Name(), nil
					}
				}
			}
		} else {
			path := dir + string(filepath.Separator) + command
			if fi, err := os.Stat(path); err == nil && isExecutable(fi) {
				return path, nil
			}
		}
	}
	return "", &exec.Error{Name: command, Err: exec.ErrNotFound}
}

// waitOrStop waits for the already-started command cmd by calling its Wait method.
//
// If cmd does not return before ctx is done, waitOrStop sends it the given interrupt signal.
// If killDelay is positive, waitOrStop waits that additional period for Wait to return before sending os.Kill.
//
// This function is copied from the one added to x/playground/internal in
// http://golang.org/cl/228438.
func waitOrStop(ctx context.Context, cmd *exec.Cmd, interrupt os.Signal, killDelay time.Duration) error {
	if cmd.Process == nil {
		panic("waitOrStop called with a nil cmd.Process — missing Start call?")
	}
	if interrupt == nil {
		panic("waitOrStop requires a non-nil interrupt signal")
	}

	errc := make(chan error)
	go func() {
		select {
		case errc <- nil:
			return
		case <-ctx.Done():
		}

		err := cmd.Process.Signal(interrupt)
		if err == nil {
			err = ctx.Err() // Report ctx.Err() as the reason we interrupted.
		} else if err == os.ErrProcessDone {
			errc <- nil
			return
		}

		if killDelay > 0 {
			timer := time.NewTimer(killDelay)
			select {
			// Report ctx.Err() as the reason we interrupted the process...
			case errc <- ctx.Err():
				timer.Stop()
				return
			// ...but after killDelay has elapsed, fall back to a stronger signal.
			case <-timer.C:
			}

			// Wait still hasn't returned.
			// Kill the process harder to make sure that it exits.
			//
			// Ignore any error: if cmd.Process has already terminated, we still
			// want to send ctx.Err() (or the error from the Interrupt call)
			// to properly attribute the signal that may have terminated it.
			_ = cmd.Process.Kill()
		}

		errc <- err
	}()

	waitErr := cmd.Wait()
	if interruptErr := <-errc; interruptErr != nil {
		return interruptErr
	}
	return waitErr
}

// expand applies environment variable expansion to the string s.
func (ts *testScript) expand(s string, inRegexp bool) string {
	return os.Expand(s, func(key string) string {
		e := ts.envMap[key]
		if inRegexp {
			// Replace workdir with $WORK, since we have done the same substitution in
			// the text we're about to compare against.
			e = strings.ReplaceAll(e, ts.workdir, "$WORK")

			// Quote to literal strings: we want paths like C:\work\go1.4 to remain
			// paths rather than regular expressions.
			e = regexp.QuoteMeta(e)
		}
		return e
	})
}

// fatalf aborts the test with the given failure message.
func (ts *testScript) fatalf(format string, args ...any) {
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

// A condition guards execution of a command.
type condition struct {
	want bool
	tag  string
}

// A command is a complete command parsed from a script.
type command struct {
	want  simpleStatus
	conds []condition // all must be satisfied
	name  string      // the name of the command; must be non-empty
	args  []string    // shell-expanded arguments following name
}

// parse parses a single line as a list of space-separated arguments
// subject to environment variable expansion (but not resplitting).
// Single quotes around text disable splitting and expansion.
// To embed a single quote, double it:
//
//	'Don''t communicate by sharing memory.'
func (ts *testScript) parse(line string) command {
	ts.line = line

	var (
		cmd      command
		arg      string  // text of current arg so far (need to add line[start:i])
		start    = -1    // if >= 0, position where current arg text chunk starts
		quoted   = false // currently processing quoted text
		isRegexp = false // currently processing unquoted regular expression
	)

	flushArg := func() {
		defer func() {
			arg = ""
			start = -1
		}()

		if cmd.name != "" {
			cmd.args = append(cmd.args, arg)
			// Commands take only one regexp argument (after the optional flags),
			// so no subsequent args are regexps. Liberally assume an argument that
			// starts with a '-' is a flag.
			if len(arg) == 0 || arg[0] != '-' {
				isRegexp = false
			}
			return
		}

		// Command prefix ! means negate the expectations about this command:
		// go command should fail, match should not be found, etc.
		// Prefix ? means allow either success or failure.
		switch want := simpleStatus(arg); want {
		case failure, successOrFailure:
			if cmd.want != "" {
				ts.fatalf("duplicated '!' or '?' token")
			}
			cmd.want = want
			return
		}

		// Command prefix [cond] means only run this command if cond is satisfied.
		if strings.HasPrefix(arg, "[") && strings.HasSuffix(arg, "]") {
			want := true
			arg = strings.TrimSpace(arg[1 : len(arg)-1])
			if strings.HasPrefix(arg, "!") {
				want = false
				arg = strings.TrimSpace(arg[1:])
			}
			if arg == "" {
				ts.fatalf("empty condition")
			}
			cmd.conds = append(cmd.conds, condition{want: want, tag: arg})
			return
		}

		cmd.name = arg
		isRegexp = regexpCmd[cmd.name]
	}

	for i := 0; ; i++ {
		if !quoted && (i >= len(line) || line[i] == ' ' || line[i] == '\t' || line[i] == '\r' || line[i] == '#') {
			// Found arg-separating space.
			if start >= 0 {
				arg += ts.expand(line[start:i], isRegexp)
				flushArg()
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
					arg += ts.expand(line[start:i], isRegexp)
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
	return cmd
}

// updateSum runs 'go mod tidy', 'go list -mod=mod -m all', or
// 'go list -mod=mod all' in the test's current directory if a file named
// "go.mod" is present after the archive has been extracted. updateSum modifies
// archive and returns true if go.mod or go.sum were changed.
func (ts *testScript) updateSum(archive *txtar.Archive) (rewrite bool) {
	gomodIdx, gosumIdx := -1, -1
	for i := range archive.Files {
		switch archive.Files[i].Name {
		case "go.mod":
			gomodIdx = i
		case "go.sum":
			gosumIdx = i
		}
	}
	if gomodIdx < 0 {
		return false
	}

	switch *testSum {
	case "tidy":
		ts.cmdGo(success, []string{"mod", "tidy"})
	case "listm":
		ts.cmdGo(success, []string{"list", "-m", "-mod=mod", "all"})
	case "listall":
		ts.cmdGo(success, []string{"list", "-mod=mod", "all"})
	default:
		ts.t.Fatalf(`unknown value for -testsum %q; may be "tidy", "listm", or "listall"`, *testSum)
	}

	newGomodData, err := os.ReadFile(filepath.Join(ts.cd, "go.mod"))
	if err != nil {
		ts.t.Fatalf("reading go.mod after -testsum: %v", err)
	}
	if !bytes.Equal(newGomodData, archive.Files[gomodIdx].Data) {
		archive.Files[gomodIdx].Data = newGomodData
		rewrite = true
	}

	newGosumData, err := os.ReadFile(filepath.Join(ts.cd, "go.sum"))
	if err != nil && !os.IsNotExist(err) {
		ts.t.Fatalf("reading go.sum after -testsum: %v", err)
	}
	switch {
	case os.IsNotExist(err) && gosumIdx >= 0:
		// go.sum was deleted.
		rewrite = true
		archive.Files = append(archive.Files[:gosumIdx], archive.Files[gosumIdx+1:]...)
	case err == nil && gosumIdx < 0:
		// go.sum was created.
		rewrite = true
		gosumIdx = gomodIdx + 1
		archive.Files = append(archive.Files, txtar.File{})
		copy(archive.Files[gosumIdx+1:], archive.Files[gosumIdx:])
		archive.Files[gosumIdx] = txtar.File{Name: "go.sum", Data: newGosumData}
	case err == nil && gosumIdx >= 0 && !bytes.Equal(newGosumData, archive.Files[gosumIdx].Data):
		// go.sum was changed.
		rewrite = true
		archive.Files[gosumIdx].Data = newGosumData
	}
	return rewrite
}

// diff returns a formatted diff of the two texts,
// showing the entire text and the minimum line-level
// additions and removals to turn text1 into text2.
// (That is, lines only in text1 appear with a leading -,
// and lines only in text2 appear with a leading +.)
func diff(text1, text2 string) string {
	if text1 != "" && !strings.HasSuffix(text1, "\n") {
		text1 += "(missing final newline)"
	}
	lines1 := strings.Split(text1, "\n")
	lines1 = lines1[:len(lines1)-1] // remove empty string after final line
	if text2 != "" && !strings.HasSuffix(text2, "\n") {
		text2 += "(missing final newline)"
	}
	lines2 := strings.Split(text2, "\n")
	lines2 = lines2[:len(lines2)-1] // remove empty string after final line

	// Naive dynamic programming algorithm for edit distance.
	// https://en.wikipedia.org/wiki/Wagner–Fischer_algorithm
	// dist[i][j] = edit distance between lines1[:len(lines1)-i] and lines2[:len(lines2)-j]
	// (The reversed indices make following the minimum cost path
	// visit lines in the same order as in the text.)
	dist := make([][]int, len(lines1)+1)
	for i := range dist {
		dist[i] = make([]int, len(lines2)+1)
		if i == 0 {
			for j := range dist[0] {
				dist[0][j] = j
			}
			continue
		}
		for j := range dist[i] {
			if j == 0 {
				dist[i][0] = i
				continue
			}
			cost := dist[i][j-1] + 1
			if cost > dist[i-1][j]+1 {
				cost = dist[i-1][j] + 1
			}
			if lines1[len(lines1)-i] == lines2[len(lines2)-j] {
				if cost > dist[i-1][j-1] {
					cost = dist[i-1][j-1]
				}
			}
			dist[i][j] = cost
		}
	}

	var buf strings.Builder
	i, j := len(lines1), len(lines2)
	for i > 0 || j > 0 {
		cost := dist[i][j]
		if i > 0 && j > 0 && cost == dist[i-1][j-1] && lines1[len(lines1)-i] == lines2[len(lines2)-j] {
			fmt.Fprintf(&buf, " %s\n", lines1[len(lines1)-i])
			i--
			j--
		} else if i > 0 && cost == dist[i-1][j]+1 {
			fmt.Fprintf(&buf, "-%s\n", lines1[len(lines1)-i])
			i--
		} else {
			fmt.Fprintf(&buf, "+%s\n", lines2[len(lines2)-j])
			j--
		}
	}
	return buf.String()
}

var diffTests = []struct {
	text1 string
	text2 string
	diff  string
}{
	{"a b c", "a b d e f", "a b -c +d +e +f"},
	{"", "a b c", "+a +b +c"},
	{"a b c", "", "-a -b -c"},
	{"a b c", "d e f", "-a -b -c +d +e +f"},
	{"a b c d e f", "a b d e f", "a b -c d e f"},
	{"a b c e f", "a b c d e f", "a b c +d e f"},
}

func TestDiff(t *testing.T) {
	t.Parallel()

	for _, tt := range diffTests {
		// Turn spaces into \n.
		text1 := strings.ReplaceAll(tt.text1, " ", "\n")
		if text1 != "" {
			text1 += "\n"
		}
		text2 := strings.ReplaceAll(tt.text2, " ", "\n")
		if text2 != "" {
			text2 += "\n"
		}
		out := diff(text1, text2)
		// Cut final \n, cut spaces, turn remaining \n into spaces.
		out = strings.ReplaceAll(strings.ReplaceAll(strings.TrimSuffix(out, "\n"), " ", ""), "\n", " ")
		if out != tt.diff {
			t.Errorf("diff(%q, %q) = %q, want %q", text1, text2, out, tt.diff)
		}
	}
}
