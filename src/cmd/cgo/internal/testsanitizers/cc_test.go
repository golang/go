// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test uses the Pdeathsig field of syscall.SysProcAttr, so it only works
// on platforms that support that.

//go:build linux || (freebsd && amd64)

// sanitizers_test checks the use of Go with sanitizers like msan, asan, etc.
// See https://github.com/google/sanitizers.
package sanitizers_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
	"unicode"
)

var overcommit struct {
	sync.Once
	value int
	err   error
}

// requireOvercommit skips t if the kernel does not allow overcommit.
func requireOvercommit(t *testing.T) {
	t.Helper()

	overcommit.Once.Do(func() {
		var out []byte
		out, overcommit.err = os.ReadFile("/proc/sys/vm/overcommit_memory")
		if overcommit.err != nil {
			return
		}
		overcommit.value, overcommit.err = strconv.Atoi(string(bytes.TrimSpace(out)))
	})

	if overcommit.err != nil {
		t.Skipf("couldn't determine vm.overcommit_memory (%v); assuming no overcommit", overcommit.err)
	}
	if overcommit.value == 2 {
		t.Skip("vm.overcommit_memory=2")
	}
}

var env struct {
	sync.Once
	m   map[string]string
	err error
}

// goEnv returns the output of $(go env) as a map.
func goEnv(key string) (string, error) {
	env.Once.Do(func() {
		var out []byte
		out, env.err = exec.Command("go", "env", "-json").Output()
		if env.err != nil {
			return
		}

		env.m = make(map[string]string)
		env.err = json.Unmarshal(out, &env.m)
	})
	if env.err != nil {
		return "", env.err
	}

	v, ok := env.m[key]
	if !ok {
		return "", fmt.Errorf("`go env`: no entry for %v", key)
	}
	return v, nil
}

// replaceEnv sets the key environment variable to value in cmd.
func replaceEnv(cmd *exec.Cmd, key, value string) {
	if cmd.Env == nil {
		cmd.Env = cmd.Environ()
	}
	cmd.Env = append(cmd.Env, key+"="+value)
}

// appendExperimentEnv appends comma-separated experiments to GOEXPERIMENT.
func appendExperimentEnv(cmd *exec.Cmd, experiments []string) {
	if cmd.Env == nil {
		cmd.Env = cmd.Environ()
	}
	exps := strings.Join(experiments, ",")
	for _, evar := range cmd.Env {
		c := strings.SplitN(evar, "=", 2)
		if c[0] == "GOEXPERIMENT" {
			exps = c[1] + "," + exps
		}
	}
	cmd.Env = append(cmd.Env, "GOEXPERIMENT="+exps)
}

// mustRun executes t and fails cmd with a well-formatted message if it fails.
func mustRun(t *testing.T, cmd *exec.Cmd) {
	t.Helper()
	out := new(strings.Builder)
	cmd.Stdout = out
	cmd.Stderr = out

	err := cmd.Start()
	if err != nil {
		t.Fatalf("%v: %v", cmd, err)
	}

	if deadline, ok := t.Deadline(); ok {
		timeout := time.Until(deadline)
		timeout -= timeout / 10 // Leave 10% headroom for logging and cleanup.
		timer := time.AfterFunc(timeout, func() {
			cmd.Process.Signal(syscall.SIGQUIT)
		})
		defer timer.Stop()
	}

	if err := cmd.Wait(); err != nil {
		t.Fatalf("%v exited with %v\n%s", cmd, err, out)
	}
}

// cc returns a cmd that executes `$(go env CC) $(go env GOGCCFLAGS) $args`.
func cc(args ...string) (*exec.Cmd, error) {
	CC, err := goEnv("CC")
	if err != nil {
		return nil, err
	}

	GOGCCFLAGS, err := goEnv("GOGCCFLAGS")
	if err != nil {
		return nil, err
	}

	// Split GOGCCFLAGS, respecting quoting.
	//
	// TODO(bcmills): This code also appears in
	// cmd/cgo/internal/testcarchive/carchive_test.go, and perhaps ought to go in
	// src/cmd/dist/test.go as well. Figure out where to put it so that it can be
	// shared.
	var flags []string
	quote := '\000'
	start := 0
	lastSpace := true
	backslash := false
	for i, c := range GOGCCFLAGS {
		if quote == '\000' && unicode.IsSpace(c) {
			if !lastSpace {
				flags = append(flags, GOGCCFLAGS[start:i])
				lastSpace = true
			}
		} else {
			if lastSpace {
				start = i
				lastSpace = false
			}
			if quote == '\000' && !backslash && (c == '"' || c == '\'') {
				quote = c
				backslash = false
			} else if !backslash && quote == c {
				quote = '\000'
			} else if (quote == '\000' || quote == '"') && !backslash && c == '\\' {
				backslash = true
			} else {
				backslash = false
			}
		}
	}
	if !lastSpace {
		flags = append(flags, GOGCCFLAGS[start:])
	}

	cmd := exec.Command(CC, flags...)
	cmd.Args = append(cmd.Args, args...)
	return cmd, nil
}

type version struct {
	name         string
	major, minor int
}

var compiler struct {
	sync.Once
	version
	err error
}

// compilerVersion detects the version of $(go env CC).
//
// It returns a non-nil error if the compiler matches a known version schema but
// the version could not be parsed, or if $(go env CC) could not be determined.
func compilerVersion() (version, error) {
	compiler.Once.Do(func() {
		compiler.err = func() error {
			compiler.name = "unknown"

			cmd, err := cc("--version")
			if err != nil {
				return err
			}
			out, err := cmd.Output()
			if err != nil {
				// Compiler does not support "--version" flag: not Clang or GCC.
				return nil
			}

			var match [][]byte
			if bytes.HasPrefix(out, []byte("gcc")) {
				compiler.name = "gcc"
				cmd, err := cc("-dumpfullversion", "-dumpversion")
				if err != nil {
					return err
				}
				out, err := cmd.Output()
				if err != nil {
					// gcc, but does not support gcc's "-v" flag?!
					return err
				}
				gccRE := regexp.MustCompile(`(\d+)\.(\d+)`)
				match = gccRE.FindSubmatch(out)
			} else {
				clangRE := regexp.MustCompile(`clang version (\d+)\.(\d+)`)
				if match = clangRE.FindSubmatch(out); len(match) > 0 {
					compiler.name = "clang"
				}
			}

			if len(match) < 3 {
				return nil // "unknown"
			}
			if compiler.major, err = strconv.Atoi(string(match[1])); err != nil {
				return err
			}
			if compiler.minor, err = strconv.Atoi(string(match[2])); err != nil {
				return err
			}
			return nil
		}()
	})
	return compiler.version, compiler.err
}

// compilerSupportsLocation reports whether the compiler should be
// able to provide file/line information in backtraces.
func compilerSupportsLocation() bool {
	compiler, err := compilerVersion()
	if err != nil {
		return false
	}
	switch compiler.name {
	case "gcc":
		// TODO(72752): the asan runtime support library
		// (libasan.so.6) shipped with GCC 10 has problems digesting
		// version 5 DWARF produced by the Go toolchain. Disable
		// location checking if gcc is not sufficiently up to date in
		// this case.
		return compiler.major > 10
	case "clang":
		// TODO(65606): The clang toolchain on the LUCI builders is not built against
		// zlib, the ASAN runtime can't actually symbolize its own stack trace. Once
		// this is resolved, one way or another, switch this back to 'true'. We still
		// have coverage from the 'gcc' case above.
		if inLUCIBuild() {
			return false
		}
		return true
	default:
		return false
	}
}

// inLUCIBuild returns true if we're currently executing in a LUCI build.
func inLUCIBuild() bool {
	u, err := user.Current()
	if err != nil {
		return false
	}
	return testenv.Builder() != "" && u.Username == "swarming"
}

// compilerRequiredTsanVersion reports whether the compiler is the version required by Tsan.
// Only restrictions for ppc64le are known; otherwise return true.
func compilerRequiredTsanVersion(goos, goarch string) bool {
	compiler, err := compilerVersion()
	if err != nil {
		return false
	}
	if compiler.name == "gcc" && goarch == "ppc64le" {
		return compiler.major >= 9
	}
	return true
}

// compilerRequiredAsanVersion reports whether the compiler is the version required by Asan.
func compilerRequiredAsanVersion(goos, goarch string) bool {
	compiler, err := compilerVersion()
	if err != nil {
		return false
	}
	switch compiler.name {
	case "gcc":
		if goarch == "loong64" {
			return compiler.major >= 14
		}
		if goarch == "ppc64le" {
			return compiler.major >= 9
		}
		return compiler.major >= 7
	case "clang":
		if goarch == "loong64" {
			return compiler.major >= 16
		}
		return compiler.major >= 9
	default:
		return false
	}
}

// compilerRequiredLsanVersion reports whether the compiler is the
// version required by Lsan.
func compilerRequiredLsanVersion(goos, goarch string) bool {
	return compilerRequiredAsanVersion(goos, goarch)
}

type compilerCheck struct {
	once sync.Once
	err  error
	skip bool // If true, skip with err instead of failing with it.
}

type config struct {
	sanitizer string

	cFlags, ldFlags, goFlags []string

	sanitizerCheck, runtimeCheck compilerCheck
}

var configs struct {
	sync.Mutex
	m map[string]*config
}

// configure returns the configuration for the given sanitizer.
func configure(sanitizer string) *config {
	configs.Lock()
	defer configs.Unlock()
	if c, ok := configs.m[sanitizer]; ok {
		return c
	}

	sanitizerOpt := sanitizer
	// For the leak detector, we use "go build -asan",
	// which implies the address sanitizer.
	// We may want to adjust this someday.
	if sanitizer == "leak" {
		sanitizerOpt = "address"
	}

	c := &config{
		sanitizer: sanitizer,
		cFlags:    []string{"-fsanitize=" + sanitizerOpt},
		ldFlags:   []string{"-fsanitize=" + sanitizerOpt},
	}

	if testing.Verbose() {
		c.goFlags = append(c.goFlags, "-x")
	}

	switch sanitizer {
	case "memory":
		c.goFlags = append(c.goFlags, "-msan")

	case "thread":
		c.goFlags = append(c.goFlags, "--installsuffix=tsan")
		compiler, _ := compilerVersion()
		if compiler.name == "gcc" {
			c.cFlags = append(c.cFlags, "-fPIC")
			c.ldFlags = append(c.ldFlags, "-fPIC", "-static-libtsan")
		}

	case "address", "leak":
		c.goFlags = append(c.goFlags, "-asan")
		// Set the debug mode to print the C stack trace.
		c.cFlags = append(c.cFlags, "-g")

	case "fuzzer":
		c.goFlags = append(c.goFlags, "-tags=libfuzzer", "-gcflags=-d=libfuzzer")

	default:
		panic(fmt.Sprintf("unrecognized sanitizer: %q", sanitizer))
	}

	if configs.m == nil {
		configs.m = make(map[string]*config)
	}
	configs.m[sanitizer] = c
	return c
}

// goCmd returns a Cmd that executes "go $subcommand $args" with appropriate
// additional flags and environment.
func (c *config) goCmd(subcommand string, args ...string) *exec.Cmd {
	return c.goCmdWithExperiments(subcommand, args, nil)
}

// goCmdWithExperiments returns a Cmd that executes
// "GOEXPERIMENT=$experiments go $subcommand $args" with appropriate
// additional flags and CGO-related environment variables.
func (c *config) goCmdWithExperiments(subcommand string, args []string, experiments []string) *exec.Cmd {
	cmd := exec.Command("go", subcommand)
	cmd.Args = append(cmd.Args, c.goFlags...)
	cmd.Args = append(cmd.Args, args...)
	replaceEnv(cmd, "CGO_CFLAGS", strings.Join(c.cFlags, " "))
	replaceEnv(cmd, "CGO_LDFLAGS", strings.Join(c.ldFlags, " "))
	appendExperimentEnv(cmd, experiments)
	return cmd
}

// skipIfCSanitizerBroken skips t if the C compiler does not produce working
// binaries as configured.
func (c *config) skipIfCSanitizerBroken(t *testing.T) {
	check := &c.sanitizerCheck
	check.once.Do(func() {
		check.skip, check.err = c.checkCSanitizer()
	})
	if check.err != nil {
		t.Helper()
		if check.skip {
			t.Skip(check.err)
		}
		t.Fatal(check.err)
	}
}

var cMain = []byte(`
int main() {
	return 0;
}
`)

var cLibFuzzerInput = []byte(`
#include <stddef.h>
int LLVMFuzzerTestOneInput(char *data, size_t size) {
	return 0;
}
`)

func (c *config) checkCSanitizer() (skip bool, err error) {
	dir, err := os.MkdirTemp("", c.sanitizer)
	if err != nil {
		return false, fmt.Errorf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, "return0.c")
	cInput := cMain
	if c.sanitizer == "fuzzer" {
		// libFuzzer generates the main function itself, and uses a different input.
		cInput = cLibFuzzerInput
	}
	if err := os.WriteFile(src, cInput, 0600); err != nil {
		return false, fmt.Errorf("failed to write C source file: %v", err)
	}

	dst := filepath.Join(dir, "return0")
	cmd, err := cc(c.cFlags...)
	if err != nil {
		return false, err
	}
	cmd.Args = append(cmd.Args, c.ldFlags...)
	cmd.Args = append(cmd.Args, "-o", dst, src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if bytes.Contains(out, []byte("-fsanitize")) &&
			(bytes.Contains(out, []byte("unrecognized")) ||
				bytes.Contains(out, []byte("unsupported"))) {
			return true, errors.New(string(out))
		}
		return true, fmt.Errorf("%#q failed: %v\n%s", cmd, err, out)
	}

	if c.sanitizer == "fuzzer" {
		// For fuzzer, don't try running the test binary. It never finishes.
		return false, nil
	}

	if out, err := exec.Command(dst).CombinedOutput(); err != nil {
		if os.IsNotExist(err) {
			return true, fmt.Errorf("%#q failed to produce executable: %v", cmd, err)
		}
		snippet, _, _ := bytes.Cut(out, []byte("\n"))
		return true, fmt.Errorf("%#q generated broken executable: %v\n%s", cmd, err, snippet)
	}

	return false, nil
}

// skipIfRuntimeIncompatible skips t if the Go runtime is suspected not to work
// with cgo as configured.
func (c *config) skipIfRuntimeIncompatible(t *testing.T) {
	check := &c.runtimeCheck
	check.once.Do(func() {
		check.skip, check.err = c.checkRuntime()
	})
	if check.err != nil {
		t.Helper()
		if check.skip {
			t.Skip(check.err)
		}
		t.Fatal(check.err)
	}
}

func (c *config) checkRuntime() (skip bool, err error) {
	if c.sanitizer != "thread" {
		return false, nil
	}

	// libcgo.h sets CGO_TSAN if it detects TSAN support in the C compiler.
	// Dump the preprocessor defines to check that works.
	// (Sometimes it doesn't: see https://golang.org/issue/15983.)
	cmd, err := cc(c.cFlags...)
	if err != nil {
		return false, err
	}
	cmd.Args = append(cmd.Args, "-dM", "-E", "../../../../runtime/cgo/libcgo.h")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return false, fmt.Errorf("%#q exited with %v\n%s", cmd, err, out)
	}
	if !bytes.Contains(out, []byte("#define CGO_TSAN")) {
		return true, fmt.Errorf("%#q did not define CGO_TSAN", cmd)
	}
	return false, nil
}

// srcPath returns the path to the given file relative to this test's source tree.
func srcPath(path string) string {
	return "./testdata/" + path
}

// A tempDir manages a temporary directory within a test.
type tempDir struct {
	base string
}

func (d *tempDir) RemoveAll(t *testing.T) {
	t.Helper()
	if d.base == "" {
		return
	}
	if err := os.RemoveAll(d.base); err != nil {
		t.Fatalf("Failed to remove temp dir: %v", err)
	}
}

func (d *tempDir) Base() string {
	return d.base
}

func (d *tempDir) Join(name string) string {
	return filepath.Join(d.base, name)
}

func newTempDir(t *testing.T) *tempDir {
	return &tempDir{base: t.TempDir()}
}

// hangProneCmd returns an exec.Cmd for a command that is likely to hang.
//
// If one of these tests hangs, the caller is likely to kill the test process
// using SIGINT, which will be sent to all of the processes in the test's group.
// Unfortunately, TSAN in particular is prone to dropping signals, so the SIGINT
// may terminate the test binary but leave the subprocess running. hangProneCmd
// configures subprocess to receive SIGKILL instead to ensure that it won't
// leak.
func hangProneCmd(name string, arg ...string) *exec.Cmd {
	cmd := exec.Command(name, arg...)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Pdeathsig: syscall.SIGKILL,
	}
	return cmd
}
