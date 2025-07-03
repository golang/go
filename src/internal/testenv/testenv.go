// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testenv provides information about what functionality
// is available in different testing environments run by the Go team.
//
// It is an internal package because these details are specific
// to the Go team's test setup (on build.golang.org) and not
// fundamental to tests in general.
package testenv

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"internal/cfg"
	"internal/goarch"
	"internal/platform"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Save the original environment during init for use in checks. A test
// binary may modify its environment before calling HasExec to change its
// behavior (such as mimicking a command-line tool), and that modified
// environment might cause environment checks to behave erratically.
var origEnv = os.Environ()

// Builder reports the name of the builder running this test
// (for example, "linux-amd64" or "windows-386-gce").
// If the test is not running on the build infrastructure,
// Builder returns the empty string.
func Builder() string {
	return os.Getenv("GO_BUILDER_NAME")
}

// HasGoBuild reports whether the current system can build programs with “go build”
// and then run them with os.StartProcess or exec.Command.
func HasGoBuild() bool {
	if os.Getenv("GO_GCFLAGS") != "" {
		// It's too much work to require every caller of the go command
		// to pass along "-gcflags="+os.Getenv("GO_GCFLAGS").
		// For now, if $GO_GCFLAGS is set, report that we simply can't
		// run go build.
		return false
	}

	return tryGoBuild() == nil
}

var tryGoBuild = sync.OnceValue(func() error {
	// To run 'go build', we need to be able to exec a 'go' command.
	// We somewhat arbitrarily choose to exec 'go tool -n compile' because that
	// also confirms that cmd/go can find the compiler. (Before CL 472096,
	// we sometimes ended up with cmd/go installed in the test environment
	// without a cmd/compile it could use to actually build things.)
	goTool, err := goTool()
	if err != nil {
		return err
	}
	cmd := exec.Command(goTool, "tool", "-n", "compile")
	cmd.Env = origEnv
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("%v: %w", cmd, err)
	}
	out = bytes.TrimSpace(out)
	if len(out) == 0 {
		return fmt.Errorf("%v: no tool reported", cmd)
	}
	if _, err := exec.LookPath(string(out)); err != nil {
		return err
	}

	if platform.MustLinkExternal(runtime.GOOS, runtime.GOARCH, false) {
		// We can assume that we always have a complete Go toolchain available.
		// However, this platform requires a C linker to build even pure Go
		// programs, including tests. Do we have one in the test environment?
		// (On Android, for example, the device running the test might not have a
		// C toolchain installed.)
		//
		// If CC is set explicitly, assume that we do. Otherwise, use 'go env CC'
		// to determine which toolchain it would use by default.
		if os.Getenv("CC") == "" {
			cmd := exec.Command(goTool, "env", "CC")
			cmd.Env = origEnv
			out, err := cmd.Output()
			if err != nil {
				return fmt.Errorf("%v: %w", cmd, err)
			}
			out = bytes.TrimSpace(out)
			if len(out) == 0 {
				return fmt.Errorf("%v: no CC reported", cmd)
			}
			_, err = exec.LookPath(string(out))
			return err
		}
	}
	return nil
})

// MustHaveGoBuild checks that the current system can build programs with “go build”
// and then run them with os.StartProcess or exec.Command.
// If not, MustHaveGoBuild calls t.Skip with an explanation.
func MustHaveGoBuild(t testing.TB) {
	if os.Getenv("GO_GCFLAGS") != "" {
		t.Helper()
		t.Skipf("skipping test: 'go build' not compatible with setting $GO_GCFLAGS")
	}
	if !HasGoBuild() {
		t.Helper()
		t.Skipf("skipping test: 'go build' unavailable: %v", tryGoBuild())
	}
}

// HasGoRun reports whether the current system can run programs with “go run”.
func HasGoRun() bool {
	// For now, having go run and having go build are the same.
	return HasGoBuild()
}

// MustHaveGoRun checks that the current system can run programs with “go run”.
// If not, MustHaveGoRun calls t.Skip with an explanation.
func MustHaveGoRun(t testing.TB) {
	if !HasGoRun() {
		t.Helper()
		t.Skipf("skipping test: 'go run' not available on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

// HasParallelism reports whether the current system can execute multiple
// threads in parallel.
// There is a copy of this function in cmd/dist/test.go.
func HasParallelism() bool {
	switch runtime.GOOS {
	case "js", "wasip1":
		return false
	}
	return true
}

// MustHaveParallelism checks that the current system can execute multiple
// threads in parallel. If not, MustHaveParallelism calls t.Skip with an explanation.
func MustHaveParallelism(t testing.TB) {
	if !HasParallelism() {
		t.Helper()
		t.Skipf("skipping test: no parallelism available on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

// GoToolPath reports the path to the Go tool.
// It is a convenience wrapper around GoTool.
// If the tool is unavailable GoToolPath calls t.Skip.
// If the tool should be available and isn't, GoToolPath calls t.Fatal.
func GoToolPath(t testing.TB) string {
	MustHaveGoBuild(t)
	path, err := GoTool()
	if err != nil {
		t.Fatal(err)
	}
	// Add all environment variables that affect the Go command to test metadata.
	// Cached test results will be invalidate when these variables change.
	// See golang.org/issue/32285.
	for _, envVar := range strings.Fields(cfg.KnownEnv) {
		os.Getenv(envVar)
	}
	return path
}

var findGOROOT = sync.OnceValues(func() (path string, err error) {
	if path := runtime.GOROOT(); path != "" {
		// If runtime.GOROOT() is non-empty, assume that it is valid.
		//
		// (It might not be: for example, the user may have explicitly set GOROOT
		// to the wrong directory. But this case is
		// rare, and if that happens the user can fix what they broke.)
		return path, nil
	}

	// runtime.GOROOT doesn't know where GOROOT is (perhaps because the test
	// binary was built with -trimpath).
	//
	// Since this is internal/testenv, we can cheat and assume that the caller
	// is a test of some package in a subdirectory of GOROOT/src. ('go test'
	// runs the test in the directory containing the packaged under test.) That
	// means that if we start walking up the tree, we should eventually find
	// GOROOT/src/go.mod, and we can report the parent directory of that.
	//
	// Notably, this works even if we can't run 'go env GOROOT' as a
	// subprocess.

	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("finding GOROOT: %w", err)
	}

	dir := cwd
	for {
		parent := filepath.Dir(dir)
		if parent == dir {
			// dir is either "." or only a volume name.
			return "", fmt.Errorf("failed to locate GOROOT/src in any parent directory")
		}

		if base := filepath.Base(dir); base != "src" {
			dir = parent
			continue // dir cannot be GOROOT/src if it doesn't end in "src".
		}

		b, err := os.ReadFile(filepath.Join(dir, "go.mod"))
		if err != nil {
			if os.IsNotExist(err) {
				dir = parent
				continue
			}
			return "", fmt.Errorf("finding GOROOT: %w", err)
		}
		goMod := string(b)

		for goMod != "" {
			var line string
			line, goMod, _ = strings.Cut(goMod, "\n")
			fields := strings.Fields(line)
			if len(fields) >= 2 && fields[0] == "module" && fields[1] == "std" {
				// Found "module std", which is the module declaration in GOROOT/src!
				return parent, nil
			}
		}
	}
})

// GOROOT reports the path to the directory containing the root of the Go
// project source tree. This is normally equivalent to runtime.GOROOT, but
// works even if the test binary was built with -trimpath and cannot exec
// 'go env GOROOT'.
//
// If GOROOT cannot be found, GOROOT skips t if t is non-nil,
// or panics otherwise.
func GOROOT(t testing.TB) string {
	path, err := findGOROOT()
	if err != nil {
		if t == nil {
			panic(err)
		}
		t.Helper()
		t.Skip(err)
	}
	return path
}

// GoTool reports the path to the Go tool.
func GoTool() (string, error) {
	if !HasGoBuild() {
		return "", errors.New("platform cannot run go tool")
	}
	return goTool()
}

var goTool = sync.OnceValues(func() (string, error) {
	return exec.LookPath("go")
})

// MustHaveSource checks that the entire source tree is available under GOROOT.
// If not, it calls t.Skip with an explanation.
func MustHaveSource(t testing.TB) {
	switch runtime.GOOS {
	case "ios":
		t.Helper()
		t.Skip("skipping test: no source tree on " + runtime.GOOS)
	}
}

// HasExternalNetwork reports whether the current system can use
// external (non-localhost) networks.
func HasExternalNetwork() bool {
	return !testing.Short() && runtime.GOOS != "js" && runtime.GOOS != "wasip1"
}

// MustHaveExternalNetwork checks that the current system can use
// external (non-localhost) networks.
// If not, MustHaveExternalNetwork calls t.Skip with an explanation.
func MustHaveExternalNetwork(t testing.TB) {
	if runtime.GOOS == "js" || runtime.GOOS == "wasip1" {
		t.Helper()
		t.Skipf("skipping test: no external network on %s", runtime.GOOS)
	}
	if testing.Short() {
		t.Helper()
		t.Skipf("skipping test: no external network in -short mode")
	}
}

// HasCGO reports whether the current system can use cgo.
func HasCGO() bool {
	return hasCgo()
}

var hasCgo = sync.OnceValue(func() bool {
	goTool, err := goTool()
	if err != nil {
		return false
	}
	cmd := exec.Command(goTool, "env", "CGO_ENABLED")
	cmd.Env = origEnv
	out, err := cmd.Output()
	if err != nil {
		panic(fmt.Sprintf("%v: %v", cmd, out))
	}
	ok, err := strconv.ParseBool(string(bytes.TrimSpace(out)))
	if err != nil {
		panic(fmt.Sprintf("%v: non-boolean output %q", cmd, out))
	}
	return ok
})

// MustHaveCGO calls t.Skip if cgo is not available.
func MustHaveCGO(t testing.TB) {
	if !HasCGO() {
		t.Helper()
		t.Skipf("skipping test: no cgo")
	}
}

// CanInternalLink reports whether the current system can link programs with
// internal linking.
func CanInternalLink(withCgo bool) bool {
	return !platform.MustLinkExternal(runtime.GOOS, runtime.GOARCH, withCgo)
}

// SpecialBuildTypes are interesting build types that may affect linking.
type SpecialBuildTypes struct {
	Cgo  bool
	Asan bool
	Msan bool
	Race bool
}

// NoSpecialBuildTypes indicates a standard, no cgo go build.
var NoSpecialBuildTypes SpecialBuildTypes

// MustInternalLink checks that the current system can link programs with internal
// linking.
// If not, MustInternalLink calls t.Skip with an explanation.
func MustInternalLink(t testing.TB, with SpecialBuildTypes) {
	if with.Asan || with.Msan || with.Race {
		t.Skipf("skipping test: internal linking with sanitizers is not supported")
	}
	if !CanInternalLink(with.Cgo) {
		t.Helper()
		if with.Cgo && CanInternalLink(false) {
			t.Skipf("skipping test: internal linking on %s/%s is not supported with cgo", runtime.GOOS, runtime.GOARCH)
		}
		t.Skipf("skipping test: internal linking on %s/%s is not supported", runtime.GOOS, runtime.GOARCH)
	}
}

// MustInternalLinkPIE checks whether the current system can link PIE binary using
// internal linking.
// If not, MustInternalLinkPIE calls t.Skip with an explanation.
func MustInternalLinkPIE(t testing.TB) {
	if !platform.InternalLinkPIESupported(runtime.GOOS, runtime.GOARCH) {
		t.Helper()
		t.Skipf("skipping test: internal linking for buildmode=pie on %s/%s is not supported", runtime.GOOS, runtime.GOARCH)
	}
}

// MustHaveBuildMode reports whether the current system can build programs in
// the given build mode.
// If not, MustHaveBuildMode calls t.Skip with an explanation.
func MustHaveBuildMode(t testing.TB, buildmode string) {
	if !platform.BuildModeSupported(runtime.Compiler, buildmode, runtime.GOOS, runtime.GOARCH) {
		t.Helper()
		t.Skipf("skipping test: build mode %s on %s/%s is not supported by the %s compiler", buildmode, runtime.GOOS, runtime.GOARCH, runtime.Compiler)
	}
}

// HasSymlink reports whether the current system can use os.Symlink.
func HasSymlink() bool {
	ok, _ := hasSymlink()
	return ok
}

// MustHaveSymlink reports whether the current system can use os.Symlink.
// If not, MustHaveSymlink calls t.Skip with an explanation.
func MustHaveSymlink(t testing.TB) {
	ok, reason := hasSymlink()
	if !ok {
		t.Helper()
		t.Skipf("skipping test: cannot make symlinks on %s/%s: %s", runtime.GOOS, runtime.GOARCH, reason)
	}
}

// HasLink reports whether the current system can use os.Link.
func HasLink() bool {
	// From Android release M (Marshmallow), hard linking files is blocked
	// and an attempt to call link() on a file will return EACCES.
	// - https://code.google.com/p/android-developer-preview/issues/detail?id=3150
	return runtime.GOOS != "plan9" && runtime.GOOS != "android"
}

// MustHaveLink reports whether the current system can use os.Link.
// If not, MustHaveLink calls t.Skip with an explanation.
func MustHaveLink(t testing.TB) {
	if !HasLink() {
		t.Helper()
		t.Skipf("skipping test: hardlinks are not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

var flaky = flag.Bool("flaky", false, "run known-flaky tests too")

func SkipFlaky(t testing.TB, issue int) {
	if !*flaky {
		t.Helper()
		t.Skipf("skipping known flaky test without the -flaky flag; see golang.org/issue/%d", issue)
	}
}

func SkipFlakyNet(t testing.TB) {
	if v, _ := strconv.ParseBool(os.Getenv("GO_BUILDER_FLAKY_NET")); v {
		t.Helper()
		t.Skip("skipping test on builder known to have frequent network failures")
	}
}

// CPUIsSlow reports whether the CPU running the test is suspected to be slow.
func CPUIsSlow() bool {
	switch runtime.GOARCH {
	case "arm", "mips", "mipsle", "mips64", "mips64le", "wasm":
		return true
	}
	return false
}

// SkipIfShortAndSlow skips t if -short is set and the CPU running the test is
// suspected to be slow.
//
// (This is useful for CPU-intensive tests that otherwise complete quickly.)
func SkipIfShortAndSlow(t testing.TB) {
	if testing.Short() && CPUIsSlow() {
		t.Helper()
		t.Skipf("skipping test in -short mode on %s", runtime.GOARCH)
	}
}

// SkipIfOptimizationOff skips t if optimization is disabled.
func SkipIfOptimizationOff(t testing.TB) {
	if OptimizationOff() {
		t.Helper()
		t.Skip("skipping test with optimization disabled")
	}
}

// WriteImportcfg writes an importcfg file used by the compiler or linker to
// dstPath containing entries for the file mappings in packageFiles, as well
// as for the packages transitively imported by the package(s) in pkgs.
//
// pkgs may include any package pattern that is valid to pass to 'go list',
// so it may also be a list of Go source files all in the same directory.
func WriteImportcfg(t testing.TB, dstPath string, packageFiles map[string]string, pkgs ...string) {
	t.Helper()

	icfg := new(bytes.Buffer)
	icfg.WriteString("# import config\n")
	for k, v := range packageFiles {
		fmt.Fprintf(icfg, "packagefile %s=%s\n", k, v)
	}

	if len(pkgs) > 0 {
		// Use 'go list' to resolve any missing packages and rewrite the import map.
		cmd := Command(t, GoToolPath(t), "list", "-export", "-deps", "-f", `{{if ne .ImportPath "command-line-arguments"}}{{if .Export}}{{.ImportPath}}={{.Export}}{{end}}{{end}}`)
		cmd.Args = append(cmd.Args, pkgs...)
		cmd.Stderr = new(strings.Builder)
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("%v: %v\n%s", cmd, err, cmd.Stderr)
		}

		for _, line := range strings.Split(string(out), "\n") {
			if line == "" {
				continue
			}
			importPath, export, ok := strings.Cut(line, "=")
			if !ok {
				t.Fatalf("invalid line in output from %v:\n%s", cmd, line)
			}
			if packageFiles[importPath] == "" {
				fmt.Fprintf(icfg, "packagefile %s=%s\n", importPath, export)
			}
		}
	}

	if err := os.WriteFile(dstPath, icfg.Bytes(), 0666); err != nil {
		t.Fatal(err)
	}
}

// SyscallIsNotSupported reports whether err may indicate that a system call is
// not supported by the current platform or execution environment.
func SyscallIsNotSupported(err error) bool {
	return syscallIsNotSupported(err)
}

// ParallelOn64Bit calls t.Parallel() unless there is a case that cannot be parallel.
// This function should be used when it is necessary to avoid t.Parallel on
// 32-bit machines, typically because the test uses lots of memory.
func ParallelOn64Bit(t *testing.T) {
	if goarch.PtrSize == 4 {
		return
	}
	t.Parallel()
}

// CPUProfilingBroken returns true if CPU profiling has known issues on this
// platform.
func CPUProfilingBroken() bool {
	switch runtime.GOOS {
	case "plan9":
		// Profiling unimplemented.
		return true
	case "aix":
		// See https://golang.org/issue/45170.
		return true
	case "ios", "dragonfly", "netbsd", "illumos", "solaris":
		// See https://golang.org/issue/13841.
		return true
	case "openbsd":
		if runtime.GOARCH == "arm" || runtime.GOARCH == "arm64" {
			// See https://golang.org/issue/13841.
			return true
		}
	}

	return false
}
