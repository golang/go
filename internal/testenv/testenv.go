// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testenv contains helper functions for skipping tests
// based on which tools are present in the environment.
package testenv

import (
	"bytes"
	"fmt"
	"go/build"
	"io/ioutil"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/internal/goroot"

	exec "golang.org/x/sys/execabs"
)

// packageMainIsDevel reports whether the module containing package main
// is a development version (if module information is available).
func packageMainIsDevel() bool {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		// Most test binaries currently lack build info, but this should become more
		// permissive once https://golang.org/issue/33976 is fixed.
		return true
	}

	// Note: info.Main.Version describes the version of the module containing
	// package main, not the version of “the main module”.
	// See https://golang.org/issue/33975.
	return info.Main.Version == "(devel)"
}

var checkGoGoroot struct {
	once sync.Once
	err  error
}

func hasTool(tool string) error {
	if tool == "cgo" {
		enabled, err := cgoEnabled(false)
		if err != nil {
			return fmt.Errorf("checking cgo: %v", err)
		}
		if !enabled {
			return fmt.Errorf("cgo not enabled")
		}
		return nil
	}

	_, err := exec.LookPath(tool)
	if err != nil {
		return err
	}

	switch tool {
	case "patch":
		// check that the patch tools supports the -o argument
		temp, err := ioutil.TempFile("", "patch-test")
		if err != nil {
			return err
		}
		temp.Close()
		defer os.Remove(temp.Name())
		cmd := exec.Command(tool, "-o", temp.Name())
		if err := cmd.Run(); err != nil {
			return err
		}

	case "go":
		checkGoGoroot.once.Do(func() {
			// Ensure that the 'go' command found by exec.LookPath is from the correct
			// GOROOT. Otherwise, 'some/path/go test ./...' will test against some
			// version of the 'go' binary other than 'some/path/go', which is almost
			// certainly not what the user intended.
			out, err := exec.Command(tool, "env", "GOROOT").CombinedOutput()
			if err != nil {
				checkGoGoroot.err = err
				return
			}
			GOROOT := strings.TrimSpace(string(out))
			if GOROOT != runtime.GOROOT() {
				checkGoGoroot.err = fmt.Errorf("'go env GOROOT' does not match runtime.GOROOT:\n\tgo env: %s\n\tGOROOT: %s", GOROOT, runtime.GOROOT())
				return
			}

			// Also ensure that that GOROOT includes a compiler: 'go' commands
			// don't in general work without it, and some builders
			// (such as android-amd64-emu) seem to lack it in the test environment.
			cmd := exec.Command(tool, "tool", "-n", "compile")
			stderr := new(bytes.Buffer)
			stderr.Write([]byte("\n"))
			cmd.Stderr = stderr
			out, err = cmd.Output()
			if err != nil {
				checkGoGoroot.err = fmt.Errorf("%v: %v%s", cmd, err, stderr)
				return
			}
			if _, err := os.Stat(string(bytes.TrimSpace(out))); err != nil {
				checkGoGoroot.err = err
			}
		})
		if checkGoGoroot.err != nil {
			return checkGoGoroot.err
		}

	case "diff":
		// Check that diff is the GNU version, needed for the -u argument and
		// to report missing newlines at the end of files.
		out, err := exec.Command(tool, "-version").Output()
		if err != nil {
			return err
		}
		if !bytes.Contains(out, []byte("GNU diffutils")) {
			return fmt.Errorf("diff is not the GNU version")
		}
	}

	return nil
}

func cgoEnabled(bypassEnvironment bool) (bool, error) {
	cmd := exec.Command("go", "env", "CGO_ENABLED")
	if bypassEnvironment {
		cmd.Env = append(append([]string(nil), os.Environ()...), "CGO_ENABLED=")
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return false, err
	}
	enabled := strings.TrimSpace(string(out))
	return enabled == "1", nil
}

func allowMissingTool(tool string) bool {
	if runtime.GOOS == "android" {
		// Android builds generally run tests on a separate machine from the build,
		// so don't expect any external tools to be available.
		return true
	}

	switch tool {
	case "cgo":
		if strings.HasSuffix(os.Getenv("GO_BUILDER_NAME"), "-nocgo") {
			// Explicitly disabled on -nocgo builders.
			return true
		}
		if enabled, err := cgoEnabled(true); err == nil && !enabled {
			// No platform support.
			return true
		}
	case "go":
		if os.Getenv("GO_BUILDER_NAME") == "illumos-amd64-joyent" {
			// Work around a misconfigured builder (see https://golang.org/issue/33950).
			return true
		}
	case "diff":
		if os.Getenv("GO_BUILDER_NAME") != "" {
			return true
		}
	case "patch":
		if os.Getenv("GO_BUILDER_NAME") != "" {
			return true
		}
	}

	// If a developer is actively working on this test, we expect them to have all
	// of its dependencies installed. However, if it's just a dependency of some
	// other module (for example, being run via 'go test all'), we should be more
	// tolerant of unusual environments.
	return !packageMainIsDevel()
}

// NeedsTool skips t if the named tool is not present in the path.
// As a special case, "cgo" means "go" is present and can compile cgo programs.
func NeedsTool(t testing.TB, tool string) {
	err := hasTool(tool)
	if err == nil {
		return
	}

	t.Helper()
	if allowMissingTool(tool) {
		t.Skipf("skipping because %s tool not available: %v", tool, err)
	} else {
		t.Fatalf("%s tool not available: %v", tool, err)
	}
}

// NeedsGoPackages skips t if the go/packages driver (or 'go' tool) implied by
// the current process environment is not present in the path.
func NeedsGoPackages(t testing.TB) {
	t.Helper()

	tool := os.Getenv("GOPACKAGESDRIVER")
	switch tool {
	case "off":
		// "off" forces go/packages to use the go command.
		tool = "go"
	case "":
		if _, err := exec.LookPath("gopackagesdriver"); err == nil {
			tool = "gopackagesdriver"
		} else {
			tool = "go"
		}
	}

	NeedsTool(t, tool)
}

// NeedsGoPackagesEnv skips t if the go/packages driver (or 'go' tool) implied
// by env is not present in the path.
func NeedsGoPackagesEnv(t testing.TB, env []string) {
	t.Helper()

	for _, v := range env {
		if strings.HasPrefix(v, "GOPACKAGESDRIVER=") {
			tool := strings.TrimPrefix(v, "GOPACKAGESDRIVER=")
			if tool == "off" {
				NeedsTool(t, "go")
			} else {
				NeedsTool(t, tool)
			}
			return
		}
	}

	NeedsGoPackages(t)
}

// NeedsGoBuild skips t if the current system can't build programs with “go build”
// and then run them with os.StartProcess or exec.Command.
// Android doesn't have the userspace go build needs to run,
// and js/wasm doesn't support running subprocesses.
func NeedsGoBuild(t testing.TB) {
	t.Helper()

	// This logic was derived from internal/testing.HasGoBuild and
	// may need to be updated as that function evolves.

	NeedsTool(t, "go")

	switch runtime.GOOS {
	case "android", "js":
		t.Skipf("skipping test: %v can't build and run Go binaries", runtime.GOOS)
	}
}

// ExitIfSmallMachine emits a helpful diagnostic and calls os.Exit(0) if the
// current machine is a builder known to have scarce resources.
//
// It should be called from within a TestMain function.
func ExitIfSmallMachine() {
	switch b := os.Getenv("GO_BUILDER_NAME"); b {
	case "linux-arm-scaleway":
		// "linux-arm" was renamed to "linux-arm-scaleway" in CL 303230.
		fmt.Fprintln(os.Stderr, "skipping test: linux-arm-scaleway builder lacks sufficient memory (https://golang.org/issue/32834)")
	case "plan9-arm":
		fmt.Fprintln(os.Stderr, "skipping test: plan9-arm builder lacks sufficient memory (https://golang.org/issue/38772)")
	case "netbsd-arm-bsiegert", "netbsd-arm64-bsiegert":
		// As of 2021-06-02, these builders are running with GO_TEST_TIMEOUT_SCALE=10,
		// and there is only one of each. We shouldn't waste those scarce resources
		// running very slow tests.
		fmt.Fprintf(os.Stderr, "skipping test: %s builder is very slow\n", b)
	case "dragonfly-amd64":
		// As of 2021-11-02, this builder is running with GO_TEST_TIMEOUT_SCALE=2,
		// and seems to have unusually slow disk performance.
		fmt.Fprintln(os.Stderr, "skipping test: dragonfly-amd64 has slow disk (https://golang.org/issue/45216)")
	case "linux-riscv64-unmatched":
		// As of 2021-11-03, this builder is empirically not fast enough to run
		// gopls tests. Ideally we should make the tests faster in short mode
		// and/or fix them to not assume arbitrary deadlines.
		// For now, we'll skip them instead.
		fmt.Fprintf(os.Stderr, "skipping test: %s builder is too slow (https://golang.org/issue/49321)\n", b)
	default:
		return
	}
	os.Exit(0)
}

// Go1Point returns the x in Go 1.x.
func Go1Point() int {
	for i := len(build.Default.ReleaseTags) - 1; i >= 0; i-- {
		var version int
		if _, err := fmt.Sscanf(build.Default.ReleaseTags[i], "go1.%d", &version); err != nil {
			continue
		}
		return version
	}
	panic("bad release tags")
}

// NeedsGo1Point skips t if the Go version used to run the test is older than
// 1.x.
func NeedsGo1Point(t testing.TB, x int) {
	if Go1Point() < x {
		t.Helper()
		t.Skipf("running Go version %q is version 1.%d, older than required 1.%d", runtime.Version(), Go1Point(), x)
	}
}

// SkipAfterGo1Point skips t if the Go version used to run the test is newer than
// 1.x.
func SkipAfterGo1Point(t testing.TB, x int) {
	if Go1Point() > x {
		t.Helper()
		t.Skipf("running Go version %q is version 1.%d, newer than maximum 1.%d", runtime.Version(), Go1Point(), x)
	}
}

// Deadline returns the deadline of t, if known,
// using the Deadline method added in Go 1.15.
func Deadline(t testing.TB) (time.Time, bool) {
	td, ok := t.(interface {
		Deadline() (time.Time, bool)
	})
	if !ok {
		return time.Time{}, false
	}
	return td.Deadline()
}

// WriteImportcfg writes an importcfg file used by the compiler or linker to
// dstPath containing entries for the packages in std and cmd in addition
// to the package to package file mappings in additionalPackageFiles.
func WriteImportcfg(t testing.TB, dstPath string, additionalPackageFiles map[string]string) {
	importcfg, err := goroot.Importcfg()
	for k, v := range additionalPackageFiles {
		importcfg += fmt.Sprintf("\npackagefile %s=%s", k, v)
	}
	if err != nil {
		t.Fatalf("preparing the importcfg failed: %s", err)
	}
	ioutil.WriteFile(dstPath, []byte(importcfg), 0655)
	if err != nil {
		t.Fatalf("writing the importcfg failed: %s", err)
	}
}

var (
	gorootOnce sync.Once
	gorootPath string
	gorootErr  error
)

func findGOROOT() (string, error) {
	gorootOnce.Do(func() {
		gorootPath = runtime.GOROOT()
		if gorootPath != "" {
			// If runtime.GOROOT() is non-empty, assume that it is valid. (It might
			// not be: for example, the user may have explicitly set GOROOT
			// to the wrong directory.)
			return
		}

		cmd := exec.Command("go", "env", "GOROOT")
		out, err := cmd.Output()
		if err != nil {
			gorootErr = fmt.Errorf("%v: %v", cmd, err)
		}
		gorootPath = strings.TrimSpace(string(out))
	})

	return gorootPath, gorootErr
}

// GOROOT reports the path to the directory containing the root of the Go
// project source tree. This is normally equivalent to runtime.GOROOT, but
// works even if the test binary was built with -trimpath.
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
