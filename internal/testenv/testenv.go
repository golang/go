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
	"os/exec"
	"runtime"
	"strings"
	"sync"
)

// Testing is an abstraction of a *testing.T.
type Testing interface {
	Skipf(format string, args ...interface{})
	Fatalf(format string, args ...interface{})
}

type helperer interface {
	Helper()
}

// packageMainIsDevel reports whether the module containing package main
// is a development version (if module information is available).
//
// Builds in GOPATH mode and builds that lack module information are assumed to
// be development versions.
var packageMainIsDevel = func() bool { return true }

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
func NeedsTool(t Testing, tool string) {
	if t, ok := t.(helperer); ok {
		t.Helper()
	}
	err := hasTool(tool)
	if err == nil {
		return
	}
	if allowMissingTool(tool) {
		t.Skipf("skipping because %s tool not available: %v", tool, err)
	} else {
		t.Fatalf("%s tool not available: %v", tool, err)
	}
}

// NeedsGoPackages skips t if the go/packages driver (or 'go' tool) implied by
// the current process environment is not present in the path.
func NeedsGoPackages(t Testing) {
	if t, ok := t.(helperer); ok {
		t.Helper()
	}

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
func NeedsGoPackagesEnv(t Testing, env []string) {
	if t, ok := t.(helperer); ok {
		t.Helper()
	}

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

// ExitIfSmallMachine emits a helpful diagnostic and calls os.Exit(0) if the
// current machine is a builder known to have scarce resources.
//
// It should be called from within a TestMain function.
func ExitIfSmallMachine() {
	switch os.Getenv("GO_BUILDER_NAME") {
	case "linux-arm":
		fmt.Fprintln(os.Stderr, "skipping test: linux-arm builder lacks sufficient memory (https://golang.org/issue/32834)")
		os.Exit(0)
	case "plan9-arm":
		fmt.Fprintln(os.Stderr, "skipping test: plan9-arm builder lacks sufficient memory (https://golang.org/issue/38772)")
		os.Exit(0)
	}
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
func NeedsGo1Point(t Testing, x int) {
	if t, ok := t.(helperer); ok {
		t.Helper()
	}
	if Go1Point() < x {
		t.Skipf("running Go version %q is version 1.%d, older than required 1.%d", runtime.Version(), Go1Point(), x)
	}
}

// SkipAfterGo1Point skips t if the Go version used to run the test is newer than
// 1.x.
func SkipAfterGo1Point(t Testing, x int) {
	if t, ok := t.(helperer); ok {
		t.Helper()
	}
	if Go1Point() > x {
		t.Skipf("running Go version %q is version 1.%d, newer than maximum 1.%d", runtime.Version(), Go1Point(), x)
	}
}
