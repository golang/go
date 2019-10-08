// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testenv contains helper functions for skipping tests
// based on which tools are present in the environment.
package testenv

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"runtime"
	"strings"
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

func hasTool(tool string) error {
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
	}
	return nil
}

func allowMissingTool(tool string) bool {
	if runtime.GOOS == "android" {
		// Android builds generally run tests on a separate machine from the build,
		// so don't expect any external tools to be available.
		return true
	}

	switch tool {
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
	if os.Getenv("GO_BUILDER_NAME") == "linux-arm" {
		fmt.Fprintln(os.Stderr, "skipping test: linux-arm builder lacks sufficient memory (https://golang.org/issue/32834)")
		os.Exit(0)
	}
}
