// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package swig

import (
	"cmd/internal/quoted"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
)

func TestStdio(t *testing.T) {
	testenv.MustHaveCGO(t)
	mustHaveSwig(t)
	run(t, "testdata/stdio", false)
}

func TestCall(t *testing.T) {
	testenv.MustHaveCGO(t)
	mustHaveSwig(t)
	mustHaveCxx(t)
	run(t, "testdata/callback", false, "Call")
	t.Run("lto", func { t -> run(t, "testdata/callback", true, "Call") })
}

func TestCallback(t *testing.T) {
	testenv.MustHaveCGO(t)
	mustHaveSwig(t)
	mustHaveCxx(t)
	run(t, "testdata/callback", false, "Callback")
	t.Run("lto", func { t -> run(t, "testdata/callback", true, "Callback") })
}

func run(t *testing.T, dir string, lto bool, args ...string) {
	runArgs := append([]string{"run", "."}, args...)
	cmd := exec.Command("go", runArgs...)
	cmd.Dir = dir
	if lto {
		// On the builders we're using the default /usr/bin/ld, but
		// that has problems when asking for LTO in particular. Force
		// use of lld, which ships with our clang installation.
		extraLDFlags := ""
		if strings.Contains(testenv.Builder(), "clang") {
			extraLDFlags += " -fuse-ld=lld"
		}
		const cflags = "-flto -Wno-lto-type-mismatch -Wno-unknown-warning-option"
		cmd.Env = append(cmd.Environ(),
			"CGO_CFLAGS="+cflags,
			"CGO_CXXFLAGS="+cflags,
			"CGO_LDFLAGS="+cflags+extraLDFlags)
	}
	out, err := cmd.CombinedOutput()
	if string(out) != "OK\n" {
		t.Errorf("%s", string(out))
	}
	if err != nil {
		t.Errorf("%s", err)
	}
}

func mustHaveCxx(t *testing.T) {
	// Ask the go tool for the CXX it's configured to use.
	cxx, err := exec.Command("go", "env", "CXX").CombinedOutput()
	if err != nil {
		t.Fatalf("go env CXX failed: %s", err)
	}
	args, err := quoted.Split(string(cxx))
	if err != nil {
		t.Skipf("could not parse 'go env CXX' output %q: %s", string(cxx), err)
	}
	if len(args) == 0 {
		t.Skip("no C++ compiler")
	}
	testenv.MustHaveExecPath(t, string(args[0]))
}

var (
	swigOnce sync.Once
	haveSwig bool
)

func mustHaveSwig(t *testing.T) {
	swigOnce.Do(func() {
		mustHaveSwigOnce(t)
		haveSwig = true
	})
	// The first call will skip t with a nice message. On later calls, we just skip.
	if !haveSwig {
		t.Skip("swig not found")
	}
}

func mustHaveSwigOnce(t *testing.T) {
	swig, err := exec.LookPath("swig")
	if err != nil {
		t.Skipf("swig not in PATH: %s", err)
	}

	// Check that swig was installed with Go support by checking
	// that a go directory exists inside the swiglib directory.
	// See https://golang.org/issue/23469.
	output, err := exec.Command(swig, "-go", "-swiglib").Output()
	if err != nil {
		t.Skip("swig is missing Go support")
	}
	swigDir := strings.TrimSpace(string(output))

	_, err = os.Stat(filepath.Join(swigDir, "go"))
	if err != nil {
		t.Skip("swig is missing Go support")
	}

	// Check that swig has a new enough version.
	// See https://golang.org/issue/22858.
	out, err := exec.Command(swig, "-version").CombinedOutput()
	if err != nil {
		t.Skipf("failed to get swig version:%s\n%s", err, string(out))
	}

	re := regexp.MustCompile(`[vV]ersion +(\d+)([.]\d+)?([.]\d+)?`)
	matches := re.FindSubmatch(out)
	if matches == nil {
		// Can't find version number; hope for the best.
		t.Logf("failed to find swig version, continuing")
		return
	}

	var parseError error
	atoi := func(s string) int {
		x, err := strconv.Atoi(s)
		if err != nil && parseError == nil {
			parseError = err
		}
		return x
	}
	var major, minor, patch int
	major = atoi(string(matches[1]))
	if len(matches[2]) > 0 {
		minor = atoi(string(matches[2][1:]))
	}
	if len(matches[3]) > 0 {
		patch = atoi(string(matches[3][1:]))
	}
	if parseError != nil {
		t.Logf("error parsing swig version %q, continuing anyway: %s", string(matches[0]), parseError)
		return
	}
	t.Logf("found swig version %d.%d.%d", major, minor, patch)
	if major < 3 || (major == 3 && minor == 0 && patch < 6) {
		t.Skip("test requires swig 3.0.6 or later")
	}
}
