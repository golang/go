// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || (freebsd && amd64)

package sanitizers_test

import (
	"internal/platform"
	"internal/testenv"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// clangMajorVersion detects the version of clang installed on the system.
// Stripped down version of compilerVersion from cmd/go/internal/work/init.go
func clangMajorVersion() (int, error) {
	cc := os.Getenv("CC")
	out, err := exec.Command(cc, "--version").Output()
	if err != nil {
		// Compiler does not support "--version" flag: not Clang or GCC.
		return 0, err
	}

	var match [][]byte
	clangRE := regexp.MustCompile(`clang version (\d+)\.(\d+)`)
	if match = clangRE.FindSubmatch(out); len(match) > 0 {
		compiler.name = "clang"
	}

	if len(match) < 3 {
		return 0, nil // "unknown"
	}
	major, err := strconv.Atoi(string(match[1]))
	if err != nil {
		return 0, err
	}
	return major, nil
}

func TestMSAN(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	goos, err := goEnv("GOOS")
	if err != nil {
		t.Fatal(err)
	}
	goarch, err := goEnv("GOARCH")
	if err != nil {
		t.Fatal(err)
	}
	// The msan tests require support for the -msan option.
	if !platform.MSanSupported(goos, goarch) {
		t.Skipf("skipping on %s/%s; -msan option is not supported.", goos, goarch)
	}

	t.Parallel()
	// Overcommit is enabled by default on FreeBSD (vm.overcommit=0, see tuning(7)).
	// Do not skip tests with stricter overcommit settings unless testing shows that FreeBSD has similar issues.
	if goos == "linux" {
		requireOvercommit(t)
	}
	config := configure("memory")
	config.skipIfCSanitizerBroken(t)

	mustRun(t, config.goCmd("build", "std"))

	cases := []struct {
		src             string
		wantErr         bool
		experiments     []string
		clangMinVersion int
		clangMaxVersion int
	}{
		{src: "msan.go"},
		{src: "msan2.go"},
		{src: "msan2_cmsan.go"},
		{src: "msan3.go"},
		{src: "msan4.go"},
		{src: "msan5.go"},
		{src: "msan6.go"},
		{src: "msan7.go"},
		{src: "msan8.go", clangMaxVersion: 15},
		{src: "msan8_clang16.go", clangMinVersion: 16},
		{src: "msan_fail.go", wantErr: true},
		// This may not always fail specifically due to MSAN. It may sometimes
		// fail because of a fault. However, we don't care what kind of error we
		// get here, just that we get an error. This is an MSAN test because without
		// MSAN it would not fail deterministically.
		{src: "arena_fail.go", wantErr: true, experiments: []string{"arenas"}},
	}

	clangVersion, err := clangMajorVersion()
	if err != nil {
		t.Logf("could not detect clang version: %v", err)
	}

	for _, tc := range cases {
		tc := tc
		name := strings.TrimSuffix(tc.src, ".go")
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			if clangVersion > 0 {
				if tc.clangMinVersion > 0 && clangVersion < tc.clangMinVersion {
					t.Skipf("skipping on clang %d; requires >= %d", clangVersion, tc.clangMinVersion)
				}

				if tc.clangMaxVersion > 0 && clangVersion > tc.clangMaxVersion {
					t.Skipf("skipping on clang %d; requires <= %d", clangVersion, tc.clangMaxVersion)
				}
			}

			dir := newTempDir(t)
			defer dir.RemoveAll(t)

			outPath := dir.Join(name)
			buildcmd := config.goCmdWithExperiments("build", []string{"-o", outPath, srcPath(tc.src)}, tc.experiments)
			// allow tests to define -f flags in CGO_CFLAGS
			buildcmd.Env = append(buildcmd.Env, "CGO_CFLAGS_ALLOW=-f.*")
			mustRun(t, buildcmd)

			cmd := hangProneCmd(outPath)
			if tc.wantErr {
				out, err := cmd.CombinedOutput()
				if err != nil {
					return
				}
				t.Fatalf("%#q exited without error; want MSAN failure\n%s", strings.Join(cmd.Args, " "), out)
			}
			mustRun(t, cmd)
		})
	}
}
