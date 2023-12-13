// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || (freebsd && amd64)

package sanitizers_test

import (
	"internal/platform"
	"internal/testenv"
	"strings"
	"testing"
)

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
		src         string
		wantErr     bool
		experiments []string
	}{
		{src: "msan.go"},
		{src: "msan2.go"},
		{src: "msan2_cmsan.go"},
		{src: "msan3.go"},
		{src: "msan4.go"},
		{src: "msan5.go"},
		{src: "msan6.go"},
		{src: "msan7.go"},
		{src: "msan8.go"},
		{src: "msan_fail.go", wantErr: true},
		// This may not always fail specifically due to MSAN. It may sometimes
		// fail because of a fault. However, we don't care what kind of error we
		// get here, just that we get an error. This is an MSAN test because without
		// MSAN it would not fail deterministically.
		{src: "arena_fail.go", wantErr: true, experiments: []string{"arenas"}},
	}
	for _, tc := range cases {
		tc := tc
		name := strings.TrimSuffix(tc.src, ".go")
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			dir := newTempDir(t)
			defer dir.RemoveAll(t)

			outPath := dir.Join(name)
			buildcmd := config.goCmdWithExperiments("build", []string{"-o", outPath, srcPath(tc.src)}, tc.experiments)
			// allow tests to define -f flags in CGO_CFLAGS
			buildcmd.Env = append(buildcmd.Environ(), "CGO_CFLAGS_ALLOW=-f.*")
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
