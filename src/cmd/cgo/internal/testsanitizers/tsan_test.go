// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || (freebsd && amd64)

package sanitizers_test

import (
	"internal/testenv"
	"strings"
	"testing"
)

func TestTSAN(t *testing.T) {
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
	if !compilerRequiredTsanVersion(goos, goarch) {
		t.Skipf("skipping on %s/%s; compiler version for -tsan option is too old.", goos, goarch)
	}

	t.Parallel()
	requireOvercommit(t)
	config := configure("thread")
	config.skipIfCSanitizerBroken(t)

	mustRun(t, config.goCmd("build", "std"))

	cases := []struct {
		src          string
		needsRuntime bool
	}{
		{src: "tsan.go"},
		{src: "tsan2.go"},
		{src: "tsan3.go"},
		{src: "tsan4.go"},
		{src: "tsan5.go", needsRuntime: true},
		{src: "tsan6.go", needsRuntime: true},
		{src: "tsan7.go", needsRuntime: true},
		{src: "tsan8.go"},
		{src: "tsan9.go"},
		{src: "tsan10.go", needsRuntime: true},
		{src: "tsan11.go", needsRuntime: true},
		{src: "tsan12.go", needsRuntime: true},
		{src: "tsan13.go", needsRuntime: true},
		{src: "tsan14.go", needsRuntime: true},
		{src: "tsan15.go", needsRuntime: true},
	}
	for _, tc := range cases {
		tc := tc
		name := strings.TrimSuffix(tc.src, ".go")
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			dir := newTempDir(t)
			defer dir.RemoveAll(t)

			outPath := dir.Join(name)
			mustRun(t, config.goCmd("build", "-o", outPath, srcPath(tc.src)))

			cmd := hangProneCmd(outPath)
			if tc.needsRuntime {
				config.skipIfRuntimeIncompatible(t)
			}
			// If we don't see halt_on_error, the program
			// will only exit non-zero if we call C.exit.
			cmd.Env = append(cmd.Environ(), "TSAN_OPTIONS=halt_on_error=1")
			mustRun(t, cmd)
		})
	}
}
