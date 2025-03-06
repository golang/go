// Copyright 2025 The Go Authors. All rights reserved.
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

func TestLSAN(t *testing.T) {
	config := mustHaveLSAN(t)

	t.Parallel()
	mustRun(t, config.goCmd("build", "std"))

	cases := []struct {
		src           string
		leakError     string
		errorLocation string
	}{
		{src: "lsan1.go", leakError: "detected memory leaks", errorLocation: "lsan1.go:11"},
		{src: "lsan2.go"},
		{src: "lsan3.go"},
	}
	for _, tc := range cases {
		name := strings.TrimSuffix(tc.src, ".go")
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			dir := newTempDir(t)
			defer dir.RemoveAll(t)

			outPath := dir.Join(name)
			mustRun(t, config.goCmd("build", "-o", outPath, srcPath(tc.src)))

			cmd := hangProneCmd(outPath)
			if tc.leakError == "" {
				mustRun(t, cmd)
			} else {
				outb, err := cmd.CombinedOutput()
				out := string(outb)
				if err != nil || len(out) > 0 {
					t.Logf("%s\n%v\n%s", cmd, err, out)
				}
				if err != nil && strings.Contains(out, tc.leakError) {
					// This string is output if the
					// sanitizer library needs a
					// symbolizer program and can't find it.
					const noSymbolizer = "external symbolizer"
					if tc.errorLocation != "" &&
						!strings.Contains(out, tc.errorLocation) &&
						!strings.Contains(out, noSymbolizer) &&
						compilerSupportsLocation() {

						t.Errorf("output does not contain expected location of the error %q", tc.errorLocation)
					}
				} else {
					t.Errorf("output does not contain expected leak error %q", tc.leakError)
				}

				// Make sure we can disable the leak check.
				cmd = hangProneCmd(outPath)
				replaceEnv(cmd, "ASAN_OPTIONS", "detect_leaks=0")
				mustRun(t, cmd)
			}
		})
	}
}

func mustHaveLSAN(t *testing.T) *config {
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
	// LSAN is a subset of ASAN, so just check for ASAN support.
	if !platform.ASanSupported(goos, goarch) {
		t.Skipf("skipping on %s/%s; -asan option is not supported.", goos, goarch)
	}

	if !compilerRequiredLsanVersion(goos, goarch) {
		t.Skipf("skipping on %s/%s: too old version of compiler", goos, goarch)
	}

	requireOvercommit(t)

	config := configure("leak")
	config.skipIfCSanitizerBroken(t)

	return config
}
