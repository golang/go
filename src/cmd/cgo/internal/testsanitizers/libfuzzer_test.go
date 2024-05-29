// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || (freebsd && amd64)

package sanitizers_test

import (
	"internal/testenv"
	"strings"
	"testing"
)

func TestLibFuzzer(t *testing.T) {
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
	if !libFuzzerSupported(goos, goarch) {
		t.Skipf("skipping on %s/%s; libfuzzer option is not supported.", goos, goarch)
	}
	config := configure("fuzzer")
	config.skipIfCSanitizerBroken(t)

	cases := []struct {
		goSrc         string
		cSrc          string
		expectedError string
		short         bool
	}{
		{goSrc: "libfuzzer1.go", expectedError: "panic: found it", short: true},
		{goSrc: "libfuzzer2.go", cSrc: "libfuzzer2.c", expectedError: "panic: found it"},
	}
	for _, tc := range cases {
		tc := tc
		name := strings.TrimSuffix(tc.goSrc, ".go")
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Skip long-running tests in short mode.
			if testing.Short() && !tc.short {
				t.Skipf("%s can take upwards of minutes to run; skipping in short mode", name)
			}

			dir := newTempDir(t)
			defer dir.RemoveAll(t)

			// build Go code in libfuzzer mode to a c-archive
			outPath := dir.Join(name)
			archivePath := dir.Join(name + ".a")
			mustRun(t, config.goCmd("build", "-buildmode=c-archive", "-o", archivePath, srcPath(tc.goSrc)))

			// build C code (if any) and link with Go code
			cmd, err := cc(config.cFlags...)
			if err != nil {
				t.Fatalf("error running cc: %v", err)
			}
			cmd.Args = append(cmd.Args, config.ldFlags...)
			cmd.Args = append(cmd.Args, "-o", outPath, "-I", dir.Base())
			if tc.cSrc != "" {
				cmd.Args = append(cmd.Args, srcPath(tc.cSrc))
			}
			cmd.Args = append(cmd.Args, archivePath)
			mustRun(t, cmd)

			cmd = hangProneCmd(outPath)
			cmd.Dir = dir.Base()
			outb, err := cmd.CombinedOutput()
			out := string(outb)
			if err == nil {
				t.Fatalf("fuzzing succeeded unexpectedly; output:\n%s", out)
			}
			if !strings.Contains(out, tc.expectedError) {
				t.Errorf("exited without expected error %q; got\n%s", tc.expectedError, out)
			}
		})
	}
}

// libFuzzerSupported is a copy of the function internal/platform.FuzzInstrumented,
// because the internal package can't be used here.
func libFuzzerSupported(goos, goarch string) bool {
	switch goarch {
	case "amd64", "arm64":
		// TODO(#14565): support more architectures.
		switch goos {
		case "darwin", "freebsd", "linux", "windows":
			return true
		default:
			return false
		}
	default:
		return false
	}
}
