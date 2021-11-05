// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sanitizers_test

import (
	"strings"
	"testing"
)

func TestASAN(t *testing.T) {
	goos, err := goEnv("GOOS")
	if err != nil {
		t.Fatal(err)
	}
	goarch, err := goEnv("GOARCH")
	if err != nil {
		t.Fatal(err)
	}
	// The asan tests require support for the -asan option.
	if !aSanSupported(goos, goarch) {
		t.Skipf("skipping on %s/%s; -asan option is not supported.", goos, goarch)
	}

	t.Parallel()
	requireOvercommit(t)
	config := configure("address")
	config.skipIfCSanitizerBroken(t)

	mustRun(t, config.goCmd("build", "std"))

	cases := []struct {
		src               string
		memoryAccessError string
	}{
		{src: "asan1_fail.go", memoryAccessError: "heap-use-after-free"},
		{src: "asan2_fail.go", memoryAccessError: "heap-buffer-overflow"},
		{src: "asan3_fail.go", memoryAccessError: "use-after-poison"},
		{src: "asan4_fail.go", memoryAccessError: "use-after-poison"},
		{src: "asan_useAfterReturn.go"},
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
			if tc.memoryAccessError != "" {
				out, err := cmd.CombinedOutput()
				if err != nil && strings.Contains(string(out), tc.memoryAccessError) {
					return
				}
				t.Fatalf("%#q exited without expected memory access error\n%s; got failure\n%s", strings.Join(cmd.Args, " "), tc.memoryAccessError, out)
			}
			mustRun(t, cmd)
		})
	}
}
