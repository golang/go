// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sanitizers_test

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestShared(t *testing.T) {
	t.Parallel()
	requireOvercommit(t)

	GOOS, err := goEnv("GOOS")
	if err != nil {
		t.Fatal(err)
	}

	GOARCH, err := goEnv("GOARCH")
	if err != nil {
		t.Fatal(err)
	}

	libExt := "so"
	if GOOS == "darwin" {
		libExt = "dylib"
	}

	cases := []struct {
		src       string
		sanitizer string
	}{
		{
			src:       "msan_shared.go",
			sanitizer: "memory",
		},
		{
			src:       "tsan_shared.go",
			sanitizer: "thread",
		},
	}

	for _, tc := range cases {
		tc := tc
		name := strings.TrimSuffix(tc.src, ".go")
		//The memory sanitizer tests require support for the -msan option.
		if tc.sanitizer == "memory" && !mSanSupported(GOOS, GOARCH) {
			t.Logf("skipping %s test on %s/%s; -msan option is not supported.", name, GOOS, GOARCH)
			continue
		}
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			config := configure(tc.sanitizer)
			config.skipIfCSanitizerBroken(t)

			dir := newTempDir(t)
			defer dir.RemoveAll(t)

			lib := dir.Join(fmt.Sprintf("lib%s.%s", name, libExt))
			mustRun(t, config.goCmd("build", "-buildmode=c-shared", "-o", lib, srcPath(tc.src)))

			cSrc := dir.Join("main.c")
			if err := os.WriteFile(cSrc, cMain, 0600); err != nil {
				t.Fatalf("failed to write C source file: %v", err)
			}

			dstBin := dir.Join(name)
			cmd, err := cc(config.cFlags...)
			if err != nil {
				t.Fatal(err)
			}
			cmd.Args = append(cmd.Args, config.ldFlags...)
			cmd.Args = append(cmd.Args, "-o", dstBin, cSrc, lib)
			mustRun(t, cmd)

			cmd = hangProneCmd(dstBin)
			replaceEnv(cmd, "LD_LIBRARY_PATH", ".")
			mustRun(t, cmd)
		})
	}
}
