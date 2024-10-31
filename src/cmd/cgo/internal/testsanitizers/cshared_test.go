// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || (freebsd && amd64)

package sanitizers_test

import (
	"fmt"
	"internal/platform"
	"internal/testenv"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestShared(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveBuildMode(t, "c-shared")

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
		if tc.sanitizer == "memory" && !platform.MSanSupported(GOOS, GOARCH) {
			t.Logf("skipping %s test on %s/%s; -msan option is not supported.", name, GOOS, GOARCH)
			continue
		}
		if tc.sanitizer == "thread" && !compilerRequiredTsanVersion(GOOS, GOARCH) {
			t.Logf("skipping %s test on %s/%s; compiler version too old for -tsan.", name, GOOS, GOARCH)
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

			cmdArgs := []string{dstBin}
			if tc.sanitizer == "thread" && GOOS == "linux" {
				// Disable ASLR for TSAN. See #59418.
				arch, err := exec.Command("uname", "-m").Output()
				if err != nil {
					t.Fatalf("failed to run `uname -m`: %v", err)
				}
				cmdArgs = []string{"setarch", strings.TrimSpace(string(arch)), "-R", dstBin}
			}
			cmd = hangProneCmd(cmdArgs[0], cmdArgs[1:]...)
			replaceEnv(cmd, "LD_LIBRARY_PATH", ".")
			mustRun(t, cmd)
		})
	}
}
