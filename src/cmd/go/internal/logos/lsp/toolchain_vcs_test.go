// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const gitFailureEnv = "LOGOS_LSP_TEST_FAILING_GIT"

func TestToolchainMetadataIgnoresVCS(t *testing.T) {
	root, tc := realWorkspace(t)
	appDir := filepath.Join(root, "lmod", "app")
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o700); err != nil {
		t.Fatal(err)
	}
	bin := t.TempDir()
	git := filepath.Join(bin, "git")
	if runtime.GOOS == "windows" {
		git += ".exe"
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Link(exe, git); err != nil {
		data, err := os.ReadFile(exe)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(git, data, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	log := filepath.Join(bin, "git.log")
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv(gitFailureEnv, log)

	// Ordinary builds retain their VCS policy; only metadata fetching opts out.
	cmd := exec.CommandContext(t.Context(), tc.exe, "build", "-n", ".")
	cmd.Args[0] = "logos"
	cmd.Dir = appDir
	if out, err := cmd.CombinedOutput(); err == nil || !strings.Contains(string(out), "error obtaining VCS status") {
		t.Fatalf("ordinary build = %v, output %s; want VCS failure", err, out)
	}
	if err := os.Remove(log); err != nil {
		t.Fatalf("ordinary build did not invoke the failing git: %v", err)
	}

	low, err := tc.Lower(t.Context(), appDir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(appDir, "main.lgo"); len(low.Files) != 1 || low.Files[0].Path != want || len(low.Missing) != 0 {
		t.Errorf("lowering = %+v, want the one file %s", low, want)
	}
	if _, err := os.Stat(log); !os.IsNotExist(err) {
		t.Errorf("metadata fetch invoked git: stat = %v", err)
	}
}
