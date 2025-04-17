// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package scripttest adapts the script engine for use in tests.
package scripttest

import (
	"internal/testenv"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// SetupTestGoRoot sets up a temporary GOROOT for use with script test
// execution. It copies the existing goroot bin and pkg dirs using
// symlinks (if possible) or raw copying. Return value is the path to
// the newly created testgoroot dir.
func SetupTestGoRoot(t *testing.T, tmpdir string, goroot string) string {
	mustMkdir := func(path string) {
		if err := os.MkdirAll(path, 0777); err != nil {
			t.Fatalf("SetupTestGoRoot mkdir %s failed: %v", path, err)
		}
	}

	replicateDir := func(srcdir, dstdir string) {
		files, err := os.ReadDir(srcdir)
		if err != nil {
			t.Fatalf("inspecting %s: %v", srcdir, err)
		}
		for _, file := range files {
			fn := file.Name()
			linkOrCopy(t, filepath.Join(srcdir, fn), filepath.Join(dstdir, fn))
		}
	}

	// Create various dirs in testgoroot.
	findToolOnce.Do(func() { findToolSub(t) })
	if toolsub == "" {
		t.Fatal("failed to find toolsub")
	}

	tomake := []string{
		"bin",
		"src",
		"pkg",
		filepath.Join("pkg", "include"),
		toolsub,
	}
	made := []string{}
	tgr := filepath.Join(tmpdir, "testgoroot")
	mustMkdir(tgr)
	for _, targ := range tomake {
		path := filepath.Join(tgr, targ)
		mustMkdir(path)
		made = append(made, path)
	}

	// Replicate selected portions of the content.
	replicateDir(filepath.Join(goroot, "bin"), made[0])
	replicateDir(filepath.Join(goroot, "src"), made[1])
	replicateDir(filepath.Join(goroot, "pkg", "include"), made[3])
	replicateDir(filepath.Join(goroot, toolsub), made[4])

	return tgr
}

// ReplaceGoToolInTestGoRoot replaces the go tool binary toolname with
// an alternate executable newtoolpath within a test GOROOT directory
// previously created by SetupTestGoRoot.
func ReplaceGoToolInTestGoRoot(t *testing.T, testgoroot, toolname, newtoolpath string) {
	findToolOnce.Do(func() { findToolSub(t) })
	if toolsub == "" {
		t.Fatal("failed to find toolsub")
	}

	exename := toolname
	if runtime.GOOS == "windows" {
		exename += ".exe"
	}
	toolpath := filepath.Join(testgoroot, toolsub, exename)
	if err := os.Remove(toolpath); err != nil {
		t.Fatalf("removing %s: %v", toolpath, err)
	}
	linkOrCopy(t, newtoolpath, toolpath)
}

// toolsub is the tool subdirectory underneath GOROOT.
var toolsub string

// findToolOnce runs findToolSub only once.
var findToolOnce sync.Once

// findToolSub sets toolsub to the value used by the current go command.
func findToolSub(t *testing.T) {
	gocmd := testenv.Command(t, testenv.GoToolPath(t), "env", "GOHOSTARCH")
	gocmd = testenv.CleanCmdEnv(gocmd)
	goHostArchBytes, err := gocmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s failed: %v\n%s", gocmd, err, goHostArchBytes)
	}
	goHostArch := strings.TrimSpace(string(goHostArchBytes))
	toolsub = filepath.Join("pkg", "tool", runtime.GOOS+"_"+goHostArch)
}

// linkOrCopy creates a link to src at dst, or if the symlink fails
// (platform doesn't support) then copies src to dst.
func linkOrCopy(t *testing.T, src, dst string) {
	err := os.Symlink(src, dst)
	if err == nil {
		return
	}
	srcf, err := os.Open(src)
	if err != nil {
		t.Fatalf("copying %s to %s: %v", src, dst, err)
	}
	defer srcf.Close()
	perm := os.O_WRONLY | os.O_CREATE | os.O_EXCL
	dstf, err := os.OpenFile(dst, perm, 0o777)
	if err != nil {
		t.Fatalf("copying %s to %s: %v", src, dst, err)
	}
	_, err = io.Copy(dstf, srcf)
	if closeErr := dstf.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatalf("copying %s to %s: %v", src, dst, err)
	}
}
