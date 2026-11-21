// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"archive/zip"
	"bytes"
	"errors"
	"internal/testenv"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInvalidRemote(t *testing.T) {
	for _, vcs := range []string{"git", "hg", "svn", "fossil"} {
		for _, remote := range []string{
			"--upload-pack=touch /tmp/x://",
			"-oProxyCommand=x://y",
			"https://example.com/repo\n[hooks]\npre-pull = x",
			"https://example.com/\x00repo",
		} {
			if _, err := NewRepo(t.Context(), vcs, remote, false); err == nil {
				t.Errorf("NewRepo(%q, %q) succeeded", vcs, remote)
			} else if !strings.Contains(err.Error(), "invalid vcs remote") {
				t.Errorf("NewRepo(%q, %q): %v, want invalid vcs remote", vcs, remote, err)
			}
		}
	}
}

func TestHgLiteralRevision(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping hg test in short mode: hg is slow")
	}
	testenv.MustHaveExecPath(t, "hg")
	t.Setenv("HGRCPATH", "")
	t.Setenv("HGPLAINEXCEPT", "")

	dir := t.TempDir()
	hg := func(args ...string) string {
		t.Helper()
		cmd := testenv.Command(t, "hg", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("hg %q: %v\n%s", args, err, out)
		}
		return string(out)
	}
	write := func(file, contents string) {
		t.Helper()
		p := filepath.Join(dir, filepath.FromSlash(file))
		if err := os.MkdirAll(filepath.Dir(p), 0777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(contents), 0666); err != nil {
			t.Fatal(err)
		}
	}

	const (
		tag    = "v1.2.3+build"
		quoted = `release"two"\notes`
		subdir = "pkg[x]"
		file   = subdir + "/note.txt"
		first  = "first\n"
	)
	hg("init")
	write(file, first)
	write("pkgx/note.txt", "sibling\n")
	hg("add", ".")
	hg("commit", "-u", "gopher", "-m", "first")
	old := hg("log", "-r", ".", "-T", "{node}")
	write(file, "second\n")
	hg("commit", "-u", "gopher", "-m", "second")
	new := hg("log", "-r", ".", "-T", "{node}")
	hg("tag", "-l", "-r", old, tag, "base")
	// A distinct tag that includes the surrounding quotes must not shadow tag.
	hg("tag", "-l", "-r", new, quoted, `"`+tag+`"`, "candidate")

	ctx := testContext(t)
	r, err := NewRepo(ctx, "hg", dir, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ tag, node string }{{tag, old}, {quoted, new}} {
		info, err := r.Stat(ctx, tt.tag)
		if err != nil {
			t.Fatalf("Stat(%q): %v", tt.tag, err)
		}
		if info.Name != tt.node {
			t.Fatalf("Stat(%q).Name = %q, want %q", tt.tag, info.Name, tt.node)
		}
	}
	// Two known tag names joined by '+' are not themselves a known tag.
	var unknown *UnknownRevisionError
	if info, err := r.Stat(ctx, "base+candidate"); !errors.As(err, &unknown) {
		t.Errorf("Stat(base+candidate) = %v, %v; want unknown revision", info, err)
	}
	if got, err := r.RecentTag(ctx, quoted, "", func(string) bool { return true }); err != nil || got != tag {
		t.Errorf("RecentTag(%q) = %q, %v; want %q", quoted, got, err, tag)
	}
	if got, err := r.DescendsFrom(ctx, quoted, tag); err != nil || !got {
		t.Errorf("DescendsFrom(%q, %q) = %v, %v; want true, nil", quoted, tag, got, err)
	}
	if got, err := r.DescendsFrom(ctx, tag, quoted); err != nil || got {
		t.Errorf("DescendsFrom(%q, %q) = %v, %v; want false, nil", tag, quoted, got, err)
	}

	// Moving the tag after Stat must not change the revision selected by reads.
	hg("tag", "-l", "-f", "-r", new, tag)
	if data, err := r.ReadFile(ctx, tag, file, 100); err != nil || string(data) != first {
		t.Errorf("ReadFile(%q, %q) = %q, %v; want %q", tag, file, data, err, first)
	}
	rc, err := r.ReadZip(ctx, tag, subdir, 100000)
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()
	data, err := io.ReadAll(rc)
	if err != nil {
		t.Fatal(err)
	}
	z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if len(z.File) != 1 || z.File[0].Name != "prefix/"+file {
		t.Fatalf("ReadZip(%q, %q) did not contain only %q", tag, subdir, "prefix/"+file)
	}
	if got, err := fs.ReadFile(z, z.File[0].Name); err != nil || string(got) != first {
		t.Errorf("archive %q = %q, %v; want %q", z.File[0].Name, got, err, first)
	}
}
