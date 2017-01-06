// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build appengine

package dl

import (
	"sort"
	"strings"
	"testing"
)

func TestParseVersion(t *testing.T) {
	for _, c := range []struct {
		in       string
		maj, min int
		tail     string
	}{
		{"go1.5", 5, 0, ""},
		{"go1.5beta1", 5, 0, "beta1"},
		{"go1.5.1", 5, 1, ""},
		{"go1.5.1rc1", 5, 1, "rc1"},
	} {
		maj, min, tail := parseVersion(c.in)
		if maj != c.maj || min != c.min || tail != c.tail {
			t.Errorf("parseVersion(%q) = %v, %v, %q; want %v, %v, %q",
				c.in, maj, min, tail, c.maj, c.min, c.tail)
		}
	}
}

func TestFileOrder(t *testing.T) {
	fs := []File{
		{Filename: "go1.3.src.tar.gz", Version: "go1.3", OS: "", Arch: "", Kind: "source"},
		{Filename: "go1.3.1.src.tar.gz", Version: "go1.3.1", OS: "", Arch: "", Kind: "source"},
		{Filename: "go1.3.linux-amd64.tar.gz", Version: "go1.3", OS: "linux", Arch: "amd64", Kind: "archive"},
		{Filename: "go1.3.1.linux-amd64.tar.gz", Version: "go1.3.1", OS: "linux", Arch: "amd64", Kind: "archive"},
		{Filename: "go1.3.darwin-amd64.tar.gz", Version: "go1.3", OS: "darwin", Arch: "amd64", Kind: "archive"},
		{Filename: "go1.3.darwin-amd64.pkg", Version: "go1.3", OS: "darwin", Arch: "amd64", Kind: "installer"},
		{Filename: "go1.3.darwin-386.tar.gz", Version: "go1.3", OS: "darwin", Arch: "386", Kind: "archive"},
		{Filename: "go1.3beta1.linux-amd64.tar.gz", Version: "go1.3beta1", OS: "linux", Arch: "amd64", Kind: "archive"},
		{Filename: "go1.3beta2.linux-amd64.tar.gz", Version: "go1.3beta2", OS: "linux", Arch: "amd64", Kind: "archive"},
		{Filename: "go1.3rc1.linux-amd64.tar.gz", Version: "go1.3rc1", OS: "linux", Arch: "amd64", Kind: "archive"},
		{Filename: "go1.2.linux-amd64.tar.gz", Version: "go1.2", OS: "linux", Arch: "amd64", Kind: "archive"},
		{Filename: "go1.2.2.linux-amd64.tar.gz", Version: "go1.2.2", OS: "linux", Arch: "amd64", Kind: "archive"},
	}
	sort.Sort(fileOrder(fs))
	var s []string
	for _, f := range fs {
		s = append(s, f.Filename)
	}
	got := strings.Join(s, "\n")
	want := strings.Join([]string{
		"go1.3.1.src.tar.gz",
		"go1.3.1.linux-amd64.tar.gz",
		"go1.3.src.tar.gz",
		"go1.3.darwin-386.tar.gz",
		"go1.3.darwin-amd64.tar.gz",
		"go1.3.darwin-amd64.pkg",
		"go1.3.linux-amd64.tar.gz",
		"go1.2.2.linux-amd64.tar.gz",
		"go1.2.linux-amd64.tar.gz",
		"go1.3rc1.linux-amd64.tar.gz",
		"go1.3beta2.linux-amd64.tar.gz",
		"go1.3beta1.linux-amd64.tar.gz",
	}, "\n")
	if got != want {
		t.Errorf("sort order is\n%s\nwant:\n%s", got, want)
	}
}

func TestFilesToReleases(t *testing.T) {
	fs := []File{
		{Version: "go1.7.4", OS: "darwin"},
		{Version: "go1.7.4", OS: "windows"},
		{Version: "go1.7", OS: "darwin"},
		{Version: "go1.7", OS: "windows"},
		{Version: "go1.6.2", OS: "darwin"},
		{Version: "go1.6.2", OS: "windows"},
		{Version: "go1.6", OS: "darwin"},
		{Version: "go1.6", OS: "windows"},
		{Version: "go1.5.2", OS: "darwin"},
		{Version: "go1.5.2", OS: "windows"},
		{Version: "go1.5", OS: "darwin"},
		{Version: "go1.5", OS: "windows"},
		{Version: "go1.5beta1", OS: "windows"},
	}
	stable, unstable, archive := filesToReleases(fs)
	if got, want := len(stable), 2; want != got {
		t.Errorf("len(stable): got %v, want %v", got, want)
	} else {
		if got, want := stable[0].Version, "go1.7.4"; want != got {
			t.Errorf("stable[0].Version: got %v, want %v", got, want)
		}
		if got, want := stable[1].Version, "go1.6.2"; want != got {
			t.Errorf("stable[1].Version: got %v, want %v", got, want)
		}
	}
	if got, want := len(unstable), 0; want != got {
		t.Errorf("len(unstable): got %v, want %v", got, want)
	}
	if got, want := len(archive), 4; want != got {
		t.Errorf("len(archive): got %v, want %v", got, want)
	}
}

func TestOldUnstableNotShown(t *testing.T) {
	fs := []File{
		{Version: "go1.7.4"},
		{Version: "go1.7"},
		{Version: "go1.7beta1"},
	}
	_, unstable, _ := filesToReleases(fs)
	if len(unstable) != 0 {
		t.Errorf("got unstable, want none")
	}
}

func TestUnstableShown(t *testing.T) {
	fs := []File{
		{Version: "go1.8beta2"},
		{Version: "go1.8rc1"},
		{Version: "go1.7.4"},
		{Version: "go1.7"},
		{Version: "go1.7beta1"},
	}
	_, unstable, _ := filesToReleases(fs)
	if got, want := len(unstable), 1; got != want {
		t.Fatalf("len(unstable): got %v, want %v", got, want)
	}
	// show rcs ahead of betas.
	if got, want := unstable[0].Version, "go1.8rc1"; got != want {
		t.Fatalf("unstable[0].Version: got %v, want %v", got, want)
	}
}
