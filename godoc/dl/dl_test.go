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
