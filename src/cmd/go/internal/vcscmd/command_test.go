// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcscmd

import (
	"path/filepath"
	"slices"
	"testing"
)

func TestBuild(t *testing.T) {
	repo := filepath.Join(t.TempDir(), "repo")
	for _, tt := range []struct {
		cmd  *Command
		want []string
	}{
		{New("hg").Option("--config", "x=y").Add("lookup").Operands("-ref"), []string{"hg", "--config=x=y", "lookup", "--", "-ref"}},
		{New("fossil", "cat").Option("-R", repo).Option("-r", "trunk").Operands("file"), []string{"fossil", "cat", "-R", repo, "-r=trunk", "--", "file"}},
		{New("git", "log").GitRevision("main", "--"), []string{"git", "log", "--end-of-options", "main", "--", "--"}},
		{New("git", "fetch").Operands("origin"), []string{"git", "fetch", "--end-of-options", "origin"}},
	} {
		got, err := tt.cmd.Build()
		if err != nil || !slices.Equal(got, tt.want) {
			t.Fatalf("Build() = %q, %v; want %q, nil", got, err, tt.want)
		}
		got[0] = "changed"
		if got, _ := tt.cmd.Build(); !slices.Equal(got, tt.want) {
			t.Fatalf("Build returned mutable command storage: %q", got)
		}
	}
}

func TestInvalidCommand(t *testing.T) {
	repo := filepath.Join(t.TempDir(), "repo")
	for _, cmd := range []*Command{
		{}, new(Command).Operands("file"), New(""), New("vcs", "--"), New("vcs").Add("--end-of-options"),
		New("vcs").Option("", "v"), New("vcs").Option("--x=y", "v"),
		New("vcs").Option("--", "v"), New("vcs").Option("--end-of-options", "v"),
		New("fossil").Option("-R", ""), New("fossil").Option("-R", "relative"), New("fossil").Option("-x", "v"),
		New("git").Option("-R", repo), New("git").Option("-r", "v"),
		New("git").GitRevision("--"), New("hg").GitRevision("tip"),
		New("git").Operands().Operands(), New("hg").Operands().Add("cat"),
		New("hg").Operands().Option("--rev", "tip"),
	} {
		partial, err := cmd.Build()
		if err == nil {
			t.Fatalf("Build() = %q, nil; want error", partial)
		}
		if len(partial) != 0 {
			got, nextErr := cmd.Option("-R", "").Add("later").Build()
			if nextErr != err || !slices.Equal(got, partial) {
				t.Fatalf("after error (%q, %v), got (%q, %v)", partial, err, got, nextErr)
			}
		}
	}
}
