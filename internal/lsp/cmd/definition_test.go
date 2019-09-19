// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

const (
	expectedDefinitionsCount     = 28
	expectedTypeDefinitionsCount = 2
)

type godefMode int

const (
	plainGodef = godefMode(1 << iota)
	jsonGoDef
)

var godefModes = []godefMode{
	plainGodef,
	jsonGoDef,
}

func TestDefinitionHelpExample(t *testing.T) {
	// TODO: https://golang.org/issue/32794.
	t.Skip()
	if runtime.GOOS == "android" {
		t.Skip("not all source files are available on android")
	}
	dir, err := os.Getwd()
	if err != nil {
		t.Errorf("could not get wd: %v", err)
		return
	}
	thisFile := filepath.Join(dir, "definition.go")
	baseArgs := []string{"query", "definition"}
	expect := regexp.MustCompile(`(?s)^[\w/\\:_-]+flag[/\\]flag.go:\d+:\d+-\d+: defined here as FlagSet struct {.*}$`)
	for _, query := range []string{
		fmt.Sprintf("%v:%v:%v", thisFile, cmd.ExampleLine, cmd.ExampleColumn),
		fmt.Sprintf("%v:#%v", thisFile, cmd.ExampleOffset)} {
		args := append(baseArgs, query)
		got := captureStdOut(t, func() {
			_ = tool.Run(tests.Context(t), cmd.New("gopls-test", "", nil), args)
		})
		if !expect.MatchString(got) {
			t.Errorf("test with %v\nexpected:\n%s\ngot:\n%s", args, expect, got)
		}
	}
}

func (r *runner) Definition(t *testing.T, data tests.Definitions) {
	// TODO: https://golang.org/issue/32794.
	t.Skip()
	for _, d := range data {
		if d.IsType || d.OnlyHover {
			// TODO: support type definition, hover queries
			continue
		}
		d.Src = span.New(d.Src.URI(), span.NewPoint(0, 0, d.Src.Start().Offset()), span.Point{})
		for _, mode := range godefModes {
			args := []string{"-remote=internal", "query"}
			tag := d.Name + "-definition"
			if mode&jsonGoDef != 0 {
				tag += "-json"
				args = append(args, "-json")
			}
			args = append(args, "definition")
			uri := d.Src.URI()
			args = append(args, fmt.Sprint(d.Src))
			got := captureStdOut(t, func() {
				app := cmd.New("gopls-test", r.data.Config.Dir, r.data.Exported.Config.Env)
				_ = tool.Run(r.ctx, app, args)
			})
			got = normalizePaths(r.data, got)
			if mode&jsonGoDef != 0 && runtime.GOOS == "windows" {
				got = strings.Replace(got, "file:///", "file://", -1)
			}
			expect := strings.TrimSpace(string(r.data.Golden(tag, uri.Filename(), func() ([]byte, error) {
				return []byte(got), nil
			})))
			if expect != "" && !strings.HasPrefix(got, expect) {
				t.Errorf("definition %v failed with %#v expected:\n%q\ngot:\n%q", tag, args, expect, got)
			}
		}
	}
}
