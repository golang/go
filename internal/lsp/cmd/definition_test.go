// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
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
	guruGoDef
)

var godefModes = []godefMode{
	plainGodef,
	jsonGoDef,
	guruGoDef,
	jsonGoDef | guruGoDef,
}

func TestDefinitionHelpExample(t *testing.T) {
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
	expect := regexp.MustCompile(`^[\w/\\:_-]+flag[/\\]flag.go:\d+:\d+-\d+: defined here as type flag.FlagSet struct{.*}$`)
	for _, query := range []string{
		fmt.Sprintf("%v:%v:%v", thisFile, cmd.ExampleLine, cmd.ExampleColumn),
		fmt.Sprintf("%v:#%v", thisFile, cmd.ExampleOffset)} {
		args := append(baseArgs, query)
		got := captureStdOut(t, func() {
			tool.Main(context.Background(), &cmd.Application{}, args)
		})
		if !expect.MatchString(got) {
			t.Errorf("test with %v\nexpected:\n%s\ngot:\n%s", args, expect, got)
		}
	}
}

var brokenDefinitionTests = map[string]bool{
	// The following tests all have extra information in the description
	"A-definition-json-guru":            true,
	"err-definition-json-guru":          true,
	"myUnclosedIf-definition-json-guru": true,
	"Other-definition-json-guru":        true,
	"RandomParamY-definition-json-guru": true,
	"S1-definition-json-guru":           true,
	"S2-definition-json-guru":           true,
	"Stuff-definition-json-guru":        true,
	"Thing-definition-json-guru":        true,
	"Things-definition-json-guru":       true,
}

func (r *runner) Definition(t *testing.T, data tests.Definitions) {
	for _, d := range data {
		if d.IsType {
			// TODO: support type definition queries
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
			if mode&guruGoDef != 0 {
				if r.exporter.Name() != "GOPATH" {
					//only run guru compatability tests in GOPATH mode
					continue
				}
				if d.Name == "PackageFoo" {
					//guru does not support definition on packages
					continue
				}
				tag += "-guru"
				args = append(args, "-emulate=guru")
			}
			if _, found := brokenDefinitionTests[tag]; found {
				continue
			}
			args = append(args, "definition")
			uri := d.Src.URI()
			filename, err := uri.Filename()
			if err != nil {
				t.Fatal(err)
			}
			args = append(args, fmt.Sprint(d.Src))
			got := captureStdOut(t, func() {
				tool.Main(context.Background(), r.app, args)
			})
			got = normalizePaths(r.data, got)
			if mode&jsonGoDef != 0 && runtime.GOOS == "windows" {
				got = strings.Replace(got, "file:///", "file://", -1)
			}
			if mode&guruGoDef == 0 {
				expect := string(r.data.Golden(tag, filename, func() ([]byte, error) {
					return []byte(got), nil
				}))
				if got != expect {
					t.Errorf("definition %v failed with %#v expected:\n%s\ngot:\n%s", tag, args, expect, got)
				}
				continue
			}
			guruArgs := []string{}
			if mode&jsonGoDef != 0 {
				guruArgs = append(guruArgs, "-json")
			}
			guruArgs = append(guruArgs, "definition", fmt.Sprint(d.Src))
			expect := strings.TrimSpace(string(r.data.Golden(tag, filename, func() ([]byte, error) {
				cmd := exec.Command("guru", guruArgs...)
				cmd.Env = r.data.Exported.Config.Env
				out, _ := cmd.Output()
				if err != nil {
					if _, ok := err.(*exec.ExitError); !ok {
						return nil, fmt.Errorf("Could not run guru %v: %v\n%s", guruArgs, err, out)
					}
				}
				result := normalizePaths(r.data, string(out))
				// guru sometimes puts the full package path in type names, but we don't
				if mode&jsonGoDef == 0 && d.Name != "AImport" {
					result = strings.Replace(result, "golang.org/x/tools/internal/lsp/godef/", "", -1)
				}
				return []byte(result), nil
			})))
			if expect != "" && !strings.HasPrefix(got, expect) {
				t.Errorf("definition %v failed with %#v expected:\n%q\ngot:\n%q", tag, args, expect, got)
			}
		}
	}
}
