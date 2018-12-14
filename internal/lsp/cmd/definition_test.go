// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"context"
	"flag"
	"fmt"
	"go/token"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/tool"
)

var verifyGuru = flag.Bool("verify-guru", false, "Check that the guru compatability matches")

func TestDefinitionHelpExample(t *testing.T) {
	dir, err := os.Getwd()
	if err != nil {
		t.Errorf("could not get wd: %v", err)
		return
	}
	thisFile := filepath.Join(dir, "definition.go")
	args := []string{"query", "definition", fmt.Sprintf("%v:#%v", thisFile, cmd.ExampleOffset)}
	expect := regexp.MustCompile(`^[\w/\\:_]+flag[/\\]flag.go:\d+:\d+,\d+:\d+: defined here as type flag.FlagSet struct{.*}$`)
	got := captureStdOut(t, func() {
		tool.Main(context.Background(), &cmd.Application{}, args)
	})
	if !expect.MatchString(got) {
		t.Errorf("test with %v\nexpected:\n%s\ngot:\n%s", args, expect, got)
	}
}

func TestDefinition(t *testing.T) {
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name:  "golang.org/fake",
		Files: packagestest.MustCopyFileTree("testdata"),
	}})
	defer exported.Cleanup()
	count := 0
	if err := exported.Expect(map[string]interface{}{
		"definition": func(fset *token.FileSet, src token.Pos, flags string, def packagestest.Range, match string) {
			count++
			args := []string{"query"}
			if flags != "" {
				args = append(args, strings.Split(flags, " ")...)
			}
			args = append(args, "definition")
			f := fset.File(src)
			loc := cmd.Location{
				Filename: f.Name(),
				Start: cmd.Position{
					Offset: f.Offset(src),
				},
			}
			loc.End = loc.Start
			args = append(args, fmt.Sprint(loc))
			app := &cmd.Application{}
			app.Config = *exported.Config
			got := captureStdOut(t, func() {
				tool.Main(context.Background(), app, args)
			})
			start := fset.Position(def.Start)
			end := fset.Position(def.End)
			expect := os.Expand(match, func(name string) string {
				switch name {
				case "file":
					return start.Filename
				case "efile":
					qfile := strconv.Quote(start.Filename)
					return qfile[1 : len(qfile)-1]
				case "line":
					return fmt.Sprint(start.Line)
				case "col":
					return fmt.Sprint(start.Column)
				case "offset":
					return fmt.Sprint(start.Offset)
				case "eline":
					return fmt.Sprint(end.Line)
				case "ecol":
					return fmt.Sprint(end.Column)
				case "eoffset":
					return fmt.Sprint(end.Offset)
				default:
					return name
				}
			})
			if *verifyGuru {
				var guruArgs []string
				runGuru := false
				for _, arg := range args {
					switch {
					case arg == "query":
						// just ignore this one
					case arg == "-json":
						guruArgs = append(guruArgs, arg)
					case arg == "-emulate=guru":
						// if we don't see this one we should not run guru
						runGuru = true
					case strings.HasPrefix(arg, "-"):
						// unknown flag, ignore it
						break
					default:
						guruArgs = append(guruArgs, arg)
					}
				}
				if runGuru {
					cmd := exec.Command("guru", guruArgs...)
					cmd.Env = exported.Config.Env
					out, err := cmd.CombinedOutput()
					if err != nil {
						t.Errorf("Could not run guru %v: %v\n%s", guruArgs, err, out)
					} else {
						guru := strings.TrimSpace(string(out))
						if !strings.HasPrefix(expect, guru) {
							t.Errorf("definition %v\nexpected:\n%s\nguru gave:\n%s", args, expect, guru)
						}
					}
				}
			}
			if expect != got {
				t.Errorf("definition %v\nexpected:\n%s\ngot:\n%s", args, expect, got)
			}
		},
	}); err != nil {
		t.Fatal(err)
	}
	if count == 0 {
		t.Fatalf("No tests were run")
	}
}

func captureStdOut(t testing.TB, f func()) string {
	r, out, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stdout
	defer func() {
		os.Stdout = old
		out.Close()
		r.Close()
	}()
	os.Stdout = out
	f()
	out.Close()
	data, err := ioutil.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(data))
}
