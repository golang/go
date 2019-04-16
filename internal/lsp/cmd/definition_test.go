// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
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

var verifyGuru = flag.Bool("verify-guru", false, "Check that the guru compatability matches")

func TestDefinitionHelpExample(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("not all source files are available on android")
	}
	if runtime.GOOS != "linux" || isRace {
		t.Skip("currently uses too much memory, see issue #31611")
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

func (r *runner) Definition(t *testing.T, data tests.Definitions) {
	for _, d := range data {
		if d.IsType {
			// TODO: support type definition queries
			continue
		}
		args := []string{"query"}
		if d.Flags != "" {
			args = append(args, strings.Split(d.Flags, " ")...)
		}
		args = append(args, "definition")
		src := span.New(d.Src.URI(), span.NewPoint(0, 0, d.Src.Start().Offset()), span.Point{})
		args = append(args, fmt.Sprint(src))
		got := captureStdOut(t, func() {
			tool.Main(context.Background(), r.app, args)
		})
		pattern := newPattern(d.Match, d.Def)
		if !pattern.matches(got) {
			t.Errorf("definition %v\nexpected:\n%s\ngot:\n%s", args, pattern, got)
		}
		if *verifyGuru {
			moduleMode := r.data.Exported.File(r.data.Exported.Modules[0].Name, "go.mod") != ""
			var guruArgs []string
			runGuru := false
			if !moduleMode {
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
			}
			if runGuru {
				cmd := exec.Command("guru", guruArgs...)
				cmd.Env = r.data.Exported.Config.Env
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Errorf("Could not run guru %v: %v\n%s", guruArgs, err, out)
				} else {
					guru := strings.TrimSpace(string(out))
					if !pattern.matches(guru) {
						t.Errorf("definition %v\nexpected:\n%s\nguru gave:\n%s", args, pattern, guru)
					}
				}
			}
		}
	}
}

type pattern struct {
	raw      string
	expanded []string
	matchAll bool
}

func newPattern(s string, def span.Span) pattern {
	p := pattern{raw: s}
	if s == "" {
		p.expanded = []string{fmt.Sprintf("%v: ", def)}
		return p
	}
	p.matchAll = strings.HasSuffix(s, "$$")
	for _, fragment := range strings.Split(s, "$$") {
		p.expanded = append(p.expanded, os.Expand(fragment, func(name string) string {
			switch name {
			case "file":
				fname, _ := def.URI().Filename()
				return fname
			case "efile":
				fname, _ := def.URI().Filename()
				qfile := strconv.Quote(fname)
				return qfile[1 : len(qfile)-1]
			case "euri":
				quri := strconv.Quote(string(def.URI()))
				return quri[1 : len(quri)-1]
			case "line":
				return fmt.Sprint(def.Start().Line())
			case "col":
				return fmt.Sprint(def.Start().Column())
			case "offset":
				return fmt.Sprint(def.Start().Offset())
			case "eline":
				return fmt.Sprint(def.End().Line())
			case "ecol":
				return fmt.Sprint(def.End().Column())
			case "eoffset":
				return fmt.Sprint(def.End().Offset())
			default:
				return name
			}
		}))
	}
	return p
}

func (p pattern) String() string {
	return strings.Join(p.expanded, "$$")
}

func (p pattern) matches(s string) bool {
	if len(p.expanded) == 0 {
		return false
	}
	if !strings.HasPrefix(s, p.expanded[0]) {
		return false
	}
	remains := s[len(p.expanded[0]):]
	for _, fragment := range p.expanded[1:] {
		i := strings.Index(remains, fragment)
		if i < 0 {
			return false
		}
		remains = remains[i+len(fragment):]
	}
	if !p.matchAll {
		return true
	}
	return len(remains) == 0
}
