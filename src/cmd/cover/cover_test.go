// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

const (
	// Data directory, also the package directory for the test.
	testdata = "testdata"

	// Binaries we compile.
	testcover = "./testcover.exe"
)

var (
	// Files we use.
	testMain     = filepath.Join(testdata, "main.go")
	testTest     = filepath.Join(testdata, "test.go")
	coverInput   = filepath.Join(testdata, "test_line.go")
	coverOutput  = filepath.Join(testdata, "test_cover.go")
	coverProfile = filepath.Join(testdata, "profile.cov")
)

var debug = flag.Bool("debug", false, "keep rewritten files for debugging")

// Run this shell script, but do it in Go so it can be run by "go test".
//
//	replace the word LINE with the line number < testdata/test.go > testdata/test_line.go
// 	go build -o ./testcover
// 	./testcover -mode=count -var=CoverTest -o ./testdata/test_cover.go testdata/test_line.go
//	go run ./testdata/main.go ./testdata/test.go
//
func TestCover(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// Read in the test file (testTest) and write it, with LINEs specified, to coverInput.
	file, err := ioutil.ReadFile(testTest)
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(file, []byte("\n"))
	for i, line := range lines {
		lines[i] = bytes.Replace(line, []byte("LINE"), []byte(fmt.Sprint(i+1)), -1)
	}
	if err := ioutil.WriteFile(coverInput, bytes.Join(lines, []byte("\n")), 0666); err != nil {
		t.Fatal(err)
	}

	// defer removal of test_line.go
	if !*debug {
		defer os.Remove(coverInput)
	}

	// go build -o testcover
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", testcover)
	run(cmd, t)

	// defer removal of testcover
	defer os.Remove(testcover)

	// ./testcover -mode=count -var=thisNameMustBeVeryLongToCauseOverflowOfCounterIncrementStatementOntoNextLineForTest -o ./testdata/test_cover.go testdata/test_line.go
	cmd = exec.Command(testcover, "-mode=count", "-var=thisNameMustBeVeryLongToCauseOverflowOfCounterIncrementStatementOntoNextLineForTest", "-o", coverOutput, coverInput)
	run(cmd, t)

	// defer removal of ./testdata/test_cover.go
	if !*debug {
		defer os.Remove(coverOutput)
	}

	// go run ./testdata/main.go ./testdata/test.go
	cmd = exec.Command(testenv.GoToolPath(t), "run", testMain, coverOutput)
	run(cmd, t)

	file, err = ioutil.ReadFile(coverOutput)
	if err != nil {
		t.Fatal(err)
	}
	// compiler directive must appear right next to function declaration.
	if got, err := regexp.MatchString(".*\n//go:nosplit\nfunc someFunction().*", string(file)); err != nil || !got {
		t.Error("misplaced compiler directive")
	}
	// "go:linkname" compiler directive should be present.
	if got, err := regexp.MatchString(`.*go\:linkname some\_name some\_name.*`, string(file)); err != nil || !got {
		t.Error("'go:linkname' compiler directive not found")
	}

	// Other comments should be preserved too.
	c := ".*// This comment didn't appear in generated go code.*"
	if got, err := regexp.MatchString(c, string(file)); err != nil || !got {
		t.Errorf("non compiler directive comment %q not found", c)
	}
}

// TestDirectives checks that compiler directives are preserved and positioned
// correctly. Directives that occur before top-level declarations should remain
// above those declarations, even if they are not part of the block of
// documentation comments.
func TestDirectives(t *testing.T) {
	// Read the source file and find all the directives. We'll keep
	// track of whether each one has been seen in the output.
	testDirectives := filepath.Join(testdata, "directives.go")
	source, err := ioutil.ReadFile(testDirectives)
	if err != nil {
		t.Fatal(err)
	}
	sourceDirectives := findDirectives(source)

	// go tool cover -mode=atomic ./testdata/directives.go
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "cover", "-mode=atomic", testDirectives)
	cmd.Stderr = os.Stderr
	output, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}

	// Check that all directives are present in the output.
	outputDirectives := findDirectives(output)
	foundDirective := make(map[string]bool)
	for _, p := range sourceDirectives {
		foundDirective[p.name] = false
	}
	for _, p := range outputDirectives {
		if found, ok := foundDirective[p.name]; !ok {
			t.Errorf("unexpected directive in output: %s", p.text)
		} else if found {
			t.Errorf("directive found multiple times in output: %s", p.text)
		}
		foundDirective[p.name] = true
	}
	for name, found := range foundDirective {
		if !found {
			t.Errorf("missing directive: %s", name)
		}
	}

	// Check that directives that start with the name of top-level declarations
	// come before the beginning of the named declaration and after the end
	// of the previous declaration.
	fset := token.NewFileSet()
	astFile, err := parser.ParseFile(fset, testDirectives, output, 0)
	if err != nil {
		t.Fatal(err)
	}

	prevEnd := 0
	for _, decl := range astFile.Decls {
		var name string
		switch d := decl.(type) {
		case *ast.FuncDecl:
			name = d.Name.Name
		case *ast.GenDecl:
			if len(d.Specs) == 0 {
				// An empty group declaration. We still want to check that
				// directives can be associated with it, so we make up a name
				// to match directives in the test data.
				name = "_empty"
			} else if spec, ok := d.Specs[0].(*ast.TypeSpec); ok {
				name = spec.Name.Name
			}
		}
		pos := fset.Position(decl.Pos()).Offset
		end := fset.Position(decl.End()).Offset
		if name == "" {
			prevEnd = end
			continue
		}
		for _, p := range outputDirectives {
			if !strings.HasPrefix(p.name, name) {
				continue
			}
			if p.offset < prevEnd || pos < p.offset {
				t.Errorf("directive %s does not appear before definition %s", p.text, name)
			}
		}
		prevEnd = end
	}
}

type directiveInfo struct {
	text   string // full text of the comment, not including newline
	name   string // text after //go:
	offset int    // byte offset of first slash in comment
}

func findDirectives(source []byte) []directiveInfo {
	var directives []directiveInfo
	directivePrefix := []byte("\n//go:")
	offset := 0
	for {
		i := bytes.Index(source[offset:], directivePrefix)
		if i < 0 {
			break
		}
		i++ // skip newline
		p := source[offset+i:]
		j := bytes.IndexByte(p, '\n')
		if j < 0 {
			// reached EOF
			j = len(p)
		}
		directive := directiveInfo{
			text:   string(p[:j]),
			name:   string(p[len(directivePrefix)-1 : j]),
			offset: offset + i,
		}
		directives = append(directives, directive)
		offset += i + j
	}
	return directives
}

// Makes sure that `cover -func=profile.cov` reports accurate coverage.
// Issue #20515.
func TestCoverFunc(t *testing.T) {
	// go tool cover -func ./testdata/profile.cov
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "cover", "-func", coverProfile)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			t.Logf("%s", ee.Stderr)
		}
		t.Fatal(err)
	}

	if got, err := regexp.Match(".*total:.*100.0.*", out); err != nil || !got {
		t.Logf("%s", out)
		t.Errorf("invalid coverage counts. got=(%v, %v); want=(true; nil)", got, err)
	}
}

func run(c *exec.Cmd, t *testing.T) {
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	err := c.Run()
	if err != nil {
		t.Fatal(err)
	}
}
