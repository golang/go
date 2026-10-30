// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gotestchecker

import (
	"errors"
	"go/analysis"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"internal/analysis/gotestchecker/protocol"
)

// TestTrimmedStacks checks that the failure messages of panics in an
// analyzer, and in the encoding of its facts, show the stack of the
// analyzer (or fact type), but not that of the driver.
//
// TODO(adonovan): make this a go test script test, when we can.
func TestTrimmedStacks(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "a.go")
	if err := os.WriteFile(file, []byte("package a\n"), 0666); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		a    *analysis.Analyzer
		want string // a frame that the message must show
	}{
		{
			name: "run",
			a:    &analysis.Analyzer{Name: "panicky", Doc: "panic", Run: panicky},
			want: "gotestchecker.panicky(",
		},
		{
			name: "facts",
			a: &analysis.Analyzer{
				Name:      "facty",
				Doc:       "export a fact that cannot be encoded",
				Run:       facty,
				FactTypes: []analysis.Fact{new(badFact)},
			},
			want: "gotestchecker.(*factSet).Encode(",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			out := filepath.Join(dir, test.name+".pass")
			areq := &protocol.AnalyzeRequest{
				ID:          "a",
				ImportPath:  "a",
				Dir:         dir,
				GoFiles:     []string{file},
				GoVersion:   "go1.28",
				GOARCH:      runtime.GOARCH,
				TypeImports: map[string]string{},
				Facts:       map[int]map[string]string{},
				Outputs:     map[int]string{0: out},
				Report:      []int{0},
			}
			if err := analyze(areq, []Var{{Symbol: "a.A", Value: test.a}}, nil); err != nil {
				t.Fatal(err)
			}
			pass, err := protocol.ReadPass(out)
			if err != nil {
				t.Fatal(err)
			}
			if pass.Failure == nil {
				t.Fatal("pass did not fail")
			}
			msg := pass.Failure.Message
			if !strings.Contains(msg, test.want) {
				t.Errorf("failure message lacks %q:\n%s", test.want, msg)
			}
			for _, unwanted := range []string{
				"panic(",                 // the call of panic
				"runtime/debug.Stack(",   // the call of debug.Stack
				"gotestchecker.runPass(", // and the driver's frames
				"gotestchecker.encodeFacts(",
				"gotestchecker.analyze(",
			} {
				if strings.Contains(msg, unwanted) {
					t.Errorf("failure message contains %q:\n%s", unwanted, msg)
				}
			}
		})
	}
}

func panicky(*analysis.Pass) (any, error) { panic("oops") }

func facty(pass *analysis.Pass) (any, error) {
	pass.ExportPackageFact(new(badFact))
	return nil, nil
}

// A badFact is a fact type whose values cannot be encoded.
type badFact struct{}

func (*badFact) AFact() {}

func (*badFact) GobEncode() ([]byte, error) { return nil, errors.New("cannot encode") }

// TestPrerequisiteFailure checks that the failure of a prerequisite
// is recorded only in its own pass file, and that the dependent pass is
// merely skipped.
//
// TODO(adonovan): make this a go test script test, when we can.
func TestPrerequisiteFailure(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "a.go")
	if err := os.WriteFile(file, []byte("package a\n"), 0666); err != nil {
		t.Fatal(err)
	}
	prereq := &analysis.Analyzer{Name: "panicky", Doc: "panic", Run: panicky}
	dependent := &analysis.Analyzer{
		Name:     "dependent",
		Doc:      "require panicky",
		Requires: []*analysis.Analyzer{prereq},
		Run:      func(*analysis.Pass) (any, error) { return nil, nil },
	}
	vars := []Var{{Symbol: "a.A", Value: dependent}}
	g := newGraph(vars)
	var (
		iprereq    = slices.Index(g.analyzers, prereq)
		idependent = slices.Index(g.analyzers, dependent)
	)
	areq := &protocol.AnalyzeRequest{
		ID:          "a",
		ImportPath:  "a",
		Dir:         dir,
		GoFiles:     []string{file},
		GoVersion:   "go1.28",
		GOARCH:      runtime.GOARCH,
		TypeImports: map[string]string{},
		Facts:       map[int]map[string]string{},
		Outputs: map[int]string{
			iprereq:    filepath.Join(dir, "prereq.pass"),
			idependent: filepath.Join(dir, "dependent.pass"),
		},
		Report: []int{idependent},
	}
	if err := analyze(areq, vars, nil); err != nil {
		t.Fatal(err)
	}
	pass, err := protocol.ReadPass(areq.Outputs[iprereq])
	if err != nil {
		t.Fatal(err)
	}
	if pass.Failure == nil || !strings.Contains(pass.Failure.Message, "panic: oops") {
		t.Errorf("prerequisite pass: got failure %v, want panic", pass.Failure)
	}
	pass, err = protocol.ReadPass(areq.Outputs[idependent])
	if err != nil {
		t.Fatal(err)
	}
	if pass.Failure != nil || !pass.Skipped {
		t.Errorf("dependent pass: got failure %v, skipped %t; want skipped", pass.Failure, pass.Skipped)
	}
}
