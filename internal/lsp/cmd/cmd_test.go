// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/lsp/tests"
)

var isRace = false

type runner struct {
	data *tests.Data
	app  *cmd.Application
}

func TestCommandLine(t *testing.T) {
	packagestest.TestAll(t, testCommandLine)
}

func testCommandLine(t *testing.T, exporter packagestest.Exporter) {
	data := tests.Load(t, exporter, "../testdata")
	defer data.Exported.Cleanup()

	r := &runner{
		data: data,
		app: &cmd.Application{
			Config: *data.Exported.Config,
		},
	}
	tests.Run(t, r, data)
}

func (r *runner) Completion(t *testing.T, data tests.Completions, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) Format(t *testing.T, data tests.Formats) {
	//TODO: add command line formatting tests when it works
}

func (r *runner) Highlight(t *testing.T, data tests.Highlights) {
	//TODO: add command line highlight tests when it works
}
func (r *runner) Symbol(t *testing.T, data tests.Symbols) {
	//TODO: add command line symbol tests when it works
}

func (r *runner) Signature(t *testing.T, data tests.Signatures) {
	//TODO: add command line signature tests when it works
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
