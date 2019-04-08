// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"os/exec"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/tool"
)

var formatModes = [][]string{
	[]string{},
	[]string{"-d"},
}

func (r *runner) Format(t *testing.T, data tests.Formats) {
	for _, spn := range data {
		for _, mode := range formatModes {
			isDiff := false
			tag := "gofmt"
			for _, arg := range mode {
				tag += arg
				if arg == "-d" {
					isDiff = true
				}
			}
			uri := spn.URI()
			filename, err := uri.Filename()
			if err != nil {
				t.Fatal(err)
			}
			args := append(mode, filename)
			expect := string(r.data.Golden(tag, filename, func(golden string) error {
				cmd := exec.Command("gofmt", args...)
				buf := &bytes.Buffer{}
				cmd.Stdout = buf
				cmd.Run() // ignore error, sometimes we have intentionally ungofmt-able files
				contents := buf.String()
				// strip the unwanted diff line
				if isDiff {
					if strings.HasPrefix(contents, "diff -u") {
						if i := strings.IndexRune(contents, '\n'); i >= 0 && i < len(contents)-1 {
							contents = contents[i+1:]
						}
					}
					contents, _ = stripFileHeader(contents)
				}
				return ioutil.WriteFile(golden, []byte(contents), 0666)
			}))
			if expect == "" {
				//TODO: our error handling differs, for now just skip unformattable files
				continue
			}
			app := &cmd.Application{}
			app.Config = r.data.Config
			got := captureStdOut(t, func() {
				tool.Main(context.Background(), app, append([]string{"format"}, args...))
			})
			if isDiff {
				got, err = stripFileHeader(got)
				if err != nil {
					t.Errorf("%v: got: %v\n%v", filename, err, got)
					continue
				}
			}
			// check the first two lines are the expected file header
			if expect != got {
				t.Errorf("format failed with %#v expected:\n%s\ngot:\n%s", args, expect, got)
			}
		}
	}
}

func stripFileHeader(s string) (string, error) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "---") {
		return s, fmt.Errorf("missing original")
	}
	if i := strings.IndexRune(s, '\n'); i >= 0 && i < len(s)-1 {
		s = s[i+1:]
	} else {
		return s, fmt.Errorf("no EOL for original")
	}
	if !strings.HasPrefix(s, "+++") {
		return s, fmt.Errorf("missing output")
	}
	if i := strings.IndexRune(s, '\n'); i >= 0 && i < len(s)-1 {
		s = s[i+1:]
	} else {
		return s, fmt.Errorf("no EOL for output")
	}
	return s, nil
}
