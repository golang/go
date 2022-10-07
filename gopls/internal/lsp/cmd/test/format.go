// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"bytes"
	"io/ioutil"
	"os"
	"regexp"
	"strings"
	"testing"

	exec "golang.org/x/sys/execabs"

	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/testenv"
)

func (r *runner) Format(t *testing.T, spn span.Span) {
	tag := "gofmt"
	uri := spn.URI()
	filename := uri.Filename()
	expect := string(r.data.Golden(t, tag, filename, func() ([]byte, error) {
		cmd := exec.Command("gofmt", filename)
		contents, _ := cmd.Output() // ignore error, sometimes we have intentionally ungofmt-able files
		contents = []byte(r.Normalize(fixFileHeader(string(contents))))
		return contents, nil
	}))
	if expect == "" {
		//TODO: our error handling differs, for now just skip unformattable files
		t.Skip("Unformattable file")
	}
	got, _ := r.NormalizeGoplsCmd(t, "format", filename)
	if expect != got {
		t.Errorf("format failed for %s expected:\n%s\ngot:\n%s", filename, expect, got)
	}
	// now check we can build a valid unified diff
	unified, _ := r.NormalizeGoplsCmd(t, "format", "-d", filename)
	checkUnified(t, filename, expect, unified)
}

var unifiedHeader = regexp.MustCompile(`^diff -u.*\n(---\s+\S+\.go\.orig)\s+[\d-:. ]+(\n\+\+\+\s+\S+\.go)\s+[\d-:. ]+(\n@@)`)

func fixFileHeader(s string) string {
	match := unifiedHeader.FindStringSubmatch(s)
	if match == nil {
		return s
	}
	return strings.Join(append(match[1:], s[len(match[0]):]), "")
}

func checkUnified(t *testing.T, filename string, expect string, patch string) {
	testenv.NeedsTool(t, "patch")
	if strings.Count(patch, "\n+++ ") > 1 {
		// TODO(golang/go/#34580)
		t.Skip("multi-file patch tests not supported yet")
	}
	applied := ""
	if patch == "" {
		applied = expect
	} else {
		temp, err := ioutil.TempFile("", "applied")
		if err != nil {
			t.Fatal(err)
		}
		temp.Close()
		defer os.Remove(temp.Name())
		cmd := exec.Command("patch", "-u", "-p0", "-o", temp.Name(), filename)
		cmd.Stdin = bytes.NewBuffer([]byte(patch))
		msg, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("failed applying patch to %s: %v\ngot:\n%s\npatch:\n%s", filename, err, msg, patch)
			return
		}
		out, err := ioutil.ReadFile(temp.Name())
		if err != nil {
			t.Errorf("failed reading patched output for %s: %v\n", filename, err)
			return
		}
		applied = string(out)
	}
	if expect != applied {
		t.Errorf("apply unified gave wrong result for %s expected:\n%s\ngot:\n%s\npatch:\n%s", filename, expect, applied, patch)
	}
}
