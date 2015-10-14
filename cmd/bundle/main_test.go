// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"io/ioutil"
	"os/exec"
	"runtime"
	"testing"

	"golang.org/x/tools/go/buildutil"
)

func TestBundle(t *testing.T) {
	load := func(name string) string {
		data, err := ioutil.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}

	ctxt = buildutil.FakeContext(map[string]map[string]string{
		"initial": {
			"a.go": load("testdata/src/initial/a.go"),
			"b.go": load("testdata/src/initial/b.go"),
		},
		"fmt": {
			"print.go": `package fmt; func Println(...interface{})`,
		},
	})

	var out bytes.Buffer
	if err := bundle(&out, "initial", "dest", "prefix"); err != nil {
		t.Fatal(err)
	}
	if got, want := out.String(), load("testdata/out.golden"); got != want {
		t.Errorf("-- got --\n%s\n-- want --\n%s\n-- diff --", got, want)

		if err := ioutil.WriteFile("testdata/out.got", out.Bytes(), 0644); err != nil {
			t.Fatal(err)
		}
		t.Log(diff("testdata/out.got", "testdata/out.golden"))
	}
}

func diff(a, b string) string {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "plan9":
		cmd = exec.Command("/bin/diff", "-c", a, b)
	default:
		cmd = exec.Command("/usr/bin/diff", "-u", a, b)
	}
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	cmd.Run() // nonzero exit is expected
	if out.Len() == 0 {
		return "(failed to compute diff)"
	}
	return out.String()
}
