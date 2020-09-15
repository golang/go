// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"io/ioutil"
	"os"
	"os/exec"
	"runtime"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
)

func TestBundle(t *testing.T) { packagestest.TestAll(t, testBundle) }
func testBundle(t *testing.T, x packagestest.Exporter) {
	load := func(name string) string {
		data, err := ioutil.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}

	e := packagestest.Export(t, x, []packagestest.Module{
		{
			Name: "initial",
			Files: map[string]interface{}{
				"a.go": load("testdata/src/initial/a.go"),
				"b.go": load("testdata/src/initial/b.go"),
				"c.go": load("testdata/src/initial/c.go"),
			},
		},
		{
			Name: "domain.name/importdecl",
			Files: map[string]interface{}{
				"p.go": load("testdata/src/domain.name/importdecl/p.go"),
			},
		},
	})
	defer e.Cleanup()
	testingOnlyPackagesConfig = e.Config

	os.Args = os.Args[:1] // avoid e.g. -test=short in the output
	out, err := bundle("initial", "github.com/dest", "dest", "prefix", "tag")
	if err != nil {
		t.Fatal(err)
	}

	if got, want := string(out), load("testdata/out.golden"); got != want {
		t.Errorf("-- got --\n%s\n-- want --\n%s\n-- diff --", got, want)

		if err := ioutil.WriteFile("testdata/out.got", out, 0644); err != nil {
			t.Fatal(err)
		}
		t.Log(diff("testdata/out.golden", "testdata/out.got"))
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
