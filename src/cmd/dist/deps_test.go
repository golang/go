// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestDeps(t *testing.T) {
	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in short mode")
	}

	current, err := ioutil.ReadFile("deps.go")
	if err != nil {
		t.Fatal(err)
	}

	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skipf("skipping because bash not found: %v", err)
	}

	outf, err := ioutil.TempFile("", "dist-deps-test")
	if err != nil {
		t.Fatal(err)
	}
	outf.Close()
	outname := outf.Name()
	defer os.Remove(outname)

	out, err := exec.Command(bash, "mkdeps.bash", outname).CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s", out)

	updated, err := ioutil.ReadFile(outname)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(current, updated) {
		// Very simple minded diff.
		t.Log("-current +generated")
		clines := strings.Split(string(current), "\n")
		for i, line := range clines {
			clines[i] = strings.Join(strings.Fields(line), " ")
		}
		ulines := strings.Split(string(updated), "\n")
		for i, line := range ulines {
			ulines[i] = strings.Join(strings.Fields(line), " ")
		}
		for len(clines) > 0 {
			cl := clines[0]
			switch {
			case len(ulines) == 0:
				t.Logf("-%s", cl)
				clines = clines[1:]
			case cl == ulines[0]:
				clines = clines[1:]
				ulines = ulines[1:]
			case pkg(cl) == pkg(ulines[0]):
				t.Logf("-%s", cl)
				t.Logf("+%s", ulines[0])
				clines = clines[1:]
				ulines = ulines[1:]
			case pkg(cl) < pkg(ulines[0]):
				t.Logf("-%s", cl)
				clines = clines[1:]
			default:
				cp := pkg(cl)
				for len(ulines) > 0 && pkg(ulines[0]) < cp {
					t.Logf("+%s", ulines[0])
					ulines = ulines[1:]
				}
			}
		}

		t.Error("cmd/dist/deps.go is out of date; run cmd/dist/mkdeps.bash")
	}
}

// pkg returns the package of a line in deps.go.
func pkg(line string) string {
	i := strings.Index(line, `"`)
	if i < 0 {
		return ""
	}
	line = line[i+1:]
	i = strings.Index(line, `"`)
	if i < 0 {
		return ""
	}
	return line[:i]
}
