// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

const aliasSrc = `
package x

type T = int
`

func TestInvalidLang(t *testing.T) {
	t.Parallel()

	testenv.MustHaveGoBuild(t)

	dir, err := ioutil.TempDir("", "TestInvalidLang")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, "alias.go")
	if err := ioutil.WriteFile(src, []byte(aliasSrc), 0644); err != nil {
		t.Fatal(err)
	}

	outfile := filepath.Join(dir, "alias.o")

	if testLang(t, "go9.99", src, outfile) == nil {
		t.Error("compilation with -lang=go9.99 succeeded unexpectedly")
	}

	// This test will have to be adjusted if we ever reach 1.99 or 2.0.
	if testLang(t, "go1.99", src, outfile) == nil {
		t.Error("compilation with -lang=go1.99 succeeded unexpectedly")
	}

	if testLang(t, "go1.8", src, outfile) == nil {
		t.Error("compilation with -lang=go1.8 succeeded unexpectedly")
	}

	if err := testLang(t, "go1.9", src, outfile); err != nil {
		t.Errorf("compilation with -lang=go1.9 failed unexpectedly: %v", err)
	}
}

func testLang(t *testing.T, lang, src, outfile string) error {
	run := []string{testenv.GoToolPath(t), "tool", "compile", "-lang", lang, "-o", outfile, src}
	t.Log(run)
	out, err := exec.Command(run[0], run[1:]...).CombinedOutput()
	t.Logf("%s", out)
	return err
}
