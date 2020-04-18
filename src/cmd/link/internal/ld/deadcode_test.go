// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// This example uses reflect.Value.Call, but not
// reflect.{Value,Type}.Method. This should not
// need to bring all methods live.
const deadcodeTestSrc = `
package main
import "reflect"

func f() { println("call") }

type T int
func (T) M() {}

func main() {
	v := reflect.ValueOf(f)
	v.Call(nil)
	i := interface{}(T(1))
	println(i)
}
`

func TestDeadcode(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	tmpdir, err := ioutil.TempDir("", "TestDeadcode")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	src := filepath.Join(tmpdir, "main.go")
	err = ioutil.WriteFile(src, []byte(deadcodeTestSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(tmpdir, "main.exe")

	cmd := exec.Command(testenv.GoToolPath(t), "build", "-ldflags=-dumpdep", "-o", exe, src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %v:\n%s", cmd.Args, err, out)
	}
	if bytes.Contains(out, []byte("main.T.M")) {
		t.Errorf("main.T.M should not be reachable. Output:\n%s", out)
	}
}
