// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc_test

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os/exec"
	"testing"
)

func TestBuiltin(t *testing.T) {
	testenv.MustHaveGoRun(t)

	old, err := ioutil.ReadFile("builtin.go")
	if err != nil {
		t.Fatal(err)
	}

	new, err := exec.Command("go", "run", "mkbuiltin.go", "-stdout").Output()
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(old, new) {
		t.Fatal("builtin.go out of date; run mkbuiltin.go")
	}
}
