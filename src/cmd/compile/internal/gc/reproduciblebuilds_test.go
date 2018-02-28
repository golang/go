// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc_test

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestReproducibleBuilds(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	iters := 10
	if testing.Short() {
		iters = 4
	}
	t.Parallel()
	var want []byte
	tmp, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("temp file creation failed: %v", err)
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()
	for i := 0; i < iters; i++ {
		out, err := exec.Command(testenv.GoToolPath(t), "tool", "compile", "-o", tmp.Name(), filepath.Join("testdata", "reproducible", "issue20272.go")).CombinedOutput()
		if err != nil {
			t.Fatalf("failed to compile: %v\n%s", err, out)
		}
		obj, err := ioutil.ReadFile(tmp.Name())
		if err != nil {
			t.Fatalf("failed to read object file: %v", err)
		}
		if i == 0 {
			want = obj
		} else {
			if !bytes.Equal(want, obj) {
				t.Fatalf("builds produced different output after %d iters (%d bytes vs %d bytes)", i, len(want), len(obj))
			}
		}
	}
}
