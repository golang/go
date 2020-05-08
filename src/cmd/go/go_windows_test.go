// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"cmd/go/internal/robustio"
)

func TestAbsolutePath(t *testing.T) {
	t.Parallel()

	tmp, err := ioutil.TempDir("", "TestAbsolutePath")
	if err != nil {
		t.Fatal(err)
	}
	defer robustio.RemoveAll(tmp)

	file := filepath.Join(tmp, "a.go")
	err = ioutil.WriteFile(file, []byte{}, 0644)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(tmp, "dir")
	err = os.Mkdir(dir, 0777)
	if err != nil {
		t.Fatal(err)
	}

	noVolume := file[len(filepath.VolumeName(file)):]
	wrongPath := filepath.Join(dir, noVolume)
	cmd := exec.Command(testenv.GoToolPath(t), "build", noVolume)
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("build should fail")
	}
	if strings.Contains(string(output), wrongPath) {
		t.Fatalf("wrong output found: %v %v", err, string(output))
	}
}
