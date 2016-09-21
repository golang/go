// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var (
	sub = flag.Bool("subtest", false, "Indicates that the program should act as the eg command rather than as a test")
)

func TestVendor(t *testing.T) {
	if *sub {
		if err := doMain(); err != nil {
			fmt.Fprintf(os.Stderr, "eg: %s\n", err)
			os.Exit(1)
		}
		os.Exit(0)
	}

	tests := []struct {
		args     []string
		goldFile string
	}{
		{
			args:     []string{"-t", "A.template", "--", "A1.go"},
			goldFile: "A1.golden",
		},
	}

	curdir, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd: %v", err)
	}

	for _, tt := range tests {
		cmd := exec.Command(os.Args[0], append([]string{"-test.run=TestVendor", "-subtest"}, tt.args...)...)
		cmd.Dir = filepath.Join(curdir, "testdata/src/prog")

		cmd.Env = append(cmd.Env, "GOPATH="+filepath.Join(curdir, "testdata"))
		for _, env := range os.Environ() {
			if strings.HasPrefix(env, "GOPATH=") {
				continue
			}
			cmd.Env = append(cmd.Env, env)
		}

		goldFile := filepath.Join(cmd.Dir, tt.goldFile)
		gold, err := ioutil.ReadFile(goldFile)
		if err != nil {
			t.Errorf("read golden file %q: %v", goldFile, err)
			continue
		}

		var stdout bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = os.Stderr
		err = cmd.Run()
		if err != nil {
			t.Errorf("eg %q exec: %v", tt.args, err)
			continue
		}

		if have, want := stdout.Bytes(), gold; !bytes.Equal(have, want) {
			t.Errorf("eg %q output does not match contents of %q:\n%s\n!=\n%s\n", tt.args, tt.goldFile, have, want)
		}
	}
}
