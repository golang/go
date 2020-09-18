// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestMarkov tests the code dependency of markov.xml.
func TestMarkov(t *testing.T) {
	cmd := exec.Command("go", "run", "markov.go")
	cmd.Stdin = strings.NewReader("foo")
	cmd.Stderr = bytes.NewBuffer(nil)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, cmd.Stderr)
	}

	if !bytes.Equal(out, []byte("foo\n")) {
		t.Fatalf(`%s with input "foo" did not output "foo":\n%s`, strings.Join(cmd.Args, " "), out)
	}
}

// TestPig tests the code dependency of functions.xml.
func TestPig(t *testing.T) {
	cmd := exec.Command("go", "run", "pig.go")
	cmd.Stderr = bytes.NewBuffer(nil)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, cmd.Stderr)
	}

	const want = "Wins, losses staying at k = 100: 210/990 (21.2%), 780/990 (78.8%)\n"
	if !bytes.Contains(out, []byte(want)) {
		t.Fatalf(`%s: unexpected output\ngot:\n%s\nwant output containing:\n%s`, strings.Join(cmd.Args, " "), out, want)
	}
}

// TestURLPoll tests the code dependency of sharemem.xml.
func TestURLPoll(t *testing.T) {
	cmd := exec.Command("go", "build", "-o", os.DevNull, "urlpoll.go")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, out)
	}
}
