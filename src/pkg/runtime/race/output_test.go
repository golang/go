// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

package race_test

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestOutput(t *testing.T) {
	for _, test := range tests {
		dir, err := ioutil.TempDir("", "go-build")
		if err != nil {
			t.Fatalf("failed to create temp directory: %v", err)
		}
		defer os.RemoveAll(dir)
		src := filepath.Join(dir, "main.go")
		f, err := os.Create(src)
		if err != nil {
			t.Fatalf("failed to create file: %v", err)
		}
		_, err = f.WriteString(test.source)
		if err != nil {
			f.Close()
			t.Fatalf("failed to write: %v", err)
		}
		if err := f.Close(); err != nil {
			t.Fatalf("failed to close file: %v", err)
		}
		// Pass -l to the compiler to test stack traces.
		cmd := exec.Command("go", "run", "-race", "-gcflags=-l", src)
		// GODEBUG spoils program output, GOMAXPROCS makes it flaky.
		for _, env := range os.Environ() {
			if strings.HasPrefix(env, "GODEBUG=") ||
				strings.HasPrefix(env, "GOMAXPROCS=") {
				continue
			}
			cmd.Env = append(cmd.Env, env)
		}
		got, _ := cmd.CombinedOutput()
		if !regexp.MustCompile(test.re).MatchString(string(got)) {
			t.Fatalf("failed test case %v, expect:\n%v\ngot:\n%s",
				test.name, test.re, got)
		}
	}
}

var tests = []struct {
	name   string
	source string
	re     string
}{
	{"simple", `
package main
func main() {
	done := make(chan bool)
	x := 0
	startRacer(&x, done)
	store(&x, 43)
	<-done
}
func store(x *int, v int) {
	*x = v
}
func startRacer(x *int, done chan bool) {
	go racer(x, done)
}
func racer(x *int, done chan bool) {
	store(x, 42)
	done <- true
}
`, `==================
WARNING: DATA RACE
Write by goroutine [0-9]:
  main\.store\(\)
      .*/main\.go:11 \+0x[0-9,a-f]+
  main\.racer\(\)
      .*/main\.go:17 \+0x[0-9,a-f]+

Previous write by goroutine 1:
  main\.store\(\)
      .*/main\.go:11 \+0x[0-9,a-f]+
  main\.main\(\)
      .*/main\.go:7 \+0x[0-9,a-f]+

Goroutine 3 \(running\) created at:
  main\.startRacer\(\)
      .*/main\.go:14 \+0x[0-9,a-f]+
  main\.main\(\)
      .*/main\.go:6 \+0x[0-9,a-f]+

Goroutine 1 \(running\) created at:
  _rt0_go\(\)
      .*/src/pkg/runtime/asm_amd64\.s:[0-9]+ \+0x[0-9,a-f]+

==================
Found 1 data race\(s\)
exit status 66
`},
}
