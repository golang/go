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
				strings.HasPrefix(env, "GOMAXPROCS=") ||
				strings.HasPrefix(env, "GORACE=") {
				continue
			}
			cmd.Env = append(cmd.Env, env)
		}
		cmd.Env = append(cmd.Env, "GORACE="+test.gorace)
		got, _ := cmd.CombinedOutput()
		if !regexp.MustCompile(test.re).MatchString(string(got)) {
			t.Fatalf("failed test case %v, expect:\n%v\ngot:\n%s",
				test.name, test.re, got)
		}
	}
}

var tests = []struct {
	name   string
	gorace string
	source string
	re     string
}{
	{"simple", "atexit_sleep_ms=0", `
package main
import "time"
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
	time.Sleep(10*time.Millisecond)
	store(x, 42)
	done <- true
}
`, `==================
WARNING: DATA RACE
Write by goroutine [0-9]:
  main\.store\(\)
      .+/main\.go:12 \+0x[0-9,a-f]+
  main\.racer\(\)
      .+/main\.go:19 \+0x[0-9,a-f]+

Previous write by main goroutine:
  main\.store\(\)
      .+/main\.go:12 \+0x[0-9,a-f]+
  main\.main\(\)
      .+/main\.go:8 \+0x[0-9,a-f]+

Goroutine [0-9] \(running\) created at:
  main\.startRacer\(\)
      .+/main\.go:15 \+0x[0-9,a-f]+
  main\.main\(\)
      .+/main\.go:7 \+0x[0-9,a-f]+
==================
Found 1 data race\(s\)
exit status 66
`},

	{"exitcode", "atexit_sleep_ms=0 exitcode=13", `
package main
func main() {
	done := make(chan bool)
	x := 0
	go func() {
		x = 42
		done <- true
	}()
	x = 43
	<-done
}
`, `exit status 13`},

	{"strip_path_prefix", "atexit_sleep_ms=0 strip_path_prefix=/main.", `
package main
func main() {
	done := make(chan bool)
	x := 0
	go func() {
		x = 42
		done <- true
	}()
	x = 43
	<-done
}
`, `
      go:7 \+0x[0-9,a-f]+
`},

	{"halt_on_error", "atexit_sleep_ms=0 halt_on_error=1", `
package main
func main() {
	done := make(chan bool)
	x := 0
	go func() {
		x = 42
		done <- true
	}()
	x = 43
	<-done
}
`, `
==================
exit status 66
`},
}
