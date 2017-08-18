// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

package race_test

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

func TestOutput(t *testing.T) {
	for _, test := range tests {
		if test.goos != "" && test.goos != runtime.GOOS {
			t.Logf("test %v runs only on %v, skipping: ", test.name, test.goos)
			continue
		}
		dir, err := ioutil.TempDir("", "go-build")
		if err != nil {
			t.Fatalf("failed to create temp directory: %v", err)
		}
		defer os.RemoveAll(dir)
		source := "main.go"
		if test.run == "test" {
			source = "main_test.go"
		}
		src := filepath.Join(dir, source)
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
		cmd := exec.Command(testenv.GoToolPath(t), test.run, "-race", "-gcflags=-l", src)
		// GODEBUG spoils program output, GOMAXPROCS makes it flaky.
		for _, env := range os.Environ() {
			if strings.HasPrefix(env, "GODEBUG=") ||
				strings.HasPrefix(env, "GOMAXPROCS=") ||
				strings.HasPrefix(env, "GORACE=") {
				continue
			}
			cmd.Env = append(cmd.Env, env)
		}
		cmd.Env = append(cmd.Env,
			"GOMAXPROCS=1", // see comment in race_test.go
			"GORACE="+test.gorace,
		)
		got, _ := cmd.CombinedOutput()
		if !regexp.MustCompile(test.re).MatchString(string(got)) {
			t.Fatalf("failed test case %v, expect:\n%v\ngot:\n%s",
				test.name, test.re, got)
		}
	}
}

var tests = []struct {
	name   string
	run    string
	goos   string
	gorace string
	source string
	re     string
}{
	{"simple", "run", "", "atexit_sleep_ms=0", `
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
Write at 0x[0-9,a-f]+ by goroutine [0-9]:
  main\.store\(\)
      .+/main\.go:12 \+0x[0-9,a-f]+
  main\.racer\(\)
      .+/main\.go:19 \+0x[0-9,a-f]+

Previous write at 0x[0-9,a-f]+ by main goroutine:
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

	{"exitcode", "run", "", "atexit_sleep_ms=0 exitcode=13", `
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

	{"strip_path_prefix", "run", "", "atexit_sleep_ms=0 strip_path_prefix=/main.", `
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

	{"halt_on_error", "run", "", "atexit_sleep_ms=0 halt_on_error=1", `
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

	{"test_fails_on_race", "test", "", "atexit_sleep_ms=0", `
package main_test
import "testing"
func TestFail(t *testing.T) {
	done := make(chan bool)
	x := 0
	go func() {
		x = 42
		done <- true
	}()
	x = 43
	<-done
	t.Log(t.Failed())
}
`, `
==================
--- FAIL: TestFail \(0...s\)
.*main_test.go:13: true
.*testing.go:.*: race detected during execution of test
FAIL`},

	{"slicebytetostring_pc", "run", "", "atexit_sleep_ms=0", `
package main
func main() {
	done := make(chan string)
	data := make([]byte, 10)
	go func() {
		done <- string(data)
	}()
	data[0] = 1
	<-done
}
`, `
  runtime\.slicebytetostring\(\)
      .*/runtime/string\.go:.*
  main\.main\.func1\(\)
      .*/main.go:7`},

	// Test for http://golang.org/issue/17190
	{"external_cgo_thread", "run", "linux", "atexit_sleep_ms=0", `
package main

/*
#include <pthread.h>
typedef struct cb {
        int foo;
} cb;
extern void goCallback();
static inline void *threadFunc(void *p) {
	goCallback();
	return 0;
}
static inline void startThread(cb* c) {
	pthread_t th;
	pthread_create(&th, 0, threadFunc, 0);
}
*/
import "C"

import "time"

var racy int

//export goCallback
func goCallback() {
	racy++
}

func main() {
	var c C.cb
	C.startThread(&c)
	time.Sleep(time.Second)
	racy++
}
`, `==================
WARNING: DATA RACE
Read at 0x[0-9,a-f]+ by main goroutine:
  main\.main\(\)
      .*/main\.go:34 \+0x[0-9,a-f]+

Previous write at 0x[0-9,a-f]+ by goroutine [0-9]:
  main\.goCallback\(\)
      .*/main\.go:27 \+0x[0-9,a-f]+
  main._cgoexpwrap_[0-9a-z]+_goCallback\(\)
      .*/_cgo_gotypes\.go:[0-9]+ \+0x[0-9,a-f]+

Goroutine [0-9] \(running\) created at:
  runtime\.newextram\(\)
      .*/runtime/proc.go:[0-9]+ \+0x[0-9,a-f]+
==================`},
	{"second_test_passes", "test", "", "atexit_sleep_ms=0", `
package main_test
import "testing"
func TestFail(t *testing.T) {
	done := make(chan bool)
	x := 0
	go func() {
		x = 42
		done <- true
	}()
	x = 43
	<-done
}

func TestPass(t *testing.T) {
}
`, `
==================
--- FAIL: TestFail \(0...s\)
.*testing.go:.*: race detected during execution of test
FAIL`},
}
