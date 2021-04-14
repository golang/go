// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

package race_test

import (
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

func TestOutput(t *testing.T) {
	pkgdir, err := os.MkdirTemp("", "go-build-race-output")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(pkgdir)
	out, err := exec.Command(testenv.GoToolPath(t), "install", "-race", "-pkgdir="+pkgdir, "testing").CombinedOutput()
	if err != nil {
		t.Fatalf("go install -race: %v\n%s", err, out)
	}

	for _, test := range tests {
		if test.goos != "" && test.goos != runtime.GOOS {
			t.Logf("test %v runs only on %v, skipping: ", test.name, test.goos)
			continue
		}
		dir, err := os.MkdirTemp("", "go-build")
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

		cmd := exec.Command(testenv.GoToolPath(t), test.run, "-race", "-pkgdir="+pkgdir, src)
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
	_ = x
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
.*main_test.go:14: true
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

	// Test for https://golang.org/issue/33309
	{"midstack_inlining_traceback", "run", "linux", "atexit_sleep_ms=0", `
package main

var x int

func main() {
	c := make(chan int)
	go f(c)
	x = 1
	<-c
}

func f(c chan int) {
	g(c)
}

func g(c chan int) {
	h(c)
}

func h(c chan int) {
	c <- x
}
`, `==================
WARNING: DATA RACE
Read at 0x[0-9,a-f]+ by goroutine [0-9]:
  main\.h\(\)
      .+/main\.go:22 \+0x[0-9,a-f]+
  main\.g\(\)
      .+/main\.go:18 \+0x[0-9,a-f]+
  main\.f\(\)
      .+/main\.go:14 \+0x[0-9,a-f]+

Previous write at 0x[0-9,a-f]+ by main goroutine:
  main\.main\(\)
      .+/main\.go:9 \+0x[0-9,a-f]+

Goroutine [0-9] \(running\) created at:
  main\.main\(\)
      .+/main\.go:8 \+0x[0-9,a-f]+
==================
Found 1 data race\(s\)
exit status 66
`},

	// Test for https://golang.org/issue/17190
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

var done chan bool
var racy int

//export goCallback
func goCallback() {
	racy++
	done <- true
}

func main() {
	done = make(chan bool)
	var c C.cb
	C.startThread(&c)
	racy++
	<- done
}
`, `==================
WARNING: DATA RACE
Read at 0x[0-9,a-f]+ by .*:
  main\..*
      .*/main\.go:[0-9]+ \+0x[0-9,a-f]+(?s).*

Previous write at 0x[0-9,a-f]+ by .*:
  main\..*
      .*/main\.go:[0-9]+ \+0x[0-9,a-f]+(?s).*

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
	_ = x
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
	{"mutex", "run", "", "atexit_sleep_ms=0", `
package main
import (
	"sync"
	"fmt"
)
func main() {
	c := make(chan bool, 1)
	threads := 1
	iterations := 20000
	data := 0
	var wg sync.WaitGroup
	for i := 0; i < threads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				c <- true
				data += 1
				<- c
			}
		}()
	}
	for i := 0; i < iterations; i++ {
		c <- true
		data += 1
		<- c
	}
	wg.Wait()
	if (data == iterations*(threads+1)) { fmt.Println("pass") }
}`, `pass`},
	// Test for https://github.com/golang/go/issues/37355
	{"chanmm", "run", "", "atexit_sleep_ms=0", `
package main
import (
	"sync"
	"time"
)
func main() {
	c := make(chan bool, 1)
	var data uint64
	var wg sync.WaitGroup
	wg.Add(2)
	c <- true
	go func() {
		defer wg.Done()
		c <- true
	}()
	go func() {
		defer wg.Done()
		time.Sleep(time.Second)
		<-c
		data = 2
	}()
	data = 1
	<-c
	wg.Wait()
	_ = data
}
`, `==================
WARNING: DATA RACE
Write at 0x[0-9,a-f]+ by goroutine [0-9]:
  main\.main\.func2\(\)
      .*/main\.go:21 \+0x[0-9,a-f]+

Previous write at 0x[0-9,a-f]+ by main goroutine:
  main\.main\(\)
      .*/main\.go:23 \+0x[0-9,a-f]+

Goroutine [0-9] \(running\) created at:
  main\.main\(\)
      .*/main.go:[0-9]+ \+0x[0-9,a-f]+
==================`},
}
