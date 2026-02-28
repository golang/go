// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

package race_test

import (
	"fmt"
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
	pkgdir := t.TempDir()
	out, err := exec.Command(testenv.GoToolPath(t), "install", "-race", "-pkgdir="+pkgdir, "testing").CombinedOutput()
	if err != nil {
		t.Fatalf("go install -race: %v\n%s", err, out)
	}

	for _, test := range tests {
		if test.goos != "" && test.goos != runtime.GOOS {
			t.Logf("test %v runs only on %v, skipping: ", test.name, test.goos)
			continue
		}
		dir := t.TempDir()
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
		matched := false
		for _, re := range test.re {
			if regexp.MustCompile(re).MatchString(string(got)) {
				matched = true
				break
			}
		}
		if !matched {
			exp := fmt.Sprintf("expect:\n%v\n", test.re[0])
			if len(test.re) > 1 {
				exp = fmt.Sprintf("expected one of %d patterns:\n",
					len(test.re))
				for k, re := range test.re {
					exp += fmt.Sprintf("pattern %d:\n%v\n", k, re)
				}
			}
			t.Fatalf("failed test case %v, %sgot:\n%s",
				test.name, exp, got)
		}
	}
}

var tests = []struct {
	name   string
	run    string
	goos   string
	gorace string
	source string
	re     []string
}{
	{"simple", "run", "", "atexit_sleep_ms=0", `
package main
import "time"
var xptr *int
var donechan chan bool
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
	xptr = x
	donechan = done
	go racer()
}
func racer() {
	time.Sleep(10*time.Millisecond)
	store(xptr, 42)
	donechan <- true
}
`, []string{`==================
WARNING: DATA RACE
Write at 0x[0-9,a-f]+ by goroutine [0-9]:
  main\.store\(\)
      .+/main\.go:14 \+0x[0-9,a-f]+
  main\.racer\(\)
      .+/main\.go:23 \+0x[0-9,a-f]+

Previous write at 0x[0-9,a-f]+ by main goroutine:
  main\.store\(\)
      .+/main\.go:14 \+0x[0-9,a-f]+
  main\.main\(\)
      .+/main\.go:10 \+0x[0-9,a-f]+

Goroutine [0-9] \(running\) created at:
  main\.startRacer\(\)
      .+/main\.go:19 \+0x[0-9,a-f]+
  main\.main\(\)
      .+/main\.go:9 \+0x[0-9,a-f]+
==================
Found 1 data race\(s\)
exit status 66
`}},

	{"exitcode", "run", "", "atexit_sleep_ms=0 exitcode=13", `
package main
func main() {
	done := make(chan bool)
	x := 0; _ = x
	go func() {
		x = 42
		done <- true
	}()
	x = 43
	<-done
}
`, []string{`exit status 13`}},

	{"strip_path_prefix", "run", "", "atexit_sleep_ms=0 strip_path_prefix=/main.", `
package main
func main() {
	done := make(chan bool)
	x := 0; _ = x
	go func() {
		x = 42
		done <- true
	}()
	x = 43
	<-done
}
`, []string{`
      go:7 \+0x[0-9,a-f]+
`}},

	{"halt_on_error", "run", "", "atexit_sleep_ms=0 halt_on_error=1", `
package main
func main() {
	done := make(chan bool)
	x := 0; _ = x
	go func() {
		x = 42
		done <- true
	}()
	x = 43
	<-done
}
`, []string{`
==================
exit status 66
`}},

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
`, []string{`
==================
--- FAIL: TestFail \([0-9.]+s\)
.*testing.go:.*: race detected during execution of test
.*main_test.go:14: true
FAIL`}},

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
`, []string{`
  runtime\.slicebytetostring\(\)
      .*/runtime/string\.go:.*
  main\.main\.func1\(\)
      .*/main.go:7`}},

	// Test for https://golang.org/issue/33309
	{"midstack_inlining_traceback", "run", "linux", "atexit_sleep_ms=0", `
package main

var x int
var c chan int
func main() {
	c = make(chan int)
	go f()
	x = 1
	<-c
}

func f() {
	g(c)
}

func g(c chan int) {
	h(c)
}

func h(c chan int) {
	c <- x
}
`, []string{`==================
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
`}},

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
`, []string{`==================
WARNING: DATA RACE
Read at 0x[0-9,a-f]+ by main goroutine:
  main\.main\(\)
      .*/main\.go:34 \+0x[0-9,a-f]+

Previous write at 0x[0-9,a-f]+ by goroutine [0-9]:
  main\.goCallback\(\)
      .*/main\.go:27 \+0x[0-9,a-f]+
  _cgoexp_[0-9a-z]+_goCallback\(\)
      .*_cgo_gotypes\.go:[0-9]+ \+0x[0-9,a-f]+
  _cgoexp_[0-9a-z]+_goCallback\(\)
      <autogenerated>:1 \+0x[0-9,a-f]+

Goroutine [0-9] \(running\) created at:
  runtime\.newextram\(\)
      .*/runtime/proc.go:[0-9]+ \+0x[0-9,a-f]+
==================`,
		`==================
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
==================`}},
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
`, []string{`
==================
--- FAIL: TestFail \([0-9.]+s\)
.*testing.go:.*: race detected during execution of test
FAIL`}},
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
}`, []string{`pass`}},
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
`, []string{`==================
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
==================`}},
	// Test symbolizing wrappers. Both (*T).f and main.gowrap1 are wrappers.
	// go.dev/issue/60245
	{"wrappersym", "run", "", "atexit_sleep_ms=0", `
package main
import "sync"
var wg sync.WaitGroup
var x int
func main() {
	f := (*T).f
	wg.Add(2)
	go f(new(T))
	f(new(T))
	wg.Wait()
}
type T struct{}
func (t T) f() {
	x = 42
	wg.Done()
}
`, []string{`==================
WARNING: DATA RACE
Write at 0x[0-9,a-f]+ by goroutine [0-9]:
  main\.T\.f\(\)
      .*/main.go:15 \+0x[0-9,a-f]+
  main\.\(\*T\)\.f\(\)
      <autogenerated>:1 \+0x[0-9,a-f]+
  main\.main\.gowrap1\(\)
      .*/main.go:9 \+0x[0-9,a-f]+

Previous write at 0x[0-9,a-f]+ by main goroutine:
  main\.T\.f\(\)
      .*/main.go:15 \+0x[0-9,a-f]+
  main\.\(\*T\)\.f\(\)
      <autogenerated>:1 \+0x[0-9,a-f]+
  main\.main\(\)
      .*/main.go:10 \+0x[0-9,a-f]+

`}},
	{"non_inline_array_compare", "run", "", "atexit_sleep_ms=0", `
package main

import (
	"math/rand/v2"
)

var x = [1024]byte{}

var ch = make(chan bool)

func main() {
	started := make(chan struct{})
	go func() {
		close(started)
		var y = [len(x)]byte{}
		eq := x == y
		ch <- eq
	}()
	<-started
	x[rand.IntN(len(x))]++
	println(<-ch)
}
`, []string{`==================
WARNING: DATA RACE
`}},
	{"non_inline_struct_compare", "run", "", "atexit_sleep_ms=0", `
package main

import "math/rand/v2"

type S struct {
	a [1024]byte
}

var x = S{a: [1024]byte{}}

var ch = make(chan bool)

func main() {
	started := make(chan struct{})
	go func() {
		close(started)
		var y = S{a: [len(x.a)]byte{}}
		eq := x == y
		ch <- eq
	}()
	<-started
	x.a[rand.IntN(len(x.a))]++
	println(<-ch)
}
`, []string{`==================
WARNING: DATA RACE
`}},
}
