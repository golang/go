// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"text/template"
)

// testEnv excludes GODEBUG from the environment
// to prevent its output from breaking tests that
// are trying to parse other command output.
func testEnv(cmd *exec.Cmd) *exec.Cmd {
	if cmd.Env != nil {
		panic("environment already set")
	}
	for _, env := range os.Environ() {
		if strings.HasPrefix(env, "GODEBUG=") {
			continue
		}
		cmd.Env = append(cmd.Env, env)
	}
	return cmd
}

func executeTest(t *testing.T, templ string, data interface{}) string {
	checkStaleRuntime(t)

	st := template.Must(template.New("crashSource").Parse(templ))

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
	err = st.Execute(f, data)
	if err != nil {
		f.Close()
		t.Fatalf("failed to execute template: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("failed to close file: %v", err)
	}

	got, _ := testEnv(exec.Command("go", "run", src)).CombinedOutput()
	return string(got)
}

func checkStaleRuntime(t *testing.T) {
	// 'go run' uses the installed copy of runtime.a, which may be out of date.
	out, err := testEnv(exec.Command("go", "list", "-f", "{{.Stale}}", "runtime")).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to execute 'go list': %v\n%v", err, string(out))
	}
	if string(out) != "false\n" {
		t.Fatalf("Stale runtime.a. Run 'go install runtime'.")
	}
}

func testCrashHandler(t *testing.T, cgo bool) {
	type crashTest struct {
		Cgo bool
	}
	output := executeTest(t, crashSource, &crashTest{Cgo: cgo})
	want := "main: recovered done\nnew-thread: recovered done\nsecond-new-thread: recovered done\nmain-again: recovered done\n"
	if output != want {
		t.Fatalf("output:\n%s\n\nwanted:\n%s", output, want)
	}
}

func TestCrashHandler(t *testing.T) {
	testCrashHandler(t, false)
}

func testDeadlock(t *testing.T, source string) {
	output := executeTest(t, source, nil)
	want := "fatal error: all goroutines are asleep - deadlock!\n"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestSimpleDeadlock(t *testing.T) {
	testDeadlock(t, simpleDeadlockSource)
}

func TestInitDeadlock(t *testing.T) {
	testDeadlock(t, initDeadlockSource)
}

func TestLockedDeadlock(t *testing.T) {
	testDeadlock(t, lockedDeadlockSource)
}

func TestLockedDeadlock2(t *testing.T) {
	testDeadlock(t, lockedDeadlockSource2)
}

func TestGoexitDeadlock(t *testing.T) {
	output := executeTest(t, goexitDeadlockSource, nil)
	if output != "" {
		t.Fatalf("expected no output, got:\n%s", output)
	}
}

func TestStackOverflow(t *testing.T) {
	output := executeTest(t, stackOverflowSource, nil)
	want := "runtime: goroutine stack exceeds 4194304-byte limit\nfatal error: stack overflow"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestThreadExhaustion(t *testing.T) {
	output := executeTest(t, threadExhaustionSource, nil)
	want := "runtime: program exceeds 10-thread limit\nfatal error: thread exhaustion"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

const crashSource = `
package main

import (
	"fmt"
	"runtime"
)

{{if .Cgo}}
import "C"
{{end}}

func test(name string) {
	defer func() {
		if x := recover(); x != nil {
			fmt.Printf(" recovered")
		}
		fmt.Printf(" done\n")
	}()
	fmt.Printf("%s:", name)
	var s *string
	_ = *s
	fmt.Print("SHOULD NOT BE HERE")
}

func testInNewThread(name string) {
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		test(name)
		c <- true
	}()
	<-c
}

func main() {
	runtime.LockOSThread()
	test("main")
	testInNewThread("new-thread")
	testInNewThread("second-new-thread")
	test("main-again")
}
`

const simpleDeadlockSource = `
package main
func main() {
	select {}
}
`

const initDeadlockSource = `
package main
func init() {
	select {}
}
func main() {
}
`

const lockedDeadlockSource = `
package main
import "runtime"
func main() {
	runtime.LockOSThread()
	select {}
}
`

const lockedDeadlockSource2 = `
package main
import (
	"runtime"
	"time"
)
func main() {
	go func() {
		runtime.LockOSThread()
		select {}
	}()
	time.Sleep(time.Millisecond)
	select {}
}
`

const goexitDeadlockSource = `
package main
import (
      "runtime"
)

func F() {
      for i := 0; i < 10; i++ {
      }
}

func main() {
      go F()
      go F()
      runtime.Goexit()
}
`

const stackOverflowSource = `
package main

import "runtime/debug"

func main() {
	debug.SetMaxStack(4<<20)
	f(make([]byte, 10))
}

func f(x []byte) byte {
	var buf [64<<10]byte
	return x[0] + f(buf[:])
}
`

const threadExhaustionSource = `
package main

import (
	"runtime"
	"runtime/debug"
)

func main() {
	debug.SetMaxThreads(10)
	c := make(chan int)
	for i := 0; i < 100; i++ {
		go func() {
			runtime.LockOSThread()
			c <- 0
			select{}
		}()
		<-c
	}
}
`
