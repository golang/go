// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
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

func executeTest(t *testing.T, templ string, data interface{}, extra ...string) string {
	switch runtime.GOOS {
	case "android", "nacl":
		t.Skipf("skipping on %s", runtime.GOOS)
	}

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

	for i := 0; i < len(extra); i += 2 {
		if err := ioutil.WriteFile(filepath.Join(dir, extra[i]), []byte(extra[i+1]), 0666); err != nil {
			t.Fatal(err)
		}
	}

	cmd := exec.Command("go", "build", "-o", "a.exe")
	cmd.Dir = dir
	out, err := testEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source: %v\n%s", err, out)
	}

	got, _ := testEnv(exec.Command(filepath.Join(dir, "a.exe"))).CombinedOutput()
	return string(got)
}

var (
	staleRuntimeOnce sync.Once // guards init of staleRuntimeErr
	staleRuntimeErr  error
)

func checkStaleRuntime(t *testing.T) {
	staleRuntimeOnce.Do(func() {
		// 'go run' uses the installed copy of runtime.a, which may be out of date.
		out, err := testEnv(exec.Command("go", "list", "-f", "{{.Stale}}", "runtime")).CombinedOutput()
		if err != nil {
			staleRuntimeErr = fmt.Errorf("failed to execute 'go list': %v\n%v", err, string(out))
			return
		}
		if string(out) != "false\n" {
			staleRuntimeErr = fmt.Errorf("Stale runtime.a. Run 'go install runtime'.")
		}
	})
	if staleRuntimeErr != nil {
		t.Fatal(staleRuntimeErr)
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
	want := "no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.Contains(output, want) {
		t.Fatalf("output:\n%s\n\nwant output containing: %s", output, want)
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

func TestRecursivePanic(t *testing.T) {
	output := executeTest(t, recursivePanicSource, nil)
	want := `wrap: bad
panic: again

`
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}

}

func TestGoexitCrash(t *testing.T) {
	output := executeTest(t, goexitExitSource, nil)
	want := "no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.Contains(output, want) {
		t.Fatalf("output:\n%s\n\nwant output containing: %s", output, want)
	}
}

func TestGoexitDefer(t *testing.T) {
	c := make(chan struct{})
	go func() {
		defer func() {
			r := recover()
			if r != nil {
				t.Errorf("non-nil recover during Goexit")
			}
			c <- struct{}{}
		}()
		runtime.Goexit()
	}()
	// Note: if the defer fails to run, we will get a deadlock here
	<-c
}

func TestGoNil(t *testing.T) {
	output := executeTest(t, goNilSource, nil)
	want := "go of nil func value"
	if !strings.Contains(output, want) {
		t.Fatalf("output:\n%s\n\nwant output containing: %s", output, want)
	}
}

func TestMainGoroutineId(t *testing.T) {
	output := executeTest(t, mainGoroutineIdSource, nil)
	want := "panic: test\n\ngoroutine 1 [running]:\n"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestBreakpoint(t *testing.T) {
	output := executeTest(t, breakpointSource, nil)
	want := "runtime.Breakpoint()"
	if !strings.Contains(output, want) {
		t.Fatalf("output:\n%s\n\nwant output containing: %s", output, want)
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

const recursivePanicSource = `
package main

import (
	"fmt"
)

func main() {
	func() {
		defer func() {
			fmt.Println(recover())
		}()
		var x [8192]byte
		func(x [8192]byte) {
			defer func() {
				if err := recover(); err != nil {
					panic("wrap: " + err.(string))
				}
			}()
			panic("bad")
		}(x)
	}()
	panic("again")
}
`

const goexitExitSource = `
package main

import (
	"runtime"
	"time"
)

func main() {
	go func() {
		time.Sleep(time.Millisecond)
	}()
	i := 0
	runtime.SetFinalizer(&i, func(p *int) {})
	runtime.GC()
	runtime.Goexit()
}
`

const goNilSource = `
package main

func main() {
	defer func() {
		recover()
	}()
	var f func()
	go f()
	select{}
}
`

const mainGoroutineIdSource = `
package main
func main() {
	panic("test")
}
`

const breakpointSource = `
package main
import "runtime"
func main() {
	runtime.Breakpoint()
}
`

func TestGoexitInPanic(t *testing.T) {
	// see issue 8774: this code used to trigger an infinite recursion
	output := executeTest(t, goexitInPanicSource, nil)
	want := "fatal error: no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

const goexitInPanicSource = `
package main
import "runtime"
func main() {
	go func() {
		defer func() {
			runtime.Goexit()
		}()
		panic("hello")
	}()
	runtime.Goexit()
}
`

func TestPanicAfterGoexit(t *testing.T) {
	// an uncaught panic should still work after goexit
	output := executeTest(t, panicAfterGoexitSource, nil)
	want := "panic: hello"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

const panicAfterGoexitSource = `
package main
import "runtime"
func main() {
	defer func() {
		panic("hello")
	}()
	runtime.Goexit()
}
`

func TestRecoveredPanicAfterGoexit(t *testing.T) {
	output := executeTest(t, recoveredPanicAfterGoexitSource, nil)
	want := "fatal error: no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

const recoveredPanicAfterGoexitSource = `
package main
import "runtime"
func main() {
	defer func() {
		defer func() {
			r := recover()
			if r == nil {
				panic("bad recover")
			}
		}()
		panic("hello")
	}()
	runtime.Goexit()
}
`

func TestRecoverBeforePanicAfterGoexit(t *testing.T) {
	// 1. defer a function that recovers
	// 2. defer a function that panics
	// 3. call goexit
	// Goexit should run the #2 defer.  Its panic
	// should be caught by the #1 defer, and execution
	// should resume in the caller.  Like the Goexit
	// never happened!
	defer func() {
		r := recover()
		if r == nil {
			panic("bad recover")
		}
	}()
	defer func() {
		panic("hello")
	}()
	runtime.Goexit()
}

func TestNetpollDeadlock(t *testing.T) {
	output := executeTest(t, netpollDeadlockSource, nil)
	want := "done\n"
	if !strings.HasSuffix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

const netpollDeadlockSource = `
package main
import (
	"fmt"
	"net"
)
func init() {
	fmt.Println("dialing")
	c, err := net.Dial("tcp", "localhost:14356")
	if err == nil {
		c.Close()
	} else {
		fmt.Println("error: ", err)
	}
}
func main() {
	fmt.Println("done")
}
`
