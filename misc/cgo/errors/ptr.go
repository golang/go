// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that cgo detects invalid pointer passing at runtime.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// ptrTest is the tests without the boilerplate.
type ptrTest struct {
	name      string   // for reporting
	c         string   // the cgo comment
	imports   []string // a list of imports
	support   string   // supporting functions
	body      string   // the body of the main function
	extra     []extra  // extra files
	fail      bool     // whether the test should fail
	expensive bool     // whether the test requires the expensive check
}

type extra struct {
	name     string
	contents string
}

var ptrTests = []ptrTest{
	{
		// Passing a pointer to a struct that contains a Go pointer.
		name: "ptr1",
		c:    `typedef struct s { int *p; } s; void f(s *ps) {}`,
		body: `C.f(&C.s{new(C.int)})`,
		fail: true,
	},
	{
		// Passing a pointer to a struct that contains a Go pointer.
		name: "ptr2",
		c:    `typedef struct s { int *p; } s; void f(s *ps) {}`,
		body: `p := &C.s{new(C.int)}; C.f(p)`,
		fail: true,
	},
	{
		// Passing a pointer to an int field of a Go struct
		// that (irrelevantly) contains a Go pointer.
		name: "ok1",
		c:    `struct s { int i; int *p; }; void f(int *p) {}`,
		body: `p := &C.struct_s{i: 0, p: new(C.int)}; C.f(&p.i)`,
		fail: false,
	},
	{
		// Passing a pointer to a pointer field of a Go struct.
		name: "ptr-field",
		c:    `struct s { int i; int *p; }; void f(int **p) {}`,
		body: `p := &C.struct_s{i: 0, p: new(C.int)}; C.f(&p.p)`,
		fail: true,
	},
	{
		// Passing a pointer to a pointer field of a Go
		// struct, where the field does not contain a Go
		// pointer, but another field (irrelevantly) does.
		name: "ptr-field-ok",
		c:    `struct s { int *p1; int *p2; }; void f(int **p) {}`,
		body: `p := &C.struct_s{p1: nil, p2: new(C.int)}; C.f(&p.p1)`,
		fail: false,
	},
	{
		// Passing the address of a slice with no Go pointers.
		name:    "slice-ok-1",
		c:       `void f(void **p) {}`,
		imports: []string{"unsafe"},
		body:    `s := []unsafe.Pointer{nil}; C.f(&s[0])`,
		fail:    false,
	},
	{
		// Passing the address of a slice with a Go pointer.
		name:    "slice-ptr-1",
		c:       `void f(void **p) {}`,
		imports: []string{"unsafe"},
		body:    `i := 0; s := []unsafe.Pointer{unsafe.Pointer(&i)}; C.f(&s[0])`,
		fail:    true,
	},
	{
		// Passing the address of a slice with a Go pointer,
		// where we are passing the address of an element that
		// is not a Go pointer.
		name:    "slice-ptr-2",
		c:       `void f(void **p) {}`,
		imports: []string{"unsafe"},
		body:    `i := 0; s := []unsafe.Pointer{nil, unsafe.Pointer(&i)}; C.f(&s[0])`,
		fail:    true,
	},
	{
		// Passing the address of a slice that is an element
		// in a struct only looks at the slice.
		name:    "slice-ok-2",
		c:       `void f(void **p) {}`,
		imports: []string{"unsafe"},
		support: `type S struct { p *int; s []unsafe.Pointer }`,
		body:    `i := 0; p := &S{p:&i, s:[]unsafe.Pointer{nil}}; C.f(&p.s[0])`,
		fail:    false,
	},
	{
		// Passing the address of a slice of an array that is
		// an element in a struct, with a type conversion.
		name:    "slice-ok-3",
		c:       `void f(void* p) {}`,
		imports: []string{"unsafe"},
		support: `type S struct { p *int; a [4]byte }`,
		body:    `i := 0; p := &S{p:&i}; s := p.a[:]; C.f(unsafe.Pointer(&s[0]))`,
		fail:    false,
	},
	{
		// Passing the address of a slice of an array that is
		// an element in a struct, with a type conversion.
		name:    "slice-ok-4",
		c:       `typedef void* PV; void f(PV p) {}`,
		imports: []string{"unsafe"},
		support: `type S struct { p *int; a [4]byte }`,
		body:    `i := 0; p := &S{p:&i}; C.f(C.PV(unsafe.Pointer(&p.a[0])))`,
		fail:    false,
	},
	{
		// Passing the address of a static variable with no
		// pointers doesn't matter.
		name:    "varok",
		c:       `void f(char** parg) {}`,
		support: `var hello = [...]C.char{'h', 'e', 'l', 'l', 'o'}`,
		body:    `parg := [1]*C.char{&hello[0]}; C.f(&parg[0])`,
		fail:    false,
	},
	{
		// Passing the address of a static variable with
		// pointers does matter.
		name:    "var",
		c:       `void f(char*** parg) {}`,
		support: `var hello = [...]*C.char{new(C.char)}`,
		body:    `parg := [1]**C.char{&hello[0]}; C.f(&parg[0])`,
		fail:    true,
	},
	{
		// Storing a Go pointer into C memory should fail.
		name: "barrier",
		c: `#include <stdlib.h>
                    char **f1() { return malloc(sizeof(char*)); }
                    void f2(char **p) {}`,
		body:      `p := C.f1(); *p = new(C.char); C.f2(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// Storing a Go pointer into C memory by assigning a
		// large value should fail.
		name: "barrier-struct",
		c: `#include <stdlib.h>
                    struct s { char *a[10]; };
                    struct s *f1() { return malloc(sizeof(struct s)); }
                    void f2(struct s *p) {}`,
		body:      `p := C.f1(); p.a = [10]*C.char{new(C.char)}; C.f2(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// Storing a Go pointer into C memory using a slice
		// copy should fail.
		name: "barrier-slice",
		c: `#include <stdlib.h>
                    struct s { char *a[10]; };
                    struct s *f1() { return malloc(sizeof(struct s)); }
                    void f2(struct s *p) {}`,
		body:      `p := C.f1(); copy(p.a[:], []*C.char{new(C.char)}); C.f2(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// A very large value uses a GC program, which is a
		// different code path.
		name: "barrier-gcprog-array",
		c: `#include <stdlib.h>
                    struct s { char *a[32769]; };
                    struct s *f1() { return malloc(sizeof(struct s)); }
                    void f2(struct s *p) {}`,
		body:      `p := C.f1(); p.a = [32769]*C.char{new(C.char)}; C.f2(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// Similar case, with a source on the heap.
		name: "barrier-gcprog-array-heap",
		c: `#include <stdlib.h>
                    struct s { char *a[32769]; };
                    struct s *f1() { return malloc(sizeof(struct s)); }
                    void f2(struct s *p) {}
                    void f3(void *p) {}`,
		imports:   []string{"unsafe"},
		body:      `p := C.f1(); n := &[32769]*C.char{new(C.char)}; p.a = *n; C.f2(p); n[0] = nil; C.f3(unsafe.Pointer(n))`,
		fail:      true,
		expensive: true,
	},
	{
		// A GC program with a struct.
		name: "barrier-gcprog-struct",
		c: `#include <stdlib.h>
                    struct s { char *a[32769]; };
                    struct s2 { struct s f; };
                    struct s2 *f1() { return malloc(sizeof(struct s2)); }
                    void f2(struct s2 *p) {}`,
		body:      `p := C.f1(); p.f = C.struct_s{[32769]*C.char{new(C.char)}}; C.f2(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// Similar case, with a source on the heap.
		name: "barrier-gcprog-struct-heap",
		c: `#include <stdlib.h>
                    struct s { char *a[32769]; };
                    struct s2 { struct s f; };
                    struct s2 *f1() { return malloc(sizeof(struct s2)); }
                    void f2(struct s2 *p) {}
                    void f3(void *p) {}`,
		imports:   []string{"unsafe"},
		body:      `p := C.f1(); n := &C.struct_s{[32769]*C.char{new(C.char)}}; p.f = *n; C.f2(p); n.a[0] = nil; C.f3(unsafe.Pointer(n))`,
		fail:      true,
		expensive: true,
	},
	{
		// Exported functions may not return Go pointers.
		name: "export1",
		c:    `extern unsigned char *GoFn();`,
		support: `//export GoFn
                          func GoFn() *byte { return new(byte) }`,
		body: `C.GoFn()`,
		fail: true,
	},
	{
		// Returning a C pointer is fine.
		name: "exportok",
		c: `#include <stdlib.h>
                    extern unsigned char *GoFn();`,
		support: `//export GoFn
                          func GoFn() *byte { return (*byte)(C.malloc(1)) }`,
		body: `C.GoFn()`,
	},
	{
		// Passing a Go string is fine.
		name: "pass-string",
		c: `#include <stddef.h>
                    typedef struct { const char *p; ptrdiff_t n; } gostring;
                    gostring f(gostring s) { return s; }`,
		imports: []string{"unsafe"},
		body:    `s := "a"; r := C.f(*(*C.gostring)(unsafe.Pointer(&s))); if *(*string)(unsafe.Pointer(&r)) != s { panic(r) }`,
	},
	{
		// Passing a slice of Go strings fails.
		name:    "pass-string-slice",
		c:       `void f(void *p) {}`,
		imports: []string{"strings", "unsafe"},
		support: `type S struct { a [1]string }`,
		body:    `s := S{a:[1]string{strings.Repeat("a", 2)}}; C.f(unsafe.Pointer(&s.a[0]))`,
		fail:    true,
	},
	{
		// Exported functions may not return strings.
		name:    "ret-string",
		c:       `extern void f();`,
		imports: []string{"strings"},
		support: `//export GoStr
                          func GoStr() string { return strings.Repeat("a", 2) }`,
		body: `C.f()`,
		extra: []extra{
			{
				"call.c",
				`#include <stddef.h>
                                 typedef struct { const char *p; ptrdiff_t n; } gostring;
                                 extern gostring GoStr();
                                 void f() { GoStr(); }`,
			},
		},
		fail: true,
	},
	{
		// Don't check non-pointer data.
		// Uses unsafe code to get a pointer we shouldn't check.
		// Although we use unsafe, the uintptr represents an integer
		// that happens to have the same representation as a pointer;
		// that is, we are testing something that is not unsafe.
		name: "ptrdata1",
		c: `#include <stdlib.h>
                    void f(void* p) {}`,
		imports: []string{"unsafe"},
		support: `type S struct { p *int; a [8*8]byte; u uintptr }`,
		body:    `i := 0; p := &S{u:uintptr(unsafe.Pointer(&i))}; q := (*S)(C.malloc(C.size_t(unsafe.Sizeof(*p)))); *q = *p; C.f(unsafe.Pointer(q))`,
		fail:    false,
	},
	{
		// Like ptrdata1, but with a type that uses a GC program.
		name: "ptrdata2",
		c: `#include <stdlib.h>
                    void f(void* p) {}`,
		imports: []string{"unsafe"},
		support: `type S struct { p *int; a [32769*8]byte; q *int; u uintptr }`,
		body:    `i := 0; p := S{u:uintptr(unsafe.Pointer(&i))}; q := (*S)(C.malloc(C.size_t(unsafe.Sizeof(p)))); *q = p; C.f(unsafe.Pointer(q))`,
		fail:    false,
	},
	{
		// Check deferred pointers when they are used, not
		// when the defer statement is run.
		name: "defer",
		c:    `typedef struct s { int *p; } s; void f(s *ps) {}`,
		body: `p := &C.s{}; defer C.f(p); p.p = new(C.int)`,
		fail: true,
	},
}

func main() {
	os.Exit(doTests())
}

func doTests() int {
	gopath, err := ioutil.TempDir("", "cgoerrors")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	defer os.RemoveAll(gopath)

	if err := os.MkdirAll(filepath.Join(gopath, "src"), 0777); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}

	workers := runtime.NumCPU() + 1

	var wg sync.WaitGroup
	c := make(chan int)
	errs := make(chan int)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			worker(gopath, c, errs)
			wg.Done()
		}()
	}

	for i := range ptrTests {
		c <- i
	}
	close(c)

	go func() {
		wg.Wait()
		close(errs)
	}()

	tot := 0
	for e := range errs {
		tot += e
	}
	return tot
}

func worker(gopath string, c, errs chan int) {
	e := 0
	for i := range c {
		if !doOne(gopath, i) {
			e++
		}
	}
	if e > 0 {
		errs <- e
	}
}

func doOne(gopath string, i int) bool {
	t := &ptrTests[i]

	dir := filepath.Join(gopath, "src", fmt.Sprintf("dir%d", i))
	if err := os.Mkdir(dir, 0777); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return false
	}

	name := filepath.Join(dir, fmt.Sprintf("t%d.go", i))
	f, err := os.Create(name)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return false
	}

	b := bufio.NewWriter(f)
	fmt.Fprintln(b, `package main`)
	fmt.Fprintln(b)
	fmt.Fprintln(b, `/*`)
	fmt.Fprintln(b, t.c)
	fmt.Fprintln(b, `*/`)
	fmt.Fprintln(b, `import "C"`)
	fmt.Fprintln(b)
	for _, imp := range t.imports {
		fmt.Fprintln(b, `import "`+imp+`"`)
	}
	if len(t.imports) > 0 {
		fmt.Fprintln(b)
	}
	if len(t.support) > 0 {
		fmt.Fprintln(b, t.support)
		fmt.Fprintln(b)
	}
	fmt.Fprintln(b, `func main() {`)
	fmt.Fprintln(b, t.body)
	fmt.Fprintln(b, `}`)

	if err := b.Flush(); err != nil {
		fmt.Fprintf(os.Stderr, "flushing %s: %v\n", name, err)
		return false
	}
	if err := f.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "closing %s: %v\n", name, err)
		return false
	}

	for _, e := range t.extra {
		if err := ioutil.WriteFile(filepath.Join(dir, e.name), []byte(e.contents), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "writing %s: %v\n", e.name, err)
			return false
		}
	}

	ok := true

	cmd := exec.Command("go", "build")
	cmd.Dir = dir
	cmd.Env = addEnv("GOPATH", gopath)
	buf, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "test %s failed to build: %v\n%s", t.name, err, buf)
		return false
	}

	exe := filepath.Join(dir, filepath.Base(dir))
	cmd = exec.Command(exe)
	cmd.Dir = dir

	if t.expensive {
		cmd.Env = cgocheckEnv("1")
		buf, err := cmd.CombinedOutput()
		if err != nil {
			var errbuf bytes.Buffer
			if t.fail {
				fmt.Fprintf(&errbuf, "test %s marked expensive but failed when not expensive: %v\n", t.name, err)
			} else {
				fmt.Fprintf(&errbuf, "test %s failed unexpectedly with GODEBUG=cgocheck=1: %v\n", t.name, err)
			}
			reportTestOutput(&errbuf, t.name, buf)
			os.Stderr.Write(errbuf.Bytes())
			ok = false
		}

		cmd = exec.Command(exe)
		cmd.Dir = dir
	}

	if t.expensive {
		cmd.Env = cgocheckEnv("2")
	}

	buf, err = cmd.CombinedOutput()

	if t.fail {
		if err == nil {
			var errbuf bytes.Buffer
			fmt.Fprintf(&errbuf, "test %s did not fail as expected\n", t.name)
			reportTestOutput(&errbuf, t.name, buf)
			os.Stderr.Write(errbuf.Bytes())
			ok = false
		} else if !bytes.Contains(buf, []byte("Go pointer")) {
			var errbuf bytes.Buffer
			fmt.Fprintf(&errbuf, "test %s output does not contain expected error (failed with %v)\n", t.name, err)
			reportTestOutput(&errbuf, t.name, buf)
			os.Stderr.Write(errbuf.Bytes())
			ok = false
		}
	} else {
		if err != nil {
			var errbuf bytes.Buffer
			fmt.Fprintf(&errbuf, "test %s failed unexpectedly: %v\n", t.name, err)
			reportTestOutput(&errbuf, t.name, buf)
			os.Stderr.Write(errbuf.Bytes())
			ok = false
		}

		if !t.expensive && ok {
			// Make sure it passes with the expensive checks.
			cmd := exec.Command(exe)
			cmd.Dir = dir
			cmd.Env = cgocheckEnv("2")
			buf, err := cmd.CombinedOutput()
			if err != nil {
				var errbuf bytes.Buffer
				fmt.Fprintf(&errbuf, "test %s failed unexpectedly with expensive checks: %v\n", t.name, err)
				reportTestOutput(&errbuf, t.name, buf)
				os.Stderr.Write(errbuf.Bytes())
				ok = false
			}
		}
	}

	if t.fail && ok {
		cmd = exec.Command(exe)
		cmd.Dir = dir
		cmd.Env = cgocheckEnv("0")
		buf, err := cmd.CombinedOutput()
		if err != nil {
			var errbuf bytes.Buffer
			fmt.Fprintf(&errbuf, "test %s failed unexpectedly with GODEBUG=cgocheck=0: %v\n", t.name, err)
			reportTestOutput(&errbuf, t.name, buf)
			os.Stderr.Write(errbuf.Bytes())
			ok = false
		}
	}

	return ok
}

func reportTestOutput(w io.Writer, name string, buf []byte) {
	fmt.Fprintf(w, "=== test %s output ===\n", name)
	fmt.Fprintf(w, "%s", buf)
	fmt.Fprintf(w, "=== end of test %s output ===\n", name)
}

func cgocheckEnv(val string) []string {
	return addEnv("GODEBUG", "cgocheck="+val)
}

func addEnv(key, val string) []string {
	env := []string{key + "=" + val}
	look := key + "="
	for _, e := range os.Environ() {
		if !strings.HasPrefix(e, look) {
			env = append(env, e)
		}
	}
	return env
}
