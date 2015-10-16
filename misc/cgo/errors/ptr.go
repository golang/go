// Copyright 2015 The Go Authors.  All rights reserved.
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
	c       string   // the cgo comment
	imports []string // a list of imports
	support string   // supporting functions
	body    string   // the body of the main function
	fail    bool     // whether the test should fail
}

var ptrTests = []ptrTest{
	{
		// Passing a pointer to a struct that contains a Go pointer.
		c:    `typedef struct s { int *p; } s; void f(s *ps) {}`,
		body: `C.f(&C.s{new(C.int)})`,
		fail: true,
	},
	{
		// Passing a pointer to a struct that contains a Go pointer.
		c:    `typedef struct s { int *p; } s; void f(s *ps) {}`,
		body: `p := &C.s{new(C.int)}; C.f(p)`,
		fail: true,
	},
	{
		// Passing a pointer to an int field of a Go struct
		// that (irrelevantly) contains a Go pointer.
		c:    `struct s { int i; int *p; }; void f(int *p) {}`,
		body: `p := &C.struct_s{i: 0, p: new(C.int)}; C.f(&p.i)`,
		fail: false,
	},
	{
		// Passing a pointer to a pointer field of a Go struct.
		c:    `struct s { int i; int *p; }; void f(int **p) {}`,
		body: `p := &C.struct_s{i: 0, p: new(C.int)}; C.f(&p.p)`,
		fail: true,
	},
	{
		// Passing a pointer to a pointer field of a Go
		// struct, where the field does not contain a Go
		// pointer, but another field (irrelevantly) does.
		c:    `struct s { int *p1; int *p2; }; void f(int **p) {}`,
		body: `p := &C.struct_s{p1: nil, p2: new(C.int)}; C.f(&p.p1)`,
		fail: false,
	},
	{
		// Passing the address of a slice with no Go pointers.
		c:       `void f(void **p) {}`,
		imports: []string{"unsafe"},
		body:    `s := []unsafe.Pointer{nil}; C.f(&s[0])`,
		fail:    false,
	},
	{
		// Passing the address of a slice with a Go pointer.
		c:       `void f(void **p) {}`,
		imports: []string{"unsafe"},
		body:    `i := 0; s := []unsafe.Pointer{unsafe.Pointer(&i)}; C.f(&s[0])`,
		fail:    true,
	},
	{
		// Passing the address of a slice with a Go pointer,
		// where we are passing the address of an element that
		// is not a Go pointer.
		c:       `void f(void **p) {}`,
		imports: []string{"unsafe"},
		body:    `i := 0; s := []unsafe.Pointer{nil, unsafe.Pointer(&i)}; C.f(&s[0])`,
		fail:    true,
	},
	{
		// Passing the address of a slice that is an element
		// in a struct only looks at the slice.
		c:       `void f(void **p) {}`,
		imports: []string{"unsafe"},
		support: `type S struct { p *int; s []unsafe.Pointer }`,
		body:    `i := 0; p := &S{p:&i, s:[]unsafe.Pointer{nil}}; C.f(&p.s[0])`,
		fail:    false,
	},
	{
		// Passing the address of a static variable with no
		// pointers doesn't matter.
		c:       `void f(char** parg) {}`,
		support: `var hello = [...]C.char{'h', 'e', 'l', 'l', 'o'}`,
		body:    `parg := [1]*C.char{&hello[0]}; C.f(&parg[0])`,
		fail:    false,
	},
	{
		// Passing the address of a static variable with
		// pointers does matter.
		c:       `void f(char*** parg) {}`,
		support: `var hello = [...]*C.char{new(C.char)}`,
		body:    `parg := [1]**C.char{&hello[0]}; C.f(&parg[0])`,
		fail:    true,
	},
}

func main() {
	os.Exit(doTests())
}

func doTests() int {
	dir, err := ioutil.TempDir("", "cgoerrors")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	defer os.RemoveAll(dir)

	workers := runtime.NumCPU() + 1

	var wg sync.WaitGroup
	c := make(chan int)
	errs := make(chan int)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			worker(dir, c, errs)
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

func worker(dir string, c, errs chan int) {
	e := 0
	for i := range c {
		if !doOne(dir, i) {
			e++
		}
	}
	if e > 0 {
		errs <- e
	}
}

func doOne(dir string, i int) bool {
	t := &ptrTests[i]

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
		fmt.Fprintln(os.Stderr, "closing %s: %v\n", name, err)
		return false
	}

	cmd := exec.Command("go", "run", name)
	cmd.Dir = dir
	buf, err := cmd.CombinedOutput()

	ok := true
	if t.fail {
		if err == nil {
			var errbuf bytes.Buffer
			fmt.Fprintf(&errbuf, "test %d did not fail as expected\n", i)
			reportTestOutput(&errbuf, i, buf)
			os.Stderr.Write(errbuf.Bytes())
			ok = false
		} else if !bytes.Contains(buf, []byte("Go pointer")) {
			var errbuf bytes.Buffer
			fmt.Fprintf(&errbuf, "test %d output does not contain expected error\n", i)
			reportTestOutput(&errbuf, i, buf)
			os.Stderr.Write(errbuf.Bytes())
			ok = false
		}
	} else {
		if err != nil {
			var errbuf bytes.Buffer
			fmt.Fprintf(&errbuf, "test %d failed unexpectedly: %v\n", i, err)
			reportTestOutput(&errbuf, i, buf)
			os.Stderr.Write(errbuf.Bytes())
			ok = false
		}
	}

	if t.fail && ok {
		cmd = exec.Command("go", "run", name)
		cmd.Dir = dir
		env := []string{"GODEBUG=cgocheck=0"}
		for _, e := range os.Environ() {
			if !strings.HasPrefix(e, "GODEBUG=") {
				env = append(env, e)
			}
		}
		cmd.Env = env
		buf, err := cmd.CombinedOutput()
		if err != nil {
			var errbuf bytes.Buffer
			fmt.Fprintf(&errbuf, "test %d failed unexpectedly with GODEBUG=cgocheck=0: %v\n", i, err)
			reportTestOutput(&errbuf, i, buf)
			os.Stderr.Write(errbuf.Bytes())
			ok = false
		}
	}

	return ok
}

func reportTestOutput(w io.Writer, i int, buf []byte) {
	fmt.Fprintf(w, "=== test %d output ===\n", i)
	fmt.Fprintf(w, "%s", buf)
	fmt.Fprintf(w, "=== end of test %d output ===\n", i)
}
