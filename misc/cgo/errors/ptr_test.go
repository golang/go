// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that cgo detects invalid pointer passing at runtime.

package errorstest

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
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
	{
		// Check a pointer to a union if the union has any
		// pointer fields.
		name:    "union1",
		c:       `typedef union { char **p; unsigned long i; } u; void f(u *pu) {}`,
		imports: []string{"unsafe"},
		body:    `var b C.char; p := &b; C.f((*C.u)(unsafe.Pointer(&p)))`,
		fail:    true,
	},
	{
		// Don't check a pointer to a union if the union does
		// not have any pointer fields.
		// Like ptrdata1 above, the uintptr represents an
		// integer that happens to have the same
		// representation as a pointer.
		name:    "union2",
		c:       `typedef union { unsigned long i; } u; void f(u *pu) {}`,
		imports: []string{"unsafe"},
		body:    `var b C.char; p := &b; C.f((*C.u)(unsafe.Pointer(&p)))`,
		fail:    false,
	},
	{
		// Test preemption while entering a cgo call. Issue #21306.
		name:    "preempt-during-call",
		c:       `void f() {}`,
		imports: []string{"runtime", "sync"},
		body:    `var wg sync.WaitGroup; wg.Add(100); for i := 0; i < 100; i++ { go func(i int) { for j := 0; j < 100; j++ { C.f(); runtime.GOMAXPROCS(i) }; wg.Done() }(i) }; wg.Wait()`,
		fail:    false,
	},
	{
		// Test poller deadline with cgocheck=2.  Issue #23435.
		name:    "deadline",
		c:       `#define US 10`,
		imports: []string{"os", "time"},
		body:    `r, _, _ := os.Pipe(); r.SetDeadline(time.Now().Add(C.US * time.Microsecond))`,
		fail:    false,
	},
	{
		// Test for double evaluation of channel receive.
		name:    "chan-recv",
		c:       `void f(char** p) {}`,
		imports: []string{"time"},
		body:    `c := make(chan []*C.char, 2); c <- make([]*C.char, 1); go func() { time.Sleep(10 * time.Second); panic("received twice from chan") }(); C.f(&(<-c)[0]);`,
		fail:    false,
	},
	{
		// Test that converting the address of a struct field
		// to unsafe.Pointer still just checks that field.
		// Issue #25941.
		name:    "struct-field",
		c:       `void f(void* p) {}`,
		imports: []string{"unsafe"},
		support: `type S struct { p *int; a [8]byte; u uintptr }`,
		body:    `s := &S{p: new(int)}; C.f(unsafe.Pointer(&s.a))`,
		fail:    false,
	},
	{
		// Test that converting multiple struct field
		// addresses to unsafe.Pointer still just checks those
		// fields. Issue #25941.
		name:    "struct-field-2",
		c:       `void f(void* p, int r, void* s) {}`,
		imports: []string{"unsafe"},
		support: `type S struct { a [8]byte; p *int; b int64; }`,
		body:    `s := &S{p: new(int)}; C.f(unsafe.Pointer(&s.a), 32, unsafe.Pointer(&s.b))`,
		fail:    false,
	},
	{
		// Test that second argument to cgoCheckPointer is
		// evaluated when a deferred function is deferred, not
		// when it is run.
		name:    "defer2",
		c:       `void f(char **pc) {}`,
		support: `type S1 struct { s []*C.char }; type S2 struct { ps *S1 }`,
		body:    `p := &S2{&S1{[]*C.char{nil}}}; defer C.f(&p.ps.s[0]); p.ps = nil`,
		fail:    false,
	},
	{
		// Test that indexing into a function call still
		// examines only the slice being indexed.
		name:    "buffer",
		c:       `void f(void *p) {}`,
		imports: []string{"bytes", "unsafe"},
		body:    `var b bytes.Buffer; b.WriteString("a"); C.f(unsafe.Pointer(&b.Bytes()[0]))`,
		fail:    false,
	},
	{
		// Test that bgsweep releasing a finalizer is OK.
		name:    "finalizer",
		c:       `// Nothing to declare.`,
		imports: []string{"os"},
		support: `func open() { os.Open(os.Args[0]) }; var G [][]byte`,
		body:    `for i := 0; i < 10000; i++ { G = append(G, make([]byte, 4096)); if i % 100 == 0 { G = nil; open() } }`,
		fail:    false,
	},
	{
		// Test that converting generated struct to interface is OK.
		name:    "structof",
		c:       `// Nothing to declare.`,
		imports: []string{"reflect"},
		support: `type MyInt int; func (i MyInt) Get() int { return int(i) }; type Getter interface { Get() int }`,
		body:    `t := reflect.StructOf([]reflect.StructField{{Name: "MyInt", Type: reflect.TypeOf(MyInt(0)), Anonymous: true}}); v := reflect.New(t).Elem(); v.Interface().(Getter).Get()`,
		fail:    false,
	},
}

func TestPointerChecks(t *testing.T) {
	for _, pt := range ptrTests {
		pt := pt
		t.Run(pt.name, func(t *testing.T) {
			testOne(t, pt)
		})
	}
}

func testOne(t *testing.T, pt ptrTest) {
	t.Parallel()

	gopath, err := ioutil.TempDir("", filepath.Base(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(gopath)

	src := filepath.Join(gopath, "src")
	if err := os.Mkdir(src, 0777); err != nil {
		t.Fatal(err)
	}

	name := filepath.Join(src, fmt.Sprintf("%s.go", filepath.Base(t.Name())))
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}

	b := bufio.NewWriter(f)
	fmt.Fprintln(b, `package main`)
	fmt.Fprintln(b)
	fmt.Fprintln(b, `/*`)
	fmt.Fprintln(b, pt.c)
	fmt.Fprintln(b, `*/`)
	fmt.Fprintln(b, `import "C"`)
	fmt.Fprintln(b)
	for _, imp := range pt.imports {
		fmt.Fprintln(b, `import "`+imp+`"`)
	}
	if len(pt.imports) > 0 {
		fmt.Fprintln(b)
	}
	if len(pt.support) > 0 {
		fmt.Fprintln(b, pt.support)
		fmt.Fprintln(b)
	}
	fmt.Fprintln(b, `func main() {`)
	fmt.Fprintln(b, pt.body)
	fmt.Fprintln(b, `}`)

	if err := b.Flush(); err != nil {
		t.Fatalf("flushing %s: %v", name, err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("closing %s: %v", name, err)
	}

	for _, e := range pt.extra {
		if err := ioutil.WriteFile(filepath.Join(src, e.name), []byte(e.contents), 0644); err != nil {
			t.Fatalf("writing %s: %v", e.name, err)
		}
	}

	args := func(cmd *exec.Cmd) string {
		return strings.Join(cmd.Args, " ")
	}

	cmd := exec.Command("go", "build")
	cmd.Dir = src
	cmd.Env = append(os.Environ(), "GOPATH="+gopath)
	buf, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("%#q:\n%s", args(cmd), buf)
		t.Fatalf("failed to build: %v", err)
	}

	exe := filepath.Join(src, filepath.Base(src))
	cmd = exec.Command(exe)
	cmd.Dir = src

	if pt.expensive {
		cmd.Env = cgocheckEnv("1")
		buf, err := cmd.CombinedOutput()
		if err != nil {
			t.Logf("%#q:\n%s", args(cmd), buf)
			if pt.fail {
				t.Fatalf("test marked expensive, but failed when not expensive: %v", err)
			} else {
				t.Errorf("failed unexpectedly with GODEBUG=cgocheck=1: %v", err)
			}
		}

		cmd = exec.Command(exe)
		cmd.Dir = src
	}

	if pt.expensive {
		cmd.Env = cgocheckEnv("2")
	}

	buf, err = cmd.CombinedOutput()
	if pt.fail {
		if err == nil {
			t.Logf("%#q:\n%s", args(cmd), buf)
			t.Fatalf("did not fail as expected")
		} else if !bytes.Contains(buf, []byte("Go pointer")) {
			t.Logf("%#q:\n%s", args(cmd), buf)
			t.Fatalf("did not print expected error (failed with %v)", err)
		}
	} else {
		if err != nil {
			t.Logf("%#q:\n%s", args(cmd), buf)
			t.Fatalf("failed unexpectedly: %v", err)
		}

		if !pt.expensive {
			// Make sure it passes with the expensive checks.
			cmd := exec.Command(exe)
			cmd.Dir = src
			cmd.Env = cgocheckEnv("2")
			buf, err := cmd.CombinedOutput()
			if err != nil {
				t.Logf("%#q:\n%s", args(cmd), buf)
				t.Fatalf("failed unexpectedly with expensive checks: %v", err)
			}
		}
	}

	if pt.fail {
		cmd = exec.Command(exe)
		cmd.Dir = src
		cmd.Env = cgocheckEnv("0")
		buf, err := cmd.CombinedOutput()
		if err != nil {
			t.Logf("%#q:\n%s", args(cmd), buf)
			t.Fatalf("failed unexpectedly with GODEBUG=cgocheck=0: %v", err)
		}
	}
}

func cgocheckEnv(val string) []string {
	return append(os.Environ(), "GODEBUG=cgocheck="+val)
}
