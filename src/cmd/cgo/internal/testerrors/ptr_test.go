// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that cgo detects invalid pointer passing at runtime.

package errorstest

import (
	"bytes"
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
)

var tmp = flag.String("tmp", "", "use `dir` for temporary files and do not clean up")

// ptrTest is the tests without the boilerplate.
type ptrTest struct {
	name          string   // for reporting
	c             string   // the cgo comment
	c1            string   // cgo comment forced into non-export cgo file
	imports       []string // a list of imports
	support       string   // supporting functions
	body          string   // the body of the main function
	extra         []extra  // extra files
	fail          bool     // whether the test should fail
	expensive     bool     // whether the test requires the expensive check
	errTextRegexp string   // error text regexp; if empty, use the pattern `.*unpinned Go.*`
}

type extra struct {
	name     string
	contents string
}

var ptrTests = []ptrTest{
	{
		// Passing a pointer to a struct that contains a Go pointer.
		name: "ptr1",
		c:    `typedef struct s1 { int *p; } s1; void f1(s1 *ps) {}`,
		body: `C.f1(&C.s1{new(C.int)})`,
		fail: true,
	},
	{
		// Passing a pointer to a struct that contains a Go pointer.
		name: "ptr2",
		c:    `typedef struct s2 { int *p; } s2; void f2(s2 *ps) {}`,
		body: `p := &C.s2{new(C.int)}; C.f2(p)`,
		fail: true,
	},
	{
		// Passing a pointer to an int field of a Go struct
		// that (irrelevantly) contains a Go pointer.
		name: "ok1",
		c:    `struct s3 { int i; int *p; }; void f3(int *p) {}`,
		body: `p := &C.struct_s3{i: 0, p: new(C.int)}; C.f3(&p.i)`,
		fail: false,
	},
	{
		// Passing a pointer to a pointer field of a Go struct.
		name: "ptrfield",
		c:    `struct s4 { int i; int *p; }; void f4(int **p) {}`,
		body: `p := &C.struct_s4{i: 0, p: new(C.int)}; C.f4(&p.p)`,
		fail: true,
	},
	{
		// Passing a pointer to a pointer field of a Go
		// struct, where the field does not contain a Go
		// pointer, but another field (irrelevantly) does.
		name: "ptrfieldok",
		c:    `struct s5 { int *p1; int *p2; }; void f5(int **p) {}`,
		body: `p := &C.struct_s5{p1: nil, p2: new(C.int)}; C.f5(&p.p1)`,
		fail: false,
	},
	{
		// Passing the address of a slice with no Go pointers.
		name:    "sliceok1",
		c:       `void f6(void **p) {}`,
		imports: []string{"unsafe"},
		body:    `s := []unsafe.Pointer{nil}; C.f6(&s[0])`,
		fail:    false,
	},
	{
		// Passing the address of a slice with a Go pointer.
		name:    "sliceptr1",
		c:       `void f7(void **p) {}`,
		imports: []string{"unsafe"},
		body:    `i := 0; s := []unsafe.Pointer{unsafe.Pointer(&i)}; C.f7(&s[0])`,
		fail:    true,
	},
	{
		// Passing the address of a slice with a Go pointer,
		// where we are passing the address of an element that
		// is not a Go pointer.
		name:    "sliceptr2",
		c:       `void f8(void **p) {}`,
		imports: []string{"unsafe"},
		body:    `i := 0; s := []unsafe.Pointer{nil, unsafe.Pointer(&i)}; C.f8(&s[0])`,
		fail:    true,
	},
	{
		// Passing the address of a slice that is an element
		// in a struct only looks at the slice.
		name:    "sliceok2",
		c:       `void f9(void **p) {}`,
		imports: []string{"unsafe"},
		support: `type S9 struct { p *int; s []unsafe.Pointer }`,
		body:    `i := 0; p := &S9{p:&i, s:[]unsafe.Pointer{nil}}; C.f9(&p.s[0])`,
		fail:    false,
	},
	{
		// Passing the address of a slice of an array that is
		// an element in a struct, with a type conversion.
		name:    "sliceok3",
		c:       `void f10(void* p) {}`,
		imports: []string{"unsafe"},
		support: `type S10 struct { p *int; a [4]byte }`,
		body:    `i := 0; p := &S10{p:&i}; s := p.a[:]; C.f10(unsafe.Pointer(&s[0]))`,
		fail:    false,
	},
	{
		// Passing the address of a slice of an array that is
		// an element in a struct, with a type conversion.
		name:    "sliceok4",
		c:       `typedef void* PV11; void f11(PV11 p) {}`,
		imports: []string{"unsafe"},
		support: `type S11 struct { p *int; a [4]byte }`,
		body:    `i := 0; p := &S11{p:&i}; C.f11(C.PV11(unsafe.Pointer(&p.a[0])))`,
		fail:    false,
	},
	{
		// Passing the address of a static variable with no
		// pointers doesn't matter.
		name:    "varok",
		c:       `void f12(char** parg) {}`,
		support: `var hello12 = [...]C.char{'h', 'e', 'l', 'l', 'o'}`,
		body:    `parg := [1]*C.char{&hello12[0]}; C.f12(&parg[0])`,
		fail:    false,
	},
	{
		// Passing the address of a static variable with
		// pointers does matter.
		name:    "var1",
		c:       `void f13(char*** parg) {}`,
		support: `var hello13 = [...]*C.char{new(C.char)}`,
		body:    `parg := [1]**C.char{&hello13[0]}; C.f13(&parg[0])`,
		fail:    true,
		errTextRegexp: `.*argument of cgo function has Go pointer to unpinned Go pointer.*`,
	},
	{
		// Storing a Go pointer into C memory should fail.
		name: "barrier",
		c: `#include <stdlib.h>
		    char **f14a() { return malloc(sizeof(char*)); }
		    void f14b(char **p) {}`,
		body:      `p := C.f14a(); *p = new(C.char); C.f14b(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// Storing a pinned Go pointer into C memory should succeed.
		name: "barrierpinnedok",
		c: `#include <stdlib.h>
		    char **f14a2() { return malloc(sizeof(char*)); }
		    void f14b2(char **p) {}`,
		imports:   []string{"runtime"},
		body:      `var pinr runtime.Pinner; p := C.f14a2(); x := new(C.char); pinr.Pin(x); *p = x; C.f14b2(p); pinr.Unpin()`,
		fail:      false,
		expensive: true,
	},
	{
		// Storing a Go pointer into C memory by assigning a
		// large value should fail.
		name: "barrierstruct",
		c: `#include <stdlib.h>
		    struct s15 { char *a[10]; };
		    struct s15 *f15() { return malloc(sizeof(struct s15)); }
		    void f15b(struct s15 *p) {}`,
		body:      `p := C.f15(); p.a = [10]*C.char{new(C.char)}; C.f15b(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// Storing a Go pointer into C memory using a slice
		// copy should fail.
		name: "barrierslice",
		c: `#include <stdlib.h>
		    struct s16 { char *a[10]; };
		    struct s16 *f16() { return malloc(sizeof(struct s16)); }
		    void f16b(struct s16 *p) {}`,
		body:      `p := C.f16(); copy(p.a[:], []*C.char{new(C.char)}); C.f16b(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// A very large value uses a GC program, which is a
		// different code path.
		name: "barriergcprogarray",
		c: `#include <stdlib.h>
		    struct s17 { char *a[32769]; };
		    struct s17 *f17() { return malloc(sizeof(struct s17)); }
		    void f17b(struct s17 *p) {}`,
		body:      `p := C.f17(); p.a = [32769]*C.char{new(C.char)}; C.f17b(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// Similar case, with a source on the heap.
		name: "barriergcprogarrayheap",
		c: `#include <stdlib.h>
		    struct s18 { char *a[32769]; };
		    struct s18 *f18() { return malloc(sizeof(struct s18)); }
		    void f18b(struct s18 *p) {}
		    void f18c(void *p) {}`,
		imports:   []string{"unsafe"},
		body:      `p := C.f18(); n := &[32769]*C.char{new(C.char)}; p.a = *n; C.f18b(p); n[0] = nil; C.f18c(unsafe.Pointer(n))`,
		fail:      true,
		expensive: true,
	},
	{
		// A GC program with a struct.
		name: "barriergcprogstruct",
		c: `#include <stdlib.h>
		    struct s19a { char *a[32769]; };
		    struct s19b { struct s19a f; };
		    struct s19b *f19() { return malloc(sizeof(struct s19b)); }
		    void f19b(struct s19b *p) {}`,
		body:      `p := C.f19(); p.f = C.struct_s19a{[32769]*C.char{new(C.char)}}; C.f19b(p)`,
		fail:      true,
		expensive: true,
	},
	{
		// Similar case, with a source on the heap.
		name: "barriergcprogstructheap",
		c: `#include <stdlib.h>
		    struct s20a { char *a[32769]; };
		    struct s20b { struct s20a f; };
		    struct s20b *f20() { return malloc(sizeof(struct s20b)); }
		    void f20b(struct s20b *p) {}
		    void f20c(void *p) {}`,
		imports:   []string{"unsafe"},
		body:      `p := C.f20(); n := &C.struct_s20a{[32769]*C.char{new(C.char)}}; p.f = *n; C.f20b(p); n.a[0] = nil; C.f20c(unsafe.Pointer(n))`,
		fail:      true,
		expensive: true,
	},
	{
		// Exported functions may not return Go pointers.
		name: "export1",
		c: `#ifdef _WIN32
		    __declspec(dllexport)
			#endif
		    extern unsigned char *GoFn21();`,
		support: `//export GoFn21
		          func GoFn21() *byte { return new(byte) }`,
		body: `C.GoFn21()`,
		fail: true,
	},
	{
		// Returning a C pointer is fine.
		name: "exportok",
		c: `#include <stdlib.h>
		    #ifdef _WIN32
		    __declspec(dllexport)
			#endif
		    extern unsigned char *GoFn22();`,
		support: `//export GoFn22
		          func GoFn22() *byte { return (*byte)(C.malloc(1)) }`,
		body: `C.GoFn22()`,
	},
	{
		// Passing a Go string is fine.
		name: "passstring",
		c: `#include <stddef.h>
		    typedef struct { const char *p; ptrdiff_t n; } gostring23;
		    gostring23 f23(gostring23 s) { return s; }`,
		imports: []string{"unsafe"},
		body:    `s := "a"; r := C.f23(*(*C.gostring23)(unsafe.Pointer(&s))); if *(*string)(unsafe.Pointer(&r)) != s { panic(r) }`,
	},
	{
		// Passing a slice of Go strings fails.
		name:    "passstringslice",
		c:       `void f24(void *p) {}`,
		imports: []string{"strings", "unsafe"},
		support: `type S24 struct { a [1]string }`,
		body:    `s := S24{a:[1]string{strings.Repeat("a", 2)}}; C.f24(unsafe.Pointer(&s.a[0]))`,
		fail:    true,
	},
	{
		// Exported functions may not return strings.
		name:    "retstring",
		c:       `extern void f25();`,
		imports: []string{"strings"},
		support: `//export GoStr25
		          func GoStr25() string { return strings.Repeat("a", 2) }`,
		body: `C.f25()`,
		c1: `#include <stddef.h>
		     typedef struct { const char *p; ptrdiff_t n; } gostring25;
		     extern gostring25 GoStr25();
		     void f25() { GoStr25(); }`,
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
		    void f26(void* p) {}`,
		imports: []string{"unsafe"},
		support: `type S26 struct { p *int; a [8*8]byte; u uintptr }`,
		body:    `i := 0; p := &S26{u:uintptr(unsafe.Pointer(&i))}; q := (*S26)(C.malloc(C.size_t(unsafe.Sizeof(*p)))); *q = *p; C.f26(unsafe.Pointer(q))`,
		fail:    false,
	},
	{
		// Like ptrdata1, but with a type that uses a GC program.
		name: "ptrdata2",
		c: `#include <stdlib.h>
		    void f27(void* p) {}`,
		imports: []string{"unsafe"},
		support: `type S27 struct { p *int; a [32769*8]byte; q *int; u uintptr }`,
		body:    `i := 0; p := S27{u:uintptr(unsafe.Pointer(&i))}; q := (*S27)(C.malloc(C.size_t(unsafe.Sizeof(p)))); *q = p; C.f27(unsafe.Pointer(q))`,
		fail:    false,
	},
	{
		// Check deferred pointers when they are used, not
		// when the defer statement is run.
		name: "defer1",
		c:    `typedef struct s28 { int *p; } s28; void f28(s28 *ps) {}`,
		body: `p := &C.s28{}; defer C.f28(p); p.p = new(C.int)`,
		fail: true,
	},
	{
		// Check a pointer to a union if the union has any
		// pointer fields.
		name:    "union1",
		c:       `typedef union { char **p; unsigned long i; } u29; void f29(u29 *pu) {}`,
		imports: []string{"unsafe"},
		body:    `var b C.char; p := &b; C.f29((*C.u29)(unsafe.Pointer(&p)))`,
		fail:    true,
	},
	{
		// Don't check a pointer to a union if the union does
		// not have any pointer fields.
		// Like ptrdata1 above, the uintptr represents an
		// integer that happens to have the same
		// representation as a pointer.
		name:    "union2",
		c:       `typedef union { unsigned long i; } u39; void f39(u39 *pu) {}`,
		imports: []string{"unsafe"},
		body:    `var b C.char; p := &b; C.f39((*C.u39)(unsafe.Pointer(&p)))`,
		fail:    false,
	},
	{
		// Test preemption while entering a cgo call. Issue #21306.
		name:    "preemptduringcall",
		c:       `void f30() {}`,
		imports: []string{"runtime", "sync"},
		body:    `var wg sync.WaitGroup; wg.Add(100); for i := 0; i < 100; i++ { go func(i int) { for j := 0; j < 100; j++ { C.f30(); runtime.GOMAXPROCS(i) }; wg.Done() }(i) }; wg.Wait()`,
		fail:    false,
	},
	{
		// Test poller deadline with cgocheck=2.  Issue #23435.
		name:    "deadline",
		c:       `#define US31 10`,
		imports: []string{"os", "time"},
		body:    `r, _, _ := os.Pipe(); r.SetDeadline(time.Now().Add(C.US31 * time.Microsecond))`,
		fail:    false,
	},
	{
		// Test for double evaluation of channel receive.
		name:    "chanrecv",
		c:       `void f32(char** p) {}`,
		imports: []string{"time"},
		body:    `c := make(chan []*C.char, 2); c <- make([]*C.char, 1); go func() { time.Sleep(10 * time.Second); panic("received twice from chan") }(); C.f32(&(<-c)[0]);`,
		fail:    false,
	},
	{
		// Test that converting the address of a struct field
		// to unsafe.Pointer still just checks that field.
		// Issue #25941.
		name:    "structfield",
		c:       `void f33(void* p) {}`,
		imports: []string{"unsafe"},
		support: `type S33 struct { p *int; a [8]byte; u uintptr }`,
		body:    `s := &S33{p: new(int)}; C.f33(unsafe.Pointer(&s.a))`,
		fail:    false,
	},
	{
		// Test that converting multiple struct field
		// addresses to unsafe.Pointer still just checks those
		// fields. Issue #25941.
		name:    "structfield2",
		c:       `void f34(void* p, int r, void* s) {}`,
		imports: []string{"unsafe"},
		support: `type S34 struct { a [8]byte; p *int; b int64; }`,
		body:    `s := &S34{p: new(int)}; C.f34(unsafe.Pointer(&s.a), 32, unsafe.Pointer(&s.b))`,
		fail:    false,
	},
	{
		// Test that second argument to cgoCheckPointer is
		// evaluated when a deferred function is deferred, not
		// when it is run.
		name:    "defer2",
		c:       `void f35(char **pc) {}`,
		support: `type S35a struct { s []*C.char }; type S35b struct { ps *S35a }`,
		body:    `p := &S35b{&S35a{[]*C.char{nil}}}; defer C.f35(&p.ps.s[0]); p.ps = nil`,
		fail:    false,
	},
	{
		// Test that indexing into a function call still
		// examines only the slice being indexed.
		name:    "buffer",
		c:       `void f36(void *p) {}`,
		imports: []string{"bytes", "unsafe"},
		body:    `var b bytes.Buffer; b.WriteString("a"); C.f36(unsafe.Pointer(&b.Bytes()[0]))`,
		fail:    false,
	},
	{
		// Test that bgsweep releasing a finalizer is OK.
		name:    "finalizer",
		c:       `// Nothing to declare.`,
		imports: []string{"os"},
		support: `func open37() { os.Open(os.Args[0]) }; var G37 [][]byte`,
		body:    `for i := 0; i < 10000; i++ { G37 = append(G37, make([]byte, 4096)); if i % 100 == 0 { G37 = nil; open37() } }`,
		fail:    false,
	},
	{
		// Test that converting generated struct to interface is OK.
		name:    "structof",
		c:       `// Nothing to declare.`,
		imports: []string{"reflect"},
		support: `type MyInt38 int; func (i MyInt38) Get() int { return int(i) }; type Getter38 interface { Get() int }`,
		body:    `t := reflect.StructOf([]reflect.StructField{{Name: "MyInt38", Type: reflect.TypeOf(MyInt38(0)), Anonymous: true}}); v := reflect.New(t).Elem(); v.Interface().(Getter38).Get()`,
		fail:    false,
	},
	{
		// Test that a converted address of a struct field results
		// in a check for just that field and not the whole struct.
		name:    "structfieldcast",
		c:       `struct S40i { int i; int* p; }; void f40(struct S40i* p) {}`,
		support: `type S40 struct { p *int; a C.struct_S40i }`,
		body:    `s := &S40{p: new(int)}; C.f40((*C.struct_S40i)(&s.a))`,
		fail:    false,
	},
	{
		// Test that we handle unsafe.StringData.
		name:    "stringdata",
		c:       `void f41(void* p) {}`,
		imports: []string{"unsafe"},
		body:    `s := struct { a [4]byte; p *int }{p: new(int)}; str := unsafe.String(&s.a[0], 4); C.f41(unsafe.Pointer(unsafe.StringData(str)))`,
		fail:    false,
	},
	{
		name:    "slicedata",
		c:       `void f42(void* p) {}`,
		imports: []string{"unsafe"},
		body:    `s := []*byte{nil, new(byte)}; C.f42(unsafe.Pointer(unsafe.SliceData(s)))`,
		fail:    true,
	},
	{
		name:    "slicedata2",
		c:       `void f43(void* p) {}`,
		imports: []string{"unsafe"},
		body:    `s := struct { a [4]byte; p *int }{p: new(int)}; C.f43(unsafe.Pointer(unsafe.SliceData(s.a[:])))`,
		fail:    false,
	},
	{
		// Passing the address of an element of a pointer-to-array.
		name:    "arraypointer",
		c:       `void f44(void* p) {}`,
		imports: []string{"unsafe"},
		body:    `a := new([10]byte); C.f44(unsafe.Pointer(&a[0]))`,
		fail:    false,
	},
	{
		// Passing the address of an element of a pointer-to-array
		// that contains a Go pointer.
		name:    "arraypointer2",
		c:       `void f45(void** p) {}`,
		imports: []string{"unsafe"},
		body:    `i := 0; a := &[2]unsafe.Pointer{nil, unsafe.Pointer(&i)}; C.f45(&a[0])`,
		fail:    true,
		errTextRegexp: `.*argument of cgo function has Go pointer to unpinned Go unsafe pointer`,
	},
	{
		// Passing a Go map as argument to C.
		name:          "argmap",
		c:             `void f46(void* p) {}`,
		imports:       []string{"unsafe"},
		body:          `m := map[int]int{0: 1,}; C.f46(unsafe.Pointer(&m))`,
		fail:          true,
		errTextRegexp: `.*argument of cgo function has Go pointer to unpinned Go map`,
	},
	{
		// Returning a Go map to C.
		name: "retmap",
		c:    `extern void f47();`,
		support: `//export GoMap47
		          func GoMap47() map[int]int { return map[int]int{0: 1,} }`,
		body: `C.f47()`,
		c1: `extern void* GoMap47();
		     void f47() { GoMap47(); }`,
		fail:          true,
		errTextRegexp: `.*result of Go function GoMap47 called from cgo is unpinned Go map or points to unpinned Go map.*`,
	},
}

func TestPointerChecks(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	var gopath string
	var dir string
	if *tmp != "" {
		gopath = *tmp
		dir = ""
	} else {
		d, err := os.MkdirTemp("", filepath.Base(t.Name()))
		if err != nil {
			t.Fatal(err)
		}
		dir = d
		gopath = d
	}

	exe := buildPtrTests(t, gopath, false)
	exe2 := buildPtrTests(t, gopath, true)

	// We (TestPointerChecks) return before the parallel subtest functions do,
	// so we can't just defer os.RemoveAll(dir). Instead we have to wait for
	// the parallel subtests to finish. This code looks racy but is not:
	// the add +1 run in serial before testOne blocks. The -1 run in parallel
	// after testOne finishes.
	var pending int32
	for _, pt := range ptrTests {
		t.Run(pt.name, func(t *testing.T) {
			atomic.AddInt32(&pending, +1)
			defer func() {
				if atomic.AddInt32(&pending, -1) == 0 {
					os.RemoveAll(dir)
				}
			}()
			testOne(t, pt, exe, exe2)
		})
	}
}

func buildPtrTests(t *testing.T, gopath string, cgocheck2 bool) (exe string) {

	src := filepath.Join(gopath, "src", "ptrtest")
	if err := os.MkdirAll(src, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "go.mod"), []byte("module ptrtest\ngo 1.20"), 0666); err != nil {
		t.Fatal(err)
	}

	// Prepare two cgo inputs: one for standard cgo and one for //export cgo.
	// (The latter cannot have C definitions, only declarations.)
	var cgo1, cgo2 bytes.Buffer
	fmt.Fprintf(&cgo1, "package main\n\n/*\n")
	fmt.Fprintf(&cgo2, "package main\n\n/*\n")

	// C code
	for _, pt := range ptrTests {
		cgo := &cgo1
		if strings.Contains(pt.support, "//export") {
			cgo = &cgo2
		}
		fmt.Fprintf(cgo, "%s\n", pt.c)
		fmt.Fprintf(&cgo1, "%s\n", pt.c1)
	}
	fmt.Fprintf(&cgo1, "*/\nimport \"C\"\n\n")
	fmt.Fprintf(&cgo2, "*/\nimport \"C\"\n\n")

	// Imports
	did1 := make(map[string]bool)
	did2 := make(map[string]bool)
	did1["os"] = true // for ptrTestMain
	fmt.Fprintf(&cgo1, "import \"os\"\n")

	for _, pt := range ptrTests {
		did := did1
		cgo := &cgo1
		if strings.Contains(pt.support, "//export") {
			did = did2
			cgo = &cgo2
		}
		for _, imp := range pt.imports {
			if !did[imp] {
				did[imp] = true
				fmt.Fprintf(cgo, "import %q\n", imp)
			}
		}
	}

	// Func support and bodies.
	for _, pt := range ptrTests {
		cgo := &cgo1
		if strings.Contains(pt.support, "//export") {
			cgo = &cgo2
		}
		fmt.Fprintf(cgo, "%s\nfunc %s() {\n%s\n}\n", pt.support, pt.name, pt.body)
	}

	// Func list and main dispatch.
	fmt.Fprintf(&cgo1, "var funcs = map[string]func() {\n")
	for _, pt := range ptrTests {
		fmt.Fprintf(&cgo1, "\t%q: %s,\n", pt.name, pt.name)
	}
	fmt.Fprintf(&cgo1, "}\n\n")
	fmt.Fprintf(&cgo1, "%s\n", ptrTestMain)

	if err := os.WriteFile(filepath.Join(src, "cgo1.go"), cgo1.Bytes(), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "cgo2.go"), cgo2.Bytes(), 0666); err != nil {
		t.Fatal(err)
	}

	exeName := "ptrtest.exe"
	if cgocheck2 {
		exeName = "ptrtest2.exe"
	}
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", exeName)
	cmd.Dir = src
	cmd.Env = append(os.Environ(), "GOPATH="+gopath)

	// Set or remove cgocheck2 from the environment.
	goexperiment := strings.Split(os.Getenv("GOEXPERIMENT"), ",")
	if len(goexperiment) == 1 && goexperiment[0] == "" {
		goexperiment = nil
	}
	i := slices.Index(goexperiment, "cgocheck2")
	changed := false
	if cgocheck2 && i < 0 {
		goexperiment = append(goexperiment, "cgocheck2")
		changed = true
	} else if !cgocheck2 && i >= 0 {
		goexperiment = slices.Delete(goexperiment, i, i+1)
		changed = true
	}
	if changed {
		cmd.Env = append(cmd.Env, "GOEXPERIMENT="+strings.Join(goexperiment, ","))
	}

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}

	return filepath.Join(src, exeName)
}

const ptrTestMain = `
func main() {
	for _, arg := range os.Args[1:] {
		f := funcs[arg]
		if f == nil {
			panic("missing func "+arg)
		}
		f()
	}
}
`

var csem = make(chan bool, 16)

func testOne(t *testing.T, pt ptrTest, exe, exe2 string) {
	t.Parallel()

	// Run the tests in parallel, but don't run too many
	// executions in parallel, to avoid overloading the system.
	runcmd := func(cgocheck string) ([]byte, error) {
		csem <- true
		defer func() { <-csem }()
		x := exe
		if cgocheck == "2" {
			x = exe2
			cgocheck = "1"
		}
		cmd := exec.Command(x, pt.name)
		cmd.Env = append(os.Environ(), "GODEBUG=cgocheck="+cgocheck)
		return cmd.CombinedOutput()
	}

	if pt.expensive {
		buf, err := runcmd("1")
		if err != nil {
			t.Logf("%s", buf)
			if pt.fail {
				t.Fatalf("test marked expensive, but failed when not expensive: %v", err)
			} else {
				t.Errorf("failed unexpectedly with GODEBUG=cgocheck=1: %v", err)
			}
		}

	}

	cgocheck := ""
	if pt.expensive {
		cgocheck = "2"
	}

	buf, err := runcmd(cgocheck)

	var pattern string = pt.errTextRegexp
	if pt.errTextRegexp == "" {
		pattern = `.*unpinned Go.*`
	}

	if pt.fail {
		if err == nil {
			t.Logf("%s", buf)
			t.Fatalf("did not fail as expected")
		} else if ok, _ := regexp.Match(pattern, buf); !ok {
			t.Logf("%s", buf)
			t.Fatalf("did not print expected error (failed with %v)", err)
		}
	} else {
		if err != nil {
			t.Logf("%s", buf)
			t.Fatalf("failed unexpectedly: %v", err)
		}

		if !pt.expensive {
			// Make sure it passes with the expensive checks.
			buf, err := runcmd("2")
			if err != nil {
				t.Logf("%s", buf)
				t.Fatalf("failed unexpectedly with expensive checks: %v", err)
			}
		}
	}

	if pt.fail {
		buf, err := runcmd("0")
		if err != nil {
			t.Logf("%s", buf)
			t.Fatalf("failed unexpectedly with GODEBUG=cgocheck=0: %v", err)
		}
	}
}
