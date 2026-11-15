// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func shapesCgoErr(t *testing.T, input, src string) (string, map[string]string, string, error) {
	t.Helper()
	requireCgo(t)
	t.Setenv("GOEXPERIMENT", "logosfastcgo")
	work := t.TempDir()
	if err := os.WriteFile(filepath.Join(work, input), []byte(src), 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(work, "out")
	if err := os.Mkdir(out, 0755); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, cgoBin(t), "-objdir", out, "-importpath", "scratch/fast", filepath.Join(work, input))
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOTOOLCHAIN=local")
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	text := buf.String()
	files := map[string]string{}
	if err == nil {
		ents, rerr := os.ReadDir(out)
		if rerr != nil {
			t.Fatal(rerr)
		}
		for _, e := range ents {
			if e.IsDir() {
				continue
			}
			b, rerr := os.ReadFile(filepath.Join(out, e.Name()))
			if rerr != nil {
				t.Fatal(rerr)
			}
			files[e.Name()] = string(b)
		}
	}
	return out, files, text, err
}

func shapesCgoMulti(t *testing.T, inputs map[string]string) (string, map[string]string) {
	t.Helper()
	requireCgo(t)
	t.Setenv("GOEXPERIMENT", "logosfastcgo")
	work := t.TempDir()
	args := []string{}
	order := []string{"a.go", "b.go", "f.go", "f.lgo", "a.lgo", "b.lgo"}
	seen := map[string]bool{}
	for _, k := range order {
		if src, ok := inputs[k]; ok {
			p := filepath.Join(work, k)
			if err := os.WriteFile(p, []byte(src), 0644); err != nil {
				t.Fatal(err)
			}
			args = append(args, p)
			seen[k] = true
		}
	}
	for k, src := range inputs {
		if seen[k] {
			continue
		}
		p := filepath.Join(work, k)
		if err := os.WriteFile(p, []byte(src), 0644); err != nil {
			t.Fatal(err)
		}
		args = append(args, p)
	}
	out := filepath.Join(work, "out")
	if err := os.Mkdir(out, 0755); err != nil {
		t.Fatal(err)
	}
	full := append([]string{"-objdir", out, "-importpath", "scratch/fast"}, args...)
	text, err := runCmd(t, work, cgoBin(t), full...)
	if err != nil {
		t.Fatalf("cgo multi: %v\n%s", err, text)
	}
	files := map[string]string{}
	ents, err := os.ReadDir(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(out, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		files[e.Name()] = string(b)
	}
	return out, files
}

func shapesCgoArgs(t *testing.T, input, src string, gccopts ...string) (string, map[string]string) {
	t.Helper()
	requireCgo(t)
	t.Setenv("GOEXPERIMENT", "logosfastcgo")
	work := t.TempDir()
	if err := os.WriteFile(filepath.Join(work, input), []byte(src), 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(work, "out")
	if err := os.Mkdir(out, 0755); err != nil {
		t.Fatal(err)
	}
	full := []string{"-objdir", out, "-importpath", "scratch/fast", "--"}
	full = append(full, gccopts...)
	full = append(full, filepath.Join(work, input))
	text, err := runCmd(t, work, cgoBin(t), full...)
	if err != nil {
		t.Fatalf("cgo args: %v\n%s", err, text)
	}
	files := map[string]string{}
	ents, err := os.ReadDir(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(out, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		files[e.Name()] = string(b)
	}
	return out, files
}

func shapesRunCmd(t *testing.T, dir string, env []string, name string, args ...string) (string, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOTOOLCHAIN=local")
	cmd.Env = append(cmd.Env, env...)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	return buf.String(), err
}

func shapesGotypes(t *testing.T, files map[string]string) string {
	t.Helper()
	g, ok := files["_cgo_gotypes.go"]
	if !ok {
		t.Fatal("no _cgo_gotypes.go emitted")
	}
	return g
}

func shapesCgo1(t *testing.T, files map[string]string) string {
	t.Helper()
	var sb strings.Builder
	n := 0
	for name, body := range files {
		if strings.HasSuffix(name, ".cgo1.go") {
			sb.WriteString(body)
			n++
		}
	}
	if n == 0 {
		t.Fatal("no .cgo1.go emitted")
	}
	return sb.String()
}

func shapesCgo2(t *testing.T, files map[string]string) string {
	t.Helper()
	var sb strings.Builder
	n := 0
	for name, body := range files {
		if strings.HasSuffix(name, ".cgo2.c") {
			sb.WriteString(body)
			n++
		}
	}
	if n == 0 {
		t.Fatal("no .cgo2.c emitted")
	}
	return sb.String()
}

func shapesLineHasPrefix(line string) bool {
	rest := line
	for {
		idx := strings.Index(rest, "_cgo_")
		if idx < 0 {
			return false
		}
		cand := rest[idx+5:]
		rest = rest[idx+5:]
		if len(cand) < 13 || cand[12] != '_' {
			continue
		}
		hex := cand[:12]
		good := true
		for i := range 12 {
			c := hex[i]
			if c >= '0' && c <= '9' {
				continue
			}
			if c >= 'a' && c <= 'f' {
				continue
			}
			good = false
			break
		}
		if good {
			return true
		}
	}
}
func shapesPrefixOK(t *testing.T, gotypes string) {
	t.Helper()
	for line := range strings.SplitSeq(gotypes, "\n") {
		if shapesLineHasPrefix(line) {
			return
		}
	}
	t.Fatal("no _cgo_<12hex>_ prefix found in gotypes")
}

func TestFastcallShapesErrnoEmission(t *testing.T) {
	const src = `package fast

/*
#include <errno.h>
int e_fast(int v) { if (v < 0) { errno = 22; return -1; } errno = 0; return v * 2; }
int e_slow(int v) { if (v < 0) { errno = 22; return -1; } errno = 0; return v * 2; }
void ev_fast(void) {}
void ev_slow(void) {}
#cgo noescape e_fast
#cgo nocallback e_fast
#cgo noescape ev_fast
#cgo nocallback ev_fast
*/
import "C"

func F(v int32) (int32, error) { r, err := C.e_fast(C.int(v)); return int32(r), err }

func S(v int32) (int32, error) { r, err := C.e_slow(C.int(v)); return int32(r), err }

func FV() { C.ev_fast() }

func SV() { C.ev_slow() }
`
	_, files, _, err := shapesCgoErr(t, "f.go", src)
	if err != nil {
		t.Fatalf("errno cgo failed")
	}
	g := shapesGotypes(t, files)
	if !strings.Contains(g, "errno := _cgo_runtime_cgocall_fast(") {
		t.Fatal("fast errno wrapper missing")
	}
	if !strings.Contains(g, "errno := _cgo_runtime_cgocall(") {
		t.Fatal("slow errno wrapper missing")
	}
	c := shapesCgo2(t, files)
	if strings.Count(c, "errno = 0;") < 2 {
		t.Fatalf("errno zeroing = %d, want at least 2", strings.Count(c, "errno = 0;"))
	}
	if strings.Count(c, "return _cgo_errno;") < 2 {
		t.Fatalf("errno return = %d, want at least 2", strings.Count(c, "return _cgo_errno;"))
	}
}

func TestFastcallShapesFnPtrEmission(t *testing.T) {
	const src = `package fast

/*
typedef int (*fn_t)(int);
int shape_double(int x) { return x * 2; }
int a_fast(fn_t f, int x) { return f(x); }
int a_slow(fn_t f, int x) { return f(x); }
#cgo noescape a_fast
#cgo nocallback a_fast
*/
import "C"

func F() int32 {
fn := C.fn_t(C.shape_double)
return int32(C.a_fast(fn, 21))
}

func S() int32 {
fn := C.fn_t(C.shape_double)
return int32(C.a_slow(fn, 21))
}
`
	_, files, _, err := shapesCgoErr(t, "f.go", src)
	if err != nil {
		t.Fatalf("fnptr cgo failed")
	}
	g := shapesGotypes(t, files)
	if got := countCallLines(g, "_cgo_runtime_cgocall_fast"); got != 1 {
		t.Fatalf("fast calls = %d, want 1", got)
	}
	if got := countCallLines(g, "_cgo_runtime_cgocall"); got != 1 {
		t.Fatalf("plain calls = %d, want 1", got)
	}
}

func TestFastcallShapesStructEnumEmission(t *testing.T) {
	const src = `package fast

/*
struct ShapePoint { int x; int y; };
struct ShapeHasPP { int **pp; int n; };
enum ShapeColor { SHAPE_RED = 1, SHAPE_GREEN = 2, SHAPE_BLUE = 3 };
int p_fast(struct ShapePoint p) { return p.x + p.y; }
int p_slow(struct ShapePoint p) { return p.x + p.y; }
int h_fast(struct ShapeHasPP p) { return p.n; }
int h_slow(struct ShapeHasPP p) { return p.n; }
int c_fast(enum ShapeColor c) { return (int)c; }
int c_slow(enum ShapeColor c) { return (int)c; }
#cgo noescape p_fast
#cgo nocallback p_fast
#cgo noescape h_fast
#cgo nocallback h_fast
#cgo noescape c_fast
#cgo nocallback c_fast
*/
import "C"

func F() int32 {
pt := C.struct_ShapePoint{x: 1, y: 2}
hp := C.struct_ShapeHasPP{n: 1}
return int32(C.p_fast(pt)) + int32(C.h_fast(hp)) + int32(C.c_fast(2))
}

func S() int32 {
pt := C.struct_ShapePoint{x: 1, y: 2}
hp := C.struct_ShapeHasPP{n: 1}
return int32(C.p_slow(pt)) + int32(C.h_slow(hp)) + int32(C.c_slow(2))
}
`
	_, files, _, err := shapesCgoErr(t, "f.go", src)
	if err != nil {
		t.Fatalf("struct cgo failed")
	}
	g := shapesGotypes(t, files)
	if got := countCallLines(g, "_cgo_runtime_cgocall_fast"); got != 3 {
		t.Fatalf("fast calls = %d, want 3", got)
	}
	c1 := shapesCgo1(t, files)
	if got := strings.Count(c1, "_cgoCheckPointer(_cgo"); got != 2 {
		t.Fatalf("pointer checks = %d, want 2", got)
	}
}

func TestFastcallShapesStaticInlineEmission(t *testing.T) {
	const src = `package fast

/*
static inline int shape_helper(int x) { return x * 3 + 1; }
int i_fast(int x) { return shape_helper(x); }
int i_slow(int x) { return shape_helper(x); }
#cgo noescape i_fast
#cgo nocallback i_fast
*/
import "C"

func F() int32 { return int32(C.i_fast(7)) }

func S() int32 { return int32(C.i_slow(7)) }
`
	_, files, _, err := shapesCgoErr(t, "f.go", src)
	if err != nil {
		t.Fatalf("inline cgo failed")
	}
	g := shapesGotypes(t, files)
	if got := countCallLines(g, "_cgo_runtime_cgocall_fast"); got != 1 {
		t.Fatalf("fast calls = %d, want 1", got)
	}
	c := shapesCgo2(t, files)
	if !strings.Contains(c, "shape_helper") {
		t.Fatal("helper missing from C output")
	}
	if !strings.Contains(c, "CGO_NO_SANITIZE_THREAD") {
		t.Fatal("sanitize guard missing from C output")
	}
}

func TestFastcallShapesCollisionEmission(t *testing.T) {
	const src = `package fast

/*
int cgoCheckPointer(int x) { return x + 1; }
int cgocall(int x) { return x + 2; }
#cgo noescape cgoCheckPointer
#cgo nocallback cgoCheckPointer
#cgo noescape cgocall
#cgo nocallback cgocall
*/
import "C"

func F() int32 { return int32(C.cgoCheckPointer(41)) }

func G() int32 { return int32(C.cgocall(40)) }
`
	_, files, _, err := shapesCgoErr(t, "f.go", src)
	if err != nil {
		t.Fatalf("collision cgo failed")
	}
	g := shapesGotypes(t, files)
	if got := countCallLines(g, "_cgo_runtime_cgocall_fast"); got != 2 {
		t.Fatalf("fast calls = %d, want 2", got)
	}
	c1 := shapesCgo1(t, files)
	if strings.Contains(c1, "_cgoCheckPointer(_cgo") {
		t.Fatal("spurious pointer check for collision funcs")
	}
}

func TestFastcallShapesCrossFile(t *testing.T) {
	a := `package fast

/*
int shape_xf(int a, int b) { return a + b; }
#cgo noescape shape_xf
#cgo nocallback shape_xf
*/
import "C"

func A(a, b int32) int32 { return int32(C.shape_xf(C.int(a), C.int(b))) }
`
	b := `package fast

/*
int shape_xf(int a, int b);
*/
import "C"

func B(a, b int32) int32 { return int32(C.shape_xf(C.int(a), C.int(b))) }
`
	_, files := shapesCgoMulti(t, map[string]string{"a.go": a, "b.go": b})
	g := shapesGotypes(t, files)
	if got := countCallLines(g, "_cgo_runtime_cgocall_fast"); got != 1 {
		t.Fatalf("fast calls = %d, want 1", got)
	}
	names := 0
	for name := range files {
		if strings.HasSuffix(name, ".cgo1.go") {
			names++
		}
	}
	if names != 2 {
		t.Fatalf("cgo1 files = %d, want 2", names)
	}
	c := shapesCgo2(t, files)
	if got := strings.Count(c, "shape_xf"); got < 3 {
		t.Fatalf("shape_xf mentions = %d, want at least 3", got)
	}
}

func TestFastcallShapesMarkSplit(t *testing.T) {
	a := `package fast

/*
int shape_m1(int a) { return a + 1; }
#cgo noescape shape_m1
#cgo nocallback shape_m1
*/
import "C"

func A() int32 { return int32(C.shape_m1(1)) }
`
	b := `package fast

/*
int shape_m2(int a) { return a + 1; }
*/
import "C"

func B() int32 { return int32(C.shape_m2(1)) }
`
	_, files := shapesCgoMulti(t, map[string]string{"a.go": a, "b.go": b})
	g := shapesGotypes(t, files)
	if got := countCallLines(g, "_cgo_runtime_cgocall_fast"); got != 1 {
		t.Fatalf("fast calls = %d, want 1", got)
	}
	plain := 0
	for line := range strings.SplitSeq(g, "\n") {
		if strings.HasPrefix(line, "\t") && strings.Contains(line, "_cgo_runtime_cgocall(") && !strings.Contains(line, "_fast(") {
			plain++
		}
	}
	if plain != 1 {
		t.Fatalf("plain calls = %d, want 1", plain)
	}
}

func TestFastcallShapesLgoMixed(t *testing.T) {
	const src = `package fast

/*
#include <stdint.h>
//logos:cgo_fastcall ladd
#cgo noescape sfunc
#cgo nocallback sfunc
int ladd(int a, int b) { return a + b; }
int sfunc(int a, int b) { return a - b; }
*/
import "C"

func Add(a, b int32) int32 { return int32(C.ladd(C.int(a), C.int(b))) }

func Sub(a, b int32) int32 { return int32(C.sfunc(C.int(a), C.int(b))) }
`
	_, files, _, err := shapesCgoErr(t, "f.lgo", src)
	if err != nil {
		t.Fatalf("lgo mixed cgo failed")
	}
	g := shapesGotypes(t, files)
	if got := countCallLines(g, "_cgo_runtime_cgocall_fast"); got != 2 {
		t.Fatalf("fast calls = %d, want 2", got)
	}
}

func TestFastcallShapesMalformed(t *testing.T) {
	cases := map[string]string{
		"mal0":   "package fast\n/*\n#include <stdint.h>\n//logos:cgo_fastcall\nint ladd(int a, int b) { return a + b; }\n*/\nimport \"C\"\nfunc Add(a, b int32) int32 { return int32(C.ladd(C.int(a), C.int(b))) }\n",
		"mal3":   "package fast\n/*\n#include <stdint.h>\n//logos:cgo_fastcall ladd extra\nint ladd(int a, int b) { return a + b; }\n*/\nimport \"C\"\nfunc Add(a, b int32) int32 { return int32(C.ladd(C.int(a), C.int(b))) }\n",
		"nonpre": "package fast\n/*\n#include <stdint.h>\nint ladd(int a, int b) { return a + b; }\n*/\nimport \"C\"\n//logos:cgo_fastcall ladd\nfunc Add(a, b int32) int32 { return int32(C.ladd(C.int(a), C.int(b))) }\n",
	}
	for name, src := range cases {
		_, files, _, err := shapesCgoErr(t, "f.lgo", src)
		if err != nil {
			t.Fatalf("%s cgo failed", name)
		}
		g := shapesGotypes(t, files)
		if got := countCallLines(g, "_cgo_runtime_cgocall_fast"); got != 0 {
			t.Fatalf("%s fast calls = %d, want 0", name, got)
		}
		plain := 0
		for line := range strings.SplitSeq(g, "\n") {
			if strings.HasPrefix(line, "\t") && strings.Contains(line, "_cgo_runtime_cgocall(") && !strings.Contains(line, "_fast(") {
				plain++
			}
		}
		if plain != 1 {
			t.Fatalf("%s plain calls = %d, want 1", name, plain)
		}
	}
}

func TestFastcallShapesGhostPair(t *testing.T) {
	const src = `package fast

/*
#include <stdint.h>
//logos:cgo_fastcall ghost_fn
int ladd(int a, int b) { return a + b; }
*/
import "C"

func Add(a, b int32) int32 { return int32(C.ladd(C.int(a), C.int(b))) }
`
	_, _, text, err := shapesCgoErr(t, "f.lgo", src)
	if err == nil {
		t.Fatal("ghost marker succeeded, want failure")
	}
	if !strings.Contains(text, "#cgo noescape ghost_fn: no matched C function") {
		t.Fatalf("missing noescape ghost error:\n%s", text)
	}
	if !strings.Contains(text, "#cgo nocallback ghost_fn: no matched C function") {
		t.Fatalf("missing nocallback ghost error:\n%s", text)
	}
}

func TestFastcallShapesVariadicRejected(t *testing.T) {
	const marked = `package fast

/*
#include <stdarg.h>
int v_fast(int n, ...) { return n; }
#cgo noescape v_fast
#cgo nocallback v_fast
*/
import "C"

func F() int32 { return int32(C.v_fast(1)) }
`
	const unmarked = `package slow

/*
#include <stdarg.h>
int v_slow(int n, ...) { return n; }
*/
import "C"

func F() int32 { return int32(C.v_slow(1)) }
`
	_, _, text, err := shapesCgoErr(t, "f.go", marked)
	if err == nil {
		t.Fatal("marked variadic succeeded, want failure")
	}
	if !strings.Contains(text, "unexpected type") {
		t.Fatalf("marked variadic lacks unexpected type:\n%s", text)
	}
	_, _, text, err = shapesCgoErr(t, "f.go", unmarked)
	if err == nil {
		t.Fatal("unmarked variadic succeeded, want failure")
	}
	if !strings.Contains(text, "unexpected type") {
		t.Fatalf("unmarked variadic lacks unexpected type:\n%s", text)
	}
}

func TestFastcallShapesPointerParity(t *testing.T) {
	const src = `package fast

/*
int r_pod_fast(int *p) { return *p; }
int r_pod_slow(int *p) { return *p; }
int r_pp_fast(int **pp) { return **pp; }
int r_pp_slow(int **pp) { return **pp; }
int s_fast(unsigned char *p, int n) { int s = 0; for (int i = 0; i < n; i++) { s += p[i]; } return s; }
int s_slow(unsigned char *p, int n) { int s = 0; for (int i = 0; i < n; i++) { s += p[i]; } return s; }
#cgo noescape r_pod_fast
#cgo nocallback r_pod_fast
#cgo noescape r_pp_fast
#cgo nocallback r_pp_fast
#cgo noescape s_fast
#cgo nocallback s_fast
*/
import "C"

import "unsafe"

func FP(p *int32) int32 { return int32(C.r_pod_fast((*C.int)(unsafe.Pointer(p)))) }
func SP(p *int32) int32 { return int32(C.r_pod_slow((*C.int)(unsafe.Pointer(p)))) }
func FPP(pp **C.int) int32 { return int32(C.r_pp_fast(pp)) }
func SPP(pp **C.int) int32 { return int32(C.r_pp_slow(pp)) }

func FS(b []byte) int32 {
if len(b) == 0 {
return 0
}
return int32(C.s_fast((*C.uchar)(unsafe.Pointer(&b[0])), C.int(len(b))))
}

func SS(b []byte) int32 {
if len(b) == 0 {
return 0
}
return int32(C.s_slow((*C.uchar)(unsafe.Pointer(&b[0])), C.int(len(b))))
}
`
	_, files, _, err := shapesCgoErr(t, "f.go", src)
	if err != nil {
		t.Fatalf("pointer cgo failed")
	}
	c1 := shapesCgo1(t, files)
	if got := strings.Count(c1, "_cgoCheckPointer(_cgo"); got != 2 {
		t.Fatalf("pointer checks = %d, want 2", got)
	}
}

func TestFastcallShapesSanitizerEmission(t *testing.T) {
	const src = `package fast

/*
int t_fast(int a) { return a; }
int t_slow(int a) { return a; }
#cgo noescape t_fast
#cgo nocallback t_fast
*/
import "C"

func F() int32 { return int32(C.t_fast(1)) }

func S() int32 { return int32(C.t_slow(1)) }
`
	_, files, _, err := shapesCgoErr(t, "f.go", src)
	if err != nil {
		t.Fatalf("sanitizer cgo failed")
	}
	c := shapesCgo2(t, files)
	if strings.Count(c, "CGO_NO_SANITIZE_THREAD") < 3 {
		t.Fatalf("sanitize guards = %d, want at least 3", strings.Count(c, "CGO_NO_SANITIZE_THREAD"))
	}
	if strings.Count(c, "_cgo_tsan_acquire()") < 2 {
		t.Fatalf("tsan acquire = %d, want at least 2", strings.Count(c, "_cgo_tsan_acquire()"))
	}
	if strings.Count(c, "_cgo_tsan_release()") < 2 {
		t.Fatalf("tsan release = %d, want at least 2", strings.Count(c, "_cgo_tsan_release()"))
	}
	if strings.Count(c, "_cgo_msan_write(") < 2 {
		t.Fatalf("msan write = %d, want at least 2", strings.Count(c, "_cgo_msan_write("))
	}
	_, tfiles := shapesCgoArgs(t, "f.go", src, "-fsanitize=thread")
	tg := shapesGotypes(t, tfiles)
	if got := countCallLines(tg, "_cgo_runtime_cgocall_fast"); got != 1 {
		t.Fatalf("tsan fast calls = %d, want 1", got)
	}
	tc := shapesCgo2(t, tfiles)
	if !strings.Contains(tc, "__tsan_acquire") {
		t.Fatal("tsan prolog missing under -fsanitize=thread")
	}
}

func TestFastcallShapesLayoutCacheKey(t *testing.T) {
	const marked = `package fast

/*
int l_fast(int a) { return a; }
#cgo noescape l_fast
#cgo nocallback l_fast
*/
import "C"

func F() int32 { return int32(C.l_fast(1)) }
`
	const unmarked = `package fast

/*
int l_fast(int a) { return a; }
*/
import "C"

func F() int32 { return int32(C.l_fast(1)) }
`
	_, mfiles, _, merr := shapesCgoErr(t, "f.go", marked)
	if merr != nil {
		t.Fatalf("marked cgo failed")
	}
	_, ufiles, _, uerr := shapesCgoErr(t, "f.go", unmarked)
	if uerr != nil {
		t.Fatalf("unmarked cgo failed")
	}
	mnames := map[string]bool{}
	for name := range mfiles {
		if strings.Contains(name, "fast") {
			t.Fatalf("generated name %q contains fast", name)
		}
		mnames[name] = true
	}
	for name := range ufiles {
		if !mnames[name] {
			t.Fatalf("unmarked file %q missing from marked set", name)
		}
		if strings.Contains(name, "fast") {
			t.Fatalf("generated name %q contains fast", name)
		}
	}
	if _, ok := mfiles["_cgo_gotypes.go"]; !ok {
		t.Fatal("no _cgo_gotypes.go in marked set")
	}
	shapesPrefixOK(t, shapesGotypes(t, mfiles))
	shapesPrefixOK(t, shapesGotypes(t, ufiles))
	mg := shapesGotypes(t, mfiles)
	if !strings.Contains(mg, "func _cgo_runtime_cgocall(unsafe.Pointer, uintptr) int32") {
		t.Fatal("plain trampoline decl missing")
	}
	if !strings.Contains(mg, "func _cgo_runtime_cgocall_fast(unsafe.Pointer, uintptr) int32") {
		t.Fatal("fast trampoline decl missing")
	}
}

func shapesBuildWith(t *testing.T, args []string, env []string) string {
	t.Helper()
	requireCgo(t)
	t.Setenv("GOEXPERIMENT", "logosfastcgo")
	dir := filepath.Join("testdata", "shapes")
	bin := filepath.Join(t.TempDir(), "shapes.exe")
	full := append([]string{"build"}, args...)
	full = append(full, "-o", bin, ".")
	out, err := shapesRunCmd(t, dir, env, goBin(t), full...)
	if err != nil {
		t.Fatalf("shapes build %q %q: %v\n%s", args, env, err, out)
	}
	return bin
}
func shapesBuild(t *testing.T) string {
	t.Helper()
	return shapesBuildWith(t, nil, nil)
}
func shapesBuildRace(t *testing.T) string {
	t.Helper()
	return shapesBuildWith(t, []string{"-race"}, nil)
}
func shapesBuildUbsan(t *testing.T) string {
	t.Helper()
	return shapesBuildWith(t, nil, []string{"CGO_CFLAGS=-fsanitize=undefined", "CGO_LDFLAGS=-fsanitize=undefined"})
}

func TestFastcallShapesEndToEnd(t *testing.T) {
	bin := shapesBuild(t)
	out, err := shapesRunCmd(t, "", nil, bin)
	if err != nil {
		t.Fatalf("shapes run: %v\n%s", err, out)
	}
	if strings.TrimSpace(out) != "SHAPES OK" {
		t.Fatalf("shapes output = %q, want %q", out, "SHAPES OK")
	}
	out, err = shapesRunCmd(t, "", nil, bin, "plain")
	if err != nil {
		t.Fatalf("shapes plain run: %v\n%s", err, out)
	}
	if strings.TrimSpace(out) != "SHAPES OK" {
		t.Fatalf("shapes plain output = %q, want %q", out, "SHAPES OK")
	}
}

func TestFastcallShapesRace(t *testing.T) {
	bin := shapesBuildRace(t)
	out, err := shapesRunCmd(t, "", nil, bin)
	if err != nil {
		t.Fatalf("shapes race run: %v\n%s", err, out)
	}
	if strings.TrimSpace(out) != "SHAPES OK" {
		t.Fatalf("shapes race output = %q, want %q", out, "SHAPES OK")
	}
	out, err = shapesRunCmd(t, "", nil, bin, "plain")
	if err != nil {
		t.Fatalf("shapes race plain run: %v\n%s", err, out)
	}
	if strings.TrimSpace(out) != "SHAPES OK" {
		t.Fatalf("shapes race plain output = %q, want %q", out, "SHAPES OK")
	}
}

func TestFastcallShapesUbsan(t *testing.T) {
	bin := shapesBuildUbsan(t)
	out, err := shapesRunCmd(t, "", nil, bin)
	if err != nil {
		t.Fatalf("shapes ubsan run: %v\n%s", err, out)
	}
	if strings.TrimSpace(out) != "SHAPES OK" {
		t.Fatalf("shapes ubsan output = %q, want %q", out, "SHAPES OK")
	}
}
