// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import (
	"internal/goexperiment"
	"internal/race"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const cgoOptSource = `package cgoopt

/*
#include <errno.h>

typedef struct { int x; double y; } opt_pair;

#cgo nocallback opt_scalar
unsigned opt_scalar(unsigned v) { return v + 1; }
#cgo nocallback opt_pointer
int *opt_pointer(int *p) { return p; }
#cgo nocallback opt_pair_result
opt_pair opt_pair_result(int v) { opt_pair p = {v, 2.5}; return p; }
#cgo nocallback opt_errno
int opt_errno(int v) { errno = ERANGE; return v; }
#cgo nocallback opt_void
void opt_void(int v) { (void)v; }
#cgo nocallback opt_noarg
int opt_noarg(void) { return 7; }
#cgo nocallback opt_nop
void opt_nop(void) {}
#cgo noescape opt_both
#cgo nocallback opt_both
int opt_both(int *p) { return *p; }
#cgo noescape opt_noescape
int opt_noescape(int *p) { return *p; }
int opt_unmarked(int *p) { return *p; }
*/
import "C"

func Scalar(v C.uint) C.uint { return C.opt_scalar(v) }
func Pointer(p *C.int) *C.int { return C.opt_pointer(p) }
func Pair(v C.int) C.opt_pair { return C.opt_pair_result(v) }
func Errno(v C.int) (C.int, error) { result, err := C.opt_errno(v); return result, err }
func Void(v C.int) { C.opt_void(v) }
func Noarg() C.int { return C.opt_noarg() }
func Nop() { C.opt_nop() }
func Both(p *C.int) C.int { return C.opt_both(p) }
func Noescape(p *C.int) C.int { return C.opt_noescape(p) }
func Unmarked(p *C.int) C.int { return C.opt_unmarked(p) }
`

func cgoOptWrapper(t *testing.T, source, name string) string {
	t.Helper()
	start := strings.Index(source, name+"(")
	if start < 0 {
		t.Fatalf("missing wrapper %s", name)
	}
	end := strings.Index(source[start:], "\n}\n")
	if end < 0 {
		t.Fatalf("unterminated wrapper %s", name)
	}
	return source[start : start+end+3]
}

func TestCgoOptEmission(t *testing.T) {
	settings := []struct {
		name string
		exp  string
		opt  bool
		fast bool
	}{
		{"off", "nologoscgoopt,nologosfastcgo", false, false},
		{"ordinary", "logoscgoopt,nologosfastcgo", true, false},
		{"fast", "nologoscgoopt,logosfastcgo", false, true},
		{"both", "logoscgoopt,logosfastcgo", true, true},
	}
	cases := []struct {
		name       string
		nocallback bool
		noescape   bool
		result     bool
		params     int
	}{
		{"_Cfunc_opt_scalar", true, false, true, 1},
		{"_Cfunc_opt_pointer", true, false, true, 1},
		{"_Cfunc_opt_pair_result", true, false, true, 1},
		{"_C2func_opt_errno", true, false, true, 1},
		{"_Cfunc_opt_void", true, false, false, 1},
		{"_Cfunc_opt_noarg", true, false, true, 0},
		{"_Cfunc_opt_nop", true, false, false, 0},
		{"_Cfunc_opt_both", true, true, true, 1},
		{"_Cfunc_opt_noescape", false, true, true, 1},
		{"_Cfunc_opt_unmarked", false, false, true, 1},
	}
	for _, setting := range settings {
		for _, ext := range []string{".go", ".lgo"} {
			t.Run(setting.name+ext, func(t *testing.T) {
				t.Setenv("GOEXPERIMENT", setting.exp)
				_, files := cgoOutputs(t, "ordinary"+ext, cgoOptSource)
				goSource, cSource := shapesGotypes(t, files), shapesCgo2(t, files)
				for _, tc := range cases {
					t.Run(tc.name, func(t *testing.T) {
						goBody := cgoOptWrapper(t, goSource, tc.name)
						cBody := cgoOptWrapper(t, cSource, tc.name)
						fast := setting.fast && tc.nocallback && tc.noescape
						stable := tc.nocallback && (setting.opt || fast)
						lookups := 0
						if tc.result && !stable {
							lookups = 2
						}
						if got := countCallLines(cBody, "_cgo_topofstack"); got != lookups {
							t.Errorf("stack-top lookups = %d, want %d\n%s", got, lookups, cBody)
						}
						if got := strings.Contains(cBody, "_cgo_stktop"); got != (lookups != 0) {
							t.Errorf("stack relocation = %v, want %v", got, lookups != 0)
						}
						plainCalls, fastCalls, guards := 1, 0, 0
						if fast {
							plainCalls, fastCalls = 0, 1
						} else if tc.nocallback {
							guards = 2
						}
						use, keepalive := tc.params, 0
						if tc.nocallback && tc.noescape {
							use, keepalive = 0, tc.params
						}
						for _, call := range []struct {
							name string
							want int
						}{
							{"_cgo_runtime_cgocall", plainCalls},
							{"_cgo_runtime_cgocall_fast", fastCalls},
							{"_Cgo_no_callback", guards},
							{"_Cgo_use", use},
							{"_Cgo_keepalive", keepalive},
						} {
							if got := countCallLines(goBody, call.name); got != call.want {
								t.Errorf("%s calls = %d, want %d\n%s", call.name, got, call.want, goBody)
							}
						}
						for _, call := range []string{"_cgo_tsan_acquire", "_cgo_tsan_release"} {
							if got := countCallLines(cBody, call); got != 1 {
								t.Errorf("%s calls = %d, want 1", call, got)
							}
						}
						if got := strings.Contains(cBody, "_cgo_a->r = _cgo_r;"); got != tc.result {
							t.Errorf("result store = %v, want %v", got, tc.result)
						}
						if got := strings.Contains(cBody, "_cgo_msan_write(&_cgo_a->r, sizeof(_cgo_a->r));"); got != tc.result {
							t.Errorf("result sanitizer write = %v, want %v", got, tc.result)
						}
						if tc.name == "_C2func_opt_errno" {
							for _, statement := range []string{"errno = 0;", "_cgo_errno = errno;", "return _cgo_errno;"} {
								if !strings.Contains(cBody, statement) {
									t.Errorf("missing errno statement %q", statement)
								}
							}
							if !strings.Contains(goBody, "syscall.Errno(errno)") {
								t.Error("missing Go errno result")
							}
						}
					})
				}
			})
		}
	}
}

func TestCgoOptLateDirective(t *testing.T) {
	requireCgo(t)
	const declaration = `package cgoopt
/*
int opt_cross(int);
*/
import "C"
func A(v C.int) C.int { return C.opt_cross(v) }
`
	const directive = `package cgoopt
/*
#cgo nocallback opt_cross
int opt_cross(int v) { return v + 1; }
*/
import "C"
func B(v C.int) C.int { return C.opt_cross(v) }
`
	for _, early := range []bool{false, true} {
		name := "late"
		if early {
			name = "early"
		}
		t.Run(name, func(t *testing.T) {
			t.Setenv("GOEXPERIMENT", "logoscgoopt,nologosfastcgo")
			dir := t.TempDir()
			for name, source := range map[string]string{"a.go": declaration, "b.go": directive} {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0644); err != nil {
					t.Fatal(err)
				}
			}
			first, second, lookups := "a.go", "b.go", 0
			if early {
				first, second, lookups = "b.go", "a.go", 0
			}
			out, err := runCmd(t, dir, cgoBin(t), "-objdir", dir, first, second)
			if err != nil {
				t.Fatalf("cgo: %v\n%s", err, out)
			}
			cSource, err := os.ReadFile(filepath.Join(dir, strings.TrimSuffix(first, ".go")+".cgo2.c"))
			if err != nil {
				t.Fatal(err)
			}
			body := cgoOptWrapper(t, string(cSource), "_Cfunc_opt_cross")
			if got := countCallLines(body, "_cgo_topofstack"); got != lookups {
				t.Errorf("stack-top lookups = %d, want %d", got, lookups)
			}
			goSource, err := os.ReadFile(filepath.Join(dir, "_cgo_gotypes.go"))
			if err != nil {
				t.Fatal(err)
			}
			body = cgoOptWrapper(t, string(goSource), "_Cfunc_opt_cross")
			if countCallLines(body, "_Cgo_no_callback") != 2 || countCallLines(body, "_cgo_runtime_cgocall") != 1 {
				t.Fatalf("final Go wrapper lost ordinary call or callback guards\n%s", body)
			}
		})
	}
}

func TestCgoOptGccgoUnchanged(t *testing.T) {
	requireCgo(t)
	gccgo := os.Getenv("GCCGO")
	if gccgo == "" {
		gccgo = "gccgo"
	}
	testenv.MustHaveExecPath(t, gccgo)
	dir := t.TempDir()
	input := filepath.Join(dir, "ordinary.go")
	if err := os.WriteFile(input, []byte(cgoOptSource), 0644); err != nil {
		t.Fatal(err)
	}
	var baseline []byte
	for _, experiment := range []string{"nologoscgoopt,nologosfastcgo", "logoscgoopt,logosfastcgo"} {
		t.Setenv("GOEXPERIMENT", experiment)
		out, err := runCmd(t, dir, cgoBin(t), "-gccgo", "-objdir", dir, input)
		if err != nil {
			t.Fatalf("cgo -gccgo with %s: %v\n%s", experiment, err, out)
		}
		body, err := os.ReadFile(filepath.Join(dir, "ordinary.cgo2.c"))
		if err != nil {
			t.Fatal(err)
		}
		if baseline == nil {
			baseline = body
		} else if string(body) != string(baseline) {
			t.Fatal("cgo optimization experiments changed gccgo C wrappers")
		}
	}
}

func TestCgoOptRuntime(t *testing.T) {
	requireCgo(t)
	dir, err := filepath.Abs(filepath.Join("testdata", "cgoopt"))
	if err != nil {
		t.Fatal(err)
	}
	for _, experiment := range []string{"nologoscgoopt,nologosfastcgo", "logoscgoopt,nologosfastcgo"} {
		if goexperiment.CgoCheck2 {
			experiment += ",cgocheck2"
		}
		for _, driver := range []string{"go", "logos"} {
			t.Run(experiment+"/"+driver, func(t *testing.T) {
				t.Setenv("GOEXPERIMENT", experiment)
				tool := goBin(t)
				if driver == "logos" {
					tool = logosBin(t)
				}
				args := []string{"run"}
				if race.Enabled {
					args = append(args, "-race")
				}
				out, err := runCmd(t, dir, tool, append(args, ".")...)
				if err != nil {
					t.Fatalf("fixture: %v\n%s", err, out)
				}
				if strings.TrimSpace(out) != "callback stack moved; results, errno, retained pointer, and scheduler progress ok" {
					t.Fatalf("unexpected fixture output: %q", out)
				}
			})
		}
	}
}

var cgoOptBenchmarkSink int32

func BenchmarkCgoOptNocallback(b *testing.B) {
	b.ReportAllocs()
	var value int32
	for b.Loop() {
		// The bounded input avoids signed C overflow on arbitrarily long runs.
		value = callSlowNocallback(value&1023, 1)
	}
	cgoOptBenchmarkSink = value
}
