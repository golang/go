// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import (
	"internal/goexperiment"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"unsafe"
)

func TestCgoOptSpanCodegen(t *testing.T) {
	requireCgo(t)
	if runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64" {
		t.Skip("object-code assertions cover amd64 and arm64")
	}
	testenv.MustHaveGoBuild(t)
	binary := filepath.Join(t.TempDir(), "span.test")
	args := append([]string{"test", "-c", "-o", binary}, cgoSpanToolFlags(t)...)
	args = append(args, "cmd/cgo/internal/testlogos")
	out, err := runCmd(t, filepath.Join(runtime.GOROOT(), "src", "cmd"), goBin(t), args...)
	if err != nil {
		t.Fatalf("build symbolized span test binary: %v\n%s", err, out)
	}
	body := objdumpFunc(t, binary, "^runtime\\.cgoCheckUnknownPointer$")
	if !strings.Contains(body, "TEXT runtime.cgoCheckUnknownPointer") {
		t.Fatalf("missing cgoCheckUnknownPointer disassembly:\n%s", body)
	}
	want := 1
	if goexperiment.LogosCgoOpt {
		want = 0
	}
	if got := strings.Count(body, "runtime.findObject(SB)"); got != want {
		t.Errorf("findObject calls = %d, want %d with logoscgoopt=%v:\n%s", got, want, goexperiment.LogosCgoOpt, body)
	}
	if strings.Contains(body, "runtime.logosCgoPointerObject") {
		t.Errorf("pointer-object hook must inline in both experiment settings:\n%s", body)
	}
	wrapper := objdumpFunc(t, binary, "testlogos\\._Cfunc_cgo_span_consume")
	if strings.Count(wrapper, "CALL runtime.cgocall(SB)") != 1 || strings.Contains(wrapper, "logosCgocallFast") {
		t.Fatalf("void-pointer workload must retain one ordinary cgocall:\n%s", wrapper)
	}
}

func TestCgoOptSpanPointerRules(t *testing.T) {
	requireCgo(t)
	testenv.MustHaveGoBuild(t)
	dir, err := filepath.Abs(filepath.Join("testdata", "cgospan"))
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",cgocheck=1")
	args := append([]string{"run"}, cgoSpanToolFlags(t)...)
	out, err := runCmd(t, dir, goBin(t), append(args, ".")...)
	if err != nil {
		t.Fatalf("span pointer fixture: %v\n%s", err, out)
	}
	if strings.TrimSpace(out) != "span pointer and exported-result checks ok" {
		t.Fatalf("unexpected span fixture output: %q", out)
	}
}

// cgoSpanToolFlags keeps child builds on the experiment and instrumentation
// of this test binary, with fast calls off so the cgoopt span path runs.
func cgoSpanToolFlags(t *testing.T) []string {
	t.Helper()
	return cgoTestToolFlags(t, false)
}

// cgoTestToolFlags replays this binary's experiment and instrumentation
// settings with the caller's fast-call policy.
func cgoTestToolFlags(t *testing.T, fastcgo bool) []string {
	t.Helper()
	info, ok := debug.ReadBuildInfo()
	if !ok {
		t.Fatal("child build requires embedded build settings")
	}
	var experiments []string
	flags := []string{"-race=false", "-asan=false", "-msan=false"}
	for _, setting := range info.Settings {
		switch setting.Key {
		case "GOEXPERIMENT":
			if setting.Value != "" {
				experiments = append(experiments, setting.Value)
			}
		case "-race", "-asan", "-msan":
			flags = append(flags, setting.Key+"="+setting.Value)
		}
	}
	if fastcgo {
		experiments = append(experiments, "logosfastcgo")
	} else {
		experiments = append(experiments, "nologosfastcgo")
	}
	t.Setenv("GOENV", "off")
	t.Setenv("GOEXPERIMENT", strings.Join(experiments, ","))
	return flags
}

// Both benchmarks call an unmarked void C function with one pointer argument.
// The reference has a pointer-free C parameter type and omits pointer checks;
// it is not a same-contract lower bound. Candidate and baseline comparisons
// always use BenchmarkCgoOptSpanLookup with identical pointer-check obligations.
func BenchmarkCgoOptSpanLookup(b *testing.B) {
	value := new([64]byte)
	p := unsafe.Pointer(value)
	b.ReportAllocs()
	for b.Loop() {
		callCgoSpan(p)
	}
	runtime.KeepAlive(value)
}

func BenchmarkCgoOptSpanReference(b *testing.B) {
	value := new([64]byte)
	b.ReportAllocs()
	for b.Loop() {
		callCgoSpanBytes(&value[0])
	}
	runtime.KeepAlive(value)
}
