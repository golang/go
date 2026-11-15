// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/internal/goobj"
	"internal/testenv"
)

const archDiffPkg = `package main

func Plain() int { return 7 }

func Yielder() {
	c := make(chan int)
	<-c
}

func Pick[T any](x any) T {
	return x.(T)
}

func main() {
	Yielder()
	if Pick[int](any(5)) != 5 {
		panic("bad")
	}
	println(Plain())
}
`

func archDiffEnv(cmd *exec.Cmd, goos, goarch string) []string {
	var env []string
	for _, kv := range cmd.Environ() {
		switch {
		case strings.HasPrefix(kv, "GOOS="), strings.HasPrefix(kv, "GOARCH="),
			strings.HasPrefix(kv, "GOFLAGS="), strings.HasPrefix(kv, "GOWORK="),
			strings.HasPrefix(kv, "GO111MODULE="):
		default:
			env = append(env, kv)
		}
	}
	return append(env, "GOOS="+goos, "GOARCH="+goarch, "GOFLAGS=", "GOWORK=off", "GO111MODULE=on")
}

func archDiffExport(t *testing.T, tool, dir, goos, goarch string) string {
	t.Helper()
	cmd := testenv.Command(t, tool, "list", "-export", "-f", "{{.Export}}", ".")
	cmd.Dir = dir
	cmd.Env = archDiffEnv(cmd, goos, goarch)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("%s list -export for %s/%s: %v", tool, goos, goarch, err)
	}
	p := strings.TrimSpace(string(out))
	if p == "" {
		t.Fatalf("%s: no export archive for %s/%s", tool, goos, goarch)
	}
	return p
}

func archDiffLines(f *facts.Facts) []string {
	all := strings.Split(string(facts.Encode(goobj.FingerprintType{}, f)), "\n")
	return all[2:]
}

func stripMayYield(line string) string {
	line = strings.ReplaceAll(line, ",MayYield=0", "")
	line = strings.ReplaceAll(line, ",MayYield=1", "")
	return strings.TrimSuffix(strings.TrimSuffix(line, " MayYield=1"), " MayYield=0")
}

func TestGoPackageFactLinesMatchAcrossGOARCH(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tool := wasmLogos(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module diffpkg\n\ngo 1.27\n"), 0o666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(archDiffPkg), 0o666); err != nil {
		t.Fatal(err)
	}
	amdFacts := readFacts(t, archDiffExport(t, tool, dir, runtime.GOOS, runtime.GOARCH))
	wasmFacts := readFacts(t, archDiffExport(t, tool, dir, "js", "wasm"))
	if amdFacts == nil || wasmFacts == nil {
		t.Fatalf("missing __.LOGOS member: amd64 %v, wasm %v", amdFacts != nil, wasmFacts != nil)
	}
	amd, wasm := archDiffLines(amdFacts), archDiffLines(wasmFacts)
	if !slices.Contains(amd, "lang go") || !slices.Contains(wasm, "lang go") {
		t.Errorf("lang go record missing: amd64 %v, wasm %v", slices.Contains(amd, "lang go"), slices.Contains(wasm, "lang go"))
	}
	for _, l := range amd {
		if strings.Contains(l, "MayYield") {
			t.Errorf("amd64 fact %q carries the GOARCH=wasm-only MayYield condition", l)
			continue
		}
		if !slices.ContainsFunc(wasm, func(w string) bool { return stripMayYield(w) == l }) {
			t.Errorf("amd64 fact %q is missing from the wasm build", l)
		}
	}
	for _, l := range wasm {
		if slices.Contains(amd, l) {
			continue
		}
		if !strings.Contains(l, "MayYield") {
			t.Errorf("wasm fact %q differs from the amd64 build without the GOARCH=wasm emission condition", l)
		}
	}
	if !slices.ContainsFunc(wasm, func(l string) bool { return strings.Contains(l, "MayYield=1") }) {
		t.Error("the wasm build recorded no yielding function")
	}
}
