// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestBinaryBulkWidth(t *testing.T) {
	pkg := types2.NewPackage("p", "p")
	named := func(name string, underlying types2.Type) types2.Type {
		return types2.NewNamed(types2.NewTypeName(syntax.Pos{}, pkg, name, nil), underlying, nil)
	}
	basic := func(kind types2.BasicKind) types2.Type { return types2.Typ[kind] }
	cases := []struct {
		name string
		typ  types2.Type
		want int
	}{
		{"int64", basic(types2.Int64), 8},
		{"uint64", basic(types2.Uint64), 8},
		{"float64", basic(types2.Float64), 8},
		{"float32", basic(types2.Float32), 4},
		{"named int64", named("Dur", basic(types2.Int64)), 8},
		{"named uint64", named("Tick", basic(types2.Uint64)), 8},
		{"named float64", named("Ratio", basic(types2.Float64)), 8},
		{"named float32", named("Gain", basic(types2.Float32)), 4},
		{"int", basic(types2.Int), 0},
		{"uint", basic(types2.Uint), 0},
		{"int8", basic(types2.Int8), 0},
		{"int16", basic(types2.Int16), 0},
		{"int32", basic(types2.Int32), 0},
		{"uint8", basic(types2.Uint8), 0},
		{"uint16", basic(types2.Uint16), 0},
		{"uint32", basic(types2.Uint32), 0},
		{"uintptr", basic(types2.Uintptr), 0},
		{"bool", basic(types2.Bool), 0},
		{"string", basic(types2.String), 0},
		{"complex128", basic(types2.Complex128), 0},
		{"named int32", named("Code", basic(types2.Int32)), 0},
		{"named bool", named("Flag", basic(types2.Bool)), 0},
		{"slice of float64", types2.NewSlice(basic(types2.Float64)), 0},
		{"array of int64", types2.NewArray(basic(types2.Int64), 4), 0},
		{"pointer to int64", types2.NewPointer(basic(types2.Int64)), 0},
		{"empty struct", types2.NewStruct(nil, nil), 0},
	}
	for _, c := range cases {
		if got := binaryBulkWidth(c.typ); got != c.want {
			t.Errorf("binaryBulkWidth(%s) = %d, want %d", c.name, got, c.want)
		}
	}
}

func TestBinaryBulkEjectedWire(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := filepath.Join(testenv.GOROOT(t), "test", "logos", "run")
	const name = "binary_bulk_test.lgo"
	src, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatal(err)
	}
	base := syntax.NewFileBase(filepath.Join(dir, name))
	file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	edits := ejectLower(t, pkg, info, file, src)
	ejected, err := ApplyEdits(src, base, edits)
	if err != nil {
		t.Fatalf("ApplyEdits %s: %v", name, err)
	}
	work := t.TempDir()
	if err := os.WriteFile(filepath.Join(work, "go.mod"), []byte("module ejectbulk\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(work, "main.go"), ejected, 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := testenv.Command(t, testenv.GoToolPath(t), "run", ".")
	cmd.Dir = work
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOTOOLCHAIN=local")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("ejected run: %v\n%s", err, out)
	}
	want, err := os.ReadFile(filepath.Join(dir, "binary_bulk_test.out"))
	if err != nil {
		t.Fatal(err)
	}
	if got := string(out); got != string(want) {
		t.Fatalf("ejected verdicts differ:\ngot:\n%swant:\n%s", got, want)
	}
}

const binaryBulkBenchModel = `package wirebench

type Wire struct {
	F []float64 ` + "`json:\"f\"`" + `
	I []int64   ` + "`json:\"i\"`" + `
}
`

const binaryBulkBenchTest = `package wirebench

import "testing"

const benchCount = 4096

func benchWire() Wire {
	v := Wire{F: make([]float64, benchCount), I: make([]int64, benchCount)}
	for i := range v.F {
		v.F[i] = float64(i)*0.37 - 11.5
		v.I[i] = int64(uint64(i+1) * 0x9E3779B97F4A7C15)
	}
	return v
}

func BenchmarkAppendBinary(b *testing.B) {
	v := benchWire()
	wire, err := v.MarshalBinary()
	if err != nil {
		b.Fatal(err)
	}
	dst := make([]byte, 0, len(wire))
	b.SetBytes(int64(len(wire)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := v.AppendBinary(dst[:0])
		if err != nil {
			b.Fatal(err)
		}
		dst = out
	}
}

func BenchmarkUnmarshalBinary(b *testing.B) {
	v := benchWire()
	wire, err := v.MarshalBinary()
	if err != nil {
		b.Fatal(err)
	}
	var back Wire
	b.SetBytes(int64(len(wire)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := back.UnmarshalBinary(wire); err != nil {
			b.Fatal(err)
		}
	}
}
`

func BenchmarkBinaryBulkWire(b *testing.B) {
	testenv.MustHaveGoBuild(b)
	logosTool := filepath.Join(filepath.Dir(testenv.GoToolPath(b)), "logos")
	if runtime.GOOS == "windows" {
		logosTool += ".exe"
	}
	if _, err := os.Stat(logosTool); err != nil {
		b.Skipf("no logos command beside the go tool: %v", err)
	}
	dir := b.TempDir()
	files := map[string]string{
		"go.mod":        "module wirebench\n\ngo 1.27\n",
		"wire.lgo":      binaryBulkBenchModel,
		"wire_test.lgo": binaryBulkBenchTest,
	}
	for name, text := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text), 0o666); err != nil {
			b.Fatal(err)
		}
	}
	cmd := testenv.Command(b, logosTool, "test", "-run=^$", "-bench=.", "-benchmem", "-count=1")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOFLAGS=", "GOPROXY=off", "GOTOOLCHAIN=local")
	out, err := cmd.CombinedOutput()
	if err != nil {
		b.Fatalf("logos test: %v\n%s", err, out)
	}
	b.Logf("%s", out)
	b.ReportMetric(0, "ns/op")
	for line := range strings.SplitSeq(string(out), "\n") {
		reportBinaryBulkLine(b, strings.Fields(line))
	}
}

func reportBinaryBulkLine(b *testing.B, fields []string) {
	if len(fields) < 4 {
		return
	}
	name, ok := strings.CutPrefix(fields[0], "Benchmark")
	if !ok {
		return
	}
	if before, _, found := strings.CutLast(name, "-"); found {
		name = before
	}
	for i := 2; i+1 < len(fields); i += 2 {
		value, err := strconv.ParseFloat(fields[i], 64)
		if err != nil {
			continue
		}
		b.ReportMetric(value, name+"-"+fields[i+1])
	}
}
