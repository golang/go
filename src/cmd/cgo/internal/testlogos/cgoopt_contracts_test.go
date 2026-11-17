// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const cgoOptContractsCalls = `package contracts
/*
int opt_shared(int);
#cgo noescape opt_split
int opt_split(int *);
int opt_mapped(int *);
#cgo noescape opt_noescape
int opt_noescape(int *);
int opt_unmarked(int *);
*/
import "C"

func Shared(v C.int) C.int { return C.opt_shared(v) }
func Split(p *C.int) C.int { return C.opt_split(p) }
func Mapped(p *C.int) C.int { return C.opt_mapped(p) }
func Noescape(p *C.int) C.int { return C.opt_noescape(p) }
func Unmarked(p *C.int) C.int { return C.opt_unmarked(p) }
`

const cgoOptContractsMarks = `package contracts
/*
#cgo nocallback opt_shared
int opt_shared(int v) { return v + 1; }
#cgo nocallback opt_split
int opt_split(int *p) { return *p; }
CONTRACT_MARKER
int opt_mapped(int *p) { return *p; }
int opt_noescape(int *p) { return *p; }
int opt_unmarked(int *p) { return *p; }
*/
import "C"
`

// cgoOptContractRun reuses one input and output directory so comparisons do not
// need to normalize generated paths or diagnostics.
func cgoOptContractRun(t *testing.T, dir string, inputs []string, experiment string, flags ...string) (map[string]string, string, error) {
	t.Helper()
	requireCgo(t)
	t.Setenv("GOEXPERIMENT", experiment)
	out := filepath.Join(dir, "out")
	if err := os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	args := []string{"-objdir", out, "-importpath", "scratch/contracts"}
	args = append(args, flags...)
	args = append(args, inputs...)
	text, err := runCmd(t, dir, cgoBin(t), args...)
	if err != nil {
		return nil, text, err
	}
	entries, err := os.ReadDir(out)
	if err != nil {
		t.Fatal(err)
	}
	files := make(map[string]string)
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !(strings.HasSuffix(name, ".go") || strings.HasSuffix(name, ".c") || strings.HasSuffix(name, ".h")) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(out, name))
		if err != nil {
			t.Fatal(err)
		}
		files[name] = string(data)
	}
	return files, text, nil
}

func TestCgoOptPackageContracts(t *testing.T) {
	for _, ext := range []string{".go", ".lgo"} {
		for _, early := range []bool{false, true} {
			for _, fast := range []bool{false, true} {
				order, fastExperiment := "late", "nologosfastcgo"
				if early {
					order = "early"
				}
				if fast {
					fastExperiment = "logosfastcgo"
				}
				t.Run(ext+"/"+order+"/"+fastExperiment, func(t *testing.T) {
					dir := t.TempDir()
					marker := "#cgo noescape opt_mapped\n#cgo nocallback opt_mapped"
					if ext == ".lgo" {
						marker = "//logos:cgo_fastcall opt_mapped"
					}
					sources := map[string]string{
						"calls" + ext: cgoOptContractsCalls,
						"marks" + ext: strings.ReplaceAll(cgoOptContractsMarks, "CONTRACT_MARKER", marker),
					}
					for name, source := range sources {
						if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0644); err != nil {
							t.Fatal(err)
						}
					}
					inputs := []string{"calls" + ext, "marks" + ext}
					if early {
						inputs[0], inputs[1] = inputs[1], inputs[0]
					}
					var baseline map[string]string
					for _, enabled := range []bool{false, true} {
						experiment := "nologoscgoopt," + fastExperiment
						if enabled {
							experiment = "logoscgoopt," + fastExperiment
						}
						files, output, err := cgoOptContractRun(t, dir, inputs, experiment)
						if err != nil {
							t.Fatalf("%s: %v\n%s", experiment, err, output)
						}
						if !enabled {
							baseline = files
						} else {
							if len(files) != len(baseline) {
								t.Fatalf("generated file count changed: %d versus %d", len(files), len(baseline))
							}
							for name, before := range baseline {
								after, ok := files[name]
								if !ok {
									t.Errorf("generated file %s disappeared", name)
								} else if strings.HasSuffix(name, ".go") && after != before {
									t.Errorf("package contract collection changed generated Go file %s", name)
								}
							}
						}
						cSource, goSource := shapesCgo2(t, files), shapesGotypes(t, files)
						for _, tc := range []struct {
							name       string
							nocallback bool
							noescape   bool
						}{
							{"opt_shared", true, false},
							{"opt_split", true, true},
							{"opt_mapped", true, true},
							{"opt_noescape", false, true},
							{"opt_unmarked", false, false},
						} {
							wrapper := "_Cfunc_" + tc.name
							lookups := 2
							if tc.nocallback && (enabled || early && fast && tc.noescape) {
								lookups = 0
							}
							cBody := cgoOptWrapper(t, cSource, wrapper)
							if got := countCallLines(cBody, "_cgo_topofstack"); got != lookups {
								t.Errorf("%s %s: stack-top lookups = %d, want %d", experiment, wrapper, got, lookups)
							}
							goBody := cgoOptWrapper(t, goSource, wrapper)
							fastCalls, ordinaryCalls, guards := 0, 1, 0
							if fast && tc.nocallback && tc.noescape {
								fastCalls, ordinaryCalls = 1, 0
							} else if tc.nocallback {
								guards = 2
							}
							uses, keepalives := 1, 0
							if tc.nocallback && tc.noescape {
								uses, keepalives = 0, 1
							}
							for _, call := range []struct {
								name string
								want int
							}{
								{"_cgo_runtime_cgocall", ordinaryCalls},
								{"_cgo_runtime_cgocall_fast", fastCalls},
								{"_Cgo_no_callback", guards},
								{"_Cgo_use", uses},
								{"_Cgo_keepalive", keepalives},
							} {
								if got := countCallLines(goBody, call.name); got != call.want {
									t.Errorf("%s %s: %s calls = %d, want %d", experiment, wrapper, call.name, got, call.want)
								}
							}
						}
					}
				})
			}
		}
	}
}

func TestCgoOptPackageContractErrors(t *testing.T) {
	const first = `package contracts
/*
int opt_shared(int);
*/
import "C"
func Call(v C.int) C.int { return C.opt_shared(v) }
`
	for _, tc := range []struct {
		name   string
		second string
		want   string
	}{
		{"unmatched_nocallback", "package contracts\n/*\n#cgo nocallback absent\n*/\nimport \"C\"\n", "#cgo nocallback absent: no matched C function"},
		{"unmatched_noescape", "package contracts\n/*\n#cgo noescape absent\n*/\nimport \"C\"\n", "#cgo noescape absent: no matched C function"},
		{"package", "package other\n/*\n#cgo nocallback opt_shared\nint opt_shared(int);\n*/\nimport \"C\"\n", "inconsistent package names"},
		{"package_before_directive", "package other\n/*\n#cgo nocallback absent\n*/\nimport \"C\"\n", "inconsistent package names"},
		{"signature", "package contracts\n/*\n#cgo nocallback opt_shared\nlong long opt_shared(double);\n*/\nimport \"C\"\nfunc Other(v C.double) C.longlong { return C.opt_shared(v) }\n", "inconsistent definitions for C.opt_shared"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			for name, source := range map[string]string{"a.go": first, "b.go": tc.second} {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0644); err != nil {
					t.Fatal(err)
				}
			}
			var baseline string
			for _, experiment := range []string{"nologoscgoopt,nologosfastcgo", "logoscgoopt,nologosfastcgo"} {
				_, output, err := cgoOptContractRun(t, dir, []string{"a.go", "b.go"}, experiment)
				if err == nil || !strings.Contains(output, tc.want) {
					t.Fatalf("%s: error = %v, output = %q; want %q", experiment, err, output, tc.want)
				}
				if baseline == "" {
					baseline = output
				} else if output != baseline {
					t.Fatalf("diagnostics changed:\nbaseline:\n%scandidate:\n%s", baseline, output)
				}
			}
		})
	}
}

func TestCgoOptPackageContractModes(t *testing.T) {
	for _, mode := range []string{"gccgo", "godefs"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			inputs := []string{"a.go", "b.go"}
			sources := map[string]string{
				"a.go": cgoOptContractsCalls,
				"b.go": strings.ReplaceAll(cgoOptContractsMarks, "CONTRACT_MARKER", "#cgo noescape opt_mapped\n#cgo nocallback opt_mapped"),
			}
			if mode == "gccgo" {
				compiler := os.Getenv("GCCGO")
				if compiler == "" {
					compiler = "gccgo"
				}
				testenv.MustHaveExecPath(t, compiler)
			} else {
				inputs = []string{"defs.go"}
				sources = map[string]string{
					"defs.go": "package contracts\n/*\ntypedef struct { int x; unsigned long long y; } opt_value;\n*/\nimport \"C\"\ntype Value C.opt_value\n",
				}
			}
			for name, source := range sources {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0644); err != nil {
					t.Fatal(err)
				}
			}
			var baseline map[string]string
			var baselineOutput string
			for _, experiment := range []string{"nologoscgoopt,nologosfastcgo", "logoscgoopt,logosfastcgo"} {
				files, output, err := cgoOptContractRun(t, dir, inputs, experiment, "-"+mode)
				if err != nil {
					t.Fatalf("%s: %v\n%s", experiment, err, output)
				}
				if baseline == nil {
					baseline, baselineOutput = files, output
					continue
				}
				if output != baselineOutput || len(files) != len(baseline) {
					t.Fatalf("%s output changed with cgo optimization experiments", mode)
				}
				for name, before := range baseline {
					if after, ok := files[name]; !ok || after != before {
						t.Errorf("%s generated file %s changed", mode, name)
					}
				}
			}
		})
	}
}

func TestCgoOptPackageContractCall(t *testing.T) {
	for _, tc := range []struct{ a, b uint32 }{
		{0, 0},
		{40, 2},
		{^uint32(0), 1},
		{^uint32(0), ^uint32(0)},
	} {
		if got, want := callCgoOptLateAdd(tc.a, tc.b), tc.a+tc.b; got != want {
			t.Errorf("add(%d, %d) = %d, want %d", tc.a, tc.b, got, want)
		}
	}
}

func TestCgoOptPackagePreambles(t *testing.T) {
	if got := callCgoOptLocalFirst(); got != 11 {
		t.Errorf("first file-local macro = %d, want 11", got)
	}
	if got := callCgoOptLocalSecond(); got != 23 {
		t.Errorf("second file-local macro = %d, want 23", got)
	}
}

var cgoOptContractsSink uint32

func BenchmarkCgoOptLateContract(b *testing.B) {
	b.ReportAllocs()
	var value uint32
	for b.Loop() {
		value = callCgoOptLateAdd(value&1023, 1)
	}
	cgoOptContractsSink = value
}
