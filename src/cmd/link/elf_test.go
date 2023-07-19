// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd

package main

import (
	"bytes"
	"cmd/internal/buildid"
	"cmd/internal/notsha256"
	"cmd/link/internal/ld"
	"debug/elf"
	"fmt"
	"internal/platform"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"text/template"
)

func getCCAndCCFLAGS(t *testing.T, env []string) (string, []string) {
	goTool := testenv.GoToolPath(t)
	cmd := testenv.Command(t, goTool, "env", "CC")
	cmd.Env = env
	ccb, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	cc := strings.TrimSpace(string(ccb))

	cmd = testenv.Command(t, goTool, "env", "GOGCCFLAGS")
	cmd.Env = env
	cflagsb, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	cflags := strings.Fields(string(cflagsb))

	return cc, cflags
}

var asmSource = `
	.section .text1,"ax"
s1:
	.byte 0
	.section .text2,"ax"
s2:
	.byte 0
`

var goSource = `
package main
func main() {}
`

// The linker used to crash if an ELF input file had multiple text sections
// with the same name.
func TestSectionsWithSameName(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	t.Parallel()

	objcopy, err := exec.LookPath("objcopy")
	if err != nil {
		t.Skipf("can't find objcopy: %v", err)
	}

	dir := t.TempDir()

	gopath := filepath.Join(dir, "GOPATH")
	env := append(os.Environ(), "GOPATH="+gopath)

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module elf_test\n"), 0666); err != nil {
		t.Fatal(err)
	}

	asmFile := filepath.Join(dir, "x.s")
	if err := os.WriteFile(asmFile, []byte(asmSource), 0444); err != nil {
		t.Fatal(err)
	}

	goTool := testenv.GoToolPath(t)
	cc, cflags := getCCAndCCFLAGS(t, env)

	asmObj := filepath.Join(dir, "x.o")
	t.Logf("%s %v -c -o %s %s", cc, cflags, asmObj, asmFile)
	if out, err := testenv.Command(t, cc, append(cflags, "-c", "-o", asmObj, asmFile)...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	asm2Obj := filepath.Join(dir, "x2.syso")
	t.Logf("%s --rename-section .text2=.text1 %s %s", objcopy, asmObj, asm2Obj)
	if out, err := testenv.Command(t, objcopy, "--rename-section", ".text2=.text1", asmObj, asm2Obj).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	for _, s := range []string{asmFile, asmObj} {
		if err := os.Remove(s); err != nil {
			t.Fatal(err)
		}
	}

	goFile := filepath.Join(dir, "main.go")
	if err := os.WriteFile(goFile, []byte(goSource), 0444); err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, goTool, "build")
	cmd.Dir = dir
	cmd.Env = env
	t.Logf("%s build", goTool)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
}

var cSources35779 = []string{`
static int blah() { return 42; }
int Cfunc1() { return blah(); }
`, `
static int blah() { return 42; }
int Cfunc2() { return blah(); }
`,
}

// TestMinusRSymsWithSameName tests a corner case in the new
// loader. Prior to the fix this failed with the error 'loadelf:
// $WORK/b001/_pkg_.a(ldr.syso): duplicate symbol reference: blah in
// both main(.text) and main(.text)'. See issue #35779.
func TestMinusRSymsWithSameName(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	t.Parallel()

	dir := t.TempDir()

	gopath := filepath.Join(dir, "GOPATH")
	env := append(os.Environ(), "GOPATH="+gopath)

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module elf_test\n"), 0666); err != nil {
		t.Fatal(err)
	}

	goTool := testenv.GoToolPath(t)
	cc, cflags := getCCAndCCFLAGS(t, env)

	objs := []string{}
	csrcs := []string{}
	for i, content := range cSources35779 {
		csrcFile := filepath.Join(dir, fmt.Sprintf("x%d.c", i))
		csrcs = append(csrcs, csrcFile)
		if err := os.WriteFile(csrcFile, []byte(content), 0444); err != nil {
			t.Fatal(err)
		}

		obj := filepath.Join(dir, fmt.Sprintf("x%d.o", i))
		objs = append(objs, obj)
		t.Logf("%s %v -c -o %s %s", cc, cflags, obj, csrcFile)
		if out, err := testenv.Command(t, cc, append(cflags, "-c", "-o", obj, csrcFile)...).CombinedOutput(); err != nil {
			t.Logf("%s", out)
			t.Fatal(err)
		}
	}

	sysoObj := filepath.Join(dir, "ldr.syso")
	t.Logf("%s %v -nostdlib -r -o %s %v", cc, cflags, sysoObj, objs)
	if out, err := testenv.Command(t, cc, append(cflags, "-nostdlib", "-r", "-o", sysoObj, objs[0], objs[1])...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	cruft := [][]string{objs, csrcs}
	for _, sl := range cruft {
		for _, s := range sl {
			if err := os.Remove(s); err != nil {
				t.Fatal(err)
			}
		}
	}

	goFile := filepath.Join(dir, "main.go")
	if err := os.WriteFile(goFile, []byte(goSource), 0444); err != nil {
		t.Fatal(err)
	}

	t.Logf("%s build", goTool)
	cmd := testenv.Command(t, goTool, "build")
	cmd.Dir = dir
	cmd.Env = env
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
}

func TestGNUBuildIDDerivedFromGoBuildID(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	goFile := filepath.Join(t.TempDir(), "notes.go")
	if err := os.WriteFile(goFile, []byte(goSource), 0444); err != nil {
		t.Fatal(err)
	}
	outFile := filepath.Join(t.TempDir(), "notes.exe")
	goTool := testenv.GoToolPath(t)

	cmd := testenv.Command(t, goTool, "build", "-o", outFile, "-ldflags", "-buildid 0x1234 -B gobuildid", goFile)
	cmd.Dir = t.TempDir()

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	expectedGoBuildID := notsha256.Sum256([]byte("0x1234"))

	gnuBuildID, err := buildid.ReadELFNote(outFile, string(ld.ELF_NOTE_BUILDINFO_NAME), ld.ELF_NOTE_BUILDINFO_TAG)
	if err != nil || gnuBuildID == nil {
		t.Fatalf("can't read GNU build ID")
	}

	if !bytes.Equal(gnuBuildID, expectedGoBuildID[:20]) {
		t.Fatalf("build id not matching")
	}
}

func TestMergeNoteSections(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	expected := 1

	switch runtime.GOOS {
	case "linux", "dragonfly":
	case "openbsd", "netbsd", "freebsd":
		// These OSes require independent segment
		expected = 2
	default:
		t.Skip("We should only test on elf output.")
	}
	t.Parallel()

	goFile := filepath.Join(t.TempDir(), "notes.go")
	if err := os.WriteFile(goFile, []byte(goSource), 0444); err != nil {
		t.Fatal(err)
	}
	outFile := filepath.Join(t.TempDir(), "notes.exe")
	goTool := testenv.GoToolPath(t)
	// sha1sum of "gopher"
	id := "0xf4e8cd51ce8bae2996dc3b74639cdeaa1f7fee5f"
	cmd := testenv.Command(t, goTool, "build", "-o", outFile, "-ldflags",
		"-B "+id, goFile)
	cmd.Dir = t.TempDir()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	ef, err := elf.Open(outFile)
	if err != nil {
		t.Fatalf("open elf file failed:%v", err)
	}
	defer ef.Close()
	sec := ef.Section(".note.gnu.build-id")
	if sec == nil {
		t.Fatalf("can't find gnu build id")
	}

	sec = ef.Section(".note.go.buildid")
	if sec == nil {
		t.Fatalf("can't find go build id")
	}
	cnt := 0
	for _, ph := range ef.Progs {
		if ph.Type == elf.PT_NOTE {
			cnt += 1
		}
	}
	if cnt != expected {
		t.Fatalf("want %d PT_NOTE segment, got %d", expected, cnt)
	}
}

const pieSourceTemplate = `
package main

import "fmt"

// Force the creation of a lot of type descriptors that will go into
// the .data.rel.ro section.
{{range $index, $element := .}}var V{{$index}} interface{} = [{{$index}}]int{}
{{end}}

func main() {
{{range $index, $element := .}}	fmt.Println(V{{$index}})
{{end}}
}
`

func TestPIESize(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// We don't want to test -linkmode=external if cgo is not supported.
	// On some systems -buildmode=pie implies -linkmode=external, so just
	// always skip the test if cgo is not supported.
	testenv.MustHaveCGO(t)

	if !platform.BuildModeSupported(runtime.Compiler, "pie", runtime.GOOS, runtime.GOARCH) {
		t.Skip("-buildmode=pie not supported")
	}

	t.Parallel()

	tmpl := template.Must(template.New("pie").Parse(pieSourceTemplate))

	writeGo := func(t *testing.T, dir string) {
		f, err := os.Create(filepath.Join(dir, "pie.go"))
		if err != nil {
			t.Fatal(err)
		}

		// Passing a 100-element slice here will cause
		// pieSourceTemplate to create 100 variables with
		// different types.
		if err := tmpl.Execute(f, make([]byte, 100)); err != nil {
			t.Fatal(err)
		}

		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}

	for _, external := range []bool{false, true} {
		external := external

		name := "TestPieSize-"
		if external {
			name += "external"
		} else {
			name += "internal"
		}
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			dir := t.TempDir()

			writeGo(t, dir)

			binexe := filepath.Join(dir, "exe")
			binpie := filepath.Join(dir, "pie")
			if external {
				binexe += "external"
				binpie += "external"
			}

			build := func(bin, mode string) error {
				cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", bin, "-buildmode="+mode)
				if external {
					cmd.Args = append(cmd.Args, "-ldflags=-linkmode=external")
				}
				cmd.Args = append(cmd.Args, "pie.go")
				cmd.Dir = dir
				t.Logf("%v", cmd.Args)
				out, err := cmd.CombinedOutput()
				if len(out) > 0 {
					t.Logf("%s", out)
				}
				if err != nil {
					t.Log(err)
				}
				return err
			}

			var errexe, errpie error
			var wg sync.WaitGroup
			wg.Add(2)
			go func() {
				defer wg.Done()
				errexe = build(binexe, "exe")
			}()
			go func() {
				defer wg.Done()
				errpie = build(binpie, "pie")
			}()
			wg.Wait()
			if errexe != nil || errpie != nil {
				if runtime.GOOS == "android" && runtime.GOARCH == "arm64" {
					testenv.SkipFlaky(t, 58806)
				}
				t.Fatal("link failed")
			}

			var sizeexe, sizepie uint64
			if fi, err := os.Stat(binexe); err != nil {
				t.Fatal(err)
			} else {
				sizeexe = uint64(fi.Size())
			}
			if fi, err := os.Stat(binpie); err != nil {
				t.Fatal(err)
			} else {
				sizepie = uint64(fi.Size())
			}

			elfexe, err := elf.Open(binexe)
			if err != nil {
				t.Fatal(err)
			}
			defer elfexe.Close()

			elfpie, err := elf.Open(binpie)
			if err != nil {
				t.Fatal(err)
			}
			defer elfpie.Close()

			// The difference in size between exe and PIE
			// should be approximately the difference in
			// size of the .text section plus the size of
			// the PIE dynamic data sections plus the
			// difference in size of the .got and .plt
			// sections if they exist.
			// We ignore unallocated sections.
			// There may be gaps between non-writeable and
			// writable PT_LOAD segments. We also skip those
			// gaps (see issue #36023).

			textsize := func(ef *elf.File, name string) uint64 {
				for _, s := range ef.Sections {
					if s.Name == ".text" {
						return s.Size
					}
				}
				t.Fatalf("%s: no .text section", name)
				return 0
			}
			textexe := textsize(elfexe, binexe)
			textpie := textsize(elfpie, binpie)

			dynsize := func(ef *elf.File) uint64 {
				var ret uint64
				for _, s := range ef.Sections {
					if s.Flags&elf.SHF_ALLOC == 0 {
						continue
					}
					switch s.Type {
					case elf.SHT_DYNSYM, elf.SHT_STRTAB, elf.SHT_REL, elf.SHT_RELA, elf.SHT_HASH, elf.SHT_GNU_HASH, elf.SHT_GNU_VERDEF, elf.SHT_GNU_VERNEED, elf.SHT_GNU_VERSYM:
						ret += s.Size
					}
					if s.Flags&elf.SHF_WRITE != 0 && (strings.Contains(s.Name, ".got") || strings.Contains(s.Name, ".plt")) {
						ret += s.Size
					}
				}
				return ret
			}

			dynexe := dynsize(elfexe)
			dynpie := dynsize(elfpie)

			extrasize := func(ef *elf.File) uint64 {
				var ret uint64
				// skip unallocated sections
				for _, s := range ef.Sections {
					if s.Flags&elf.SHF_ALLOC == 0 {
						ret += s.Size
					}
				}
				// also skip gaps between PT_LOAD segments
				var prev *elf.Prog
				for _, seg := range ef.Progs {
					if seg.Type != elf.PT_LOAD {
						continue
					}
					if prev != nil {
						ret += seg.Off - prev.Off - prev.Filesz
					}
					prev = seg
				}
				return ret
			}

			extraexe := extrasize(elfexe)
			extrapie := extrasize(elfpie)

			if sizepie < sizeexe || sizepie-extrapie < sizeexe-extraexe {
				return
			}
			diffReal := (sizepie - extrapie) - (sizeexe - extraexe)
			diffExpected := (textpie + dynpie) - (textexe + dynexe)

			t.Logf("real size difference %#x, expected %#x", diffReal, diffExpected)

			if diffReal > (diffExpected + diffExpected/10) {
				t.Errorf("PIE unexpectedly large: got difference of %d (%d - %d), expected difference %d", diffReal, sizepie, sizeexe, diffExpected)
			}
		})
	}
}

func TestIssue51939(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()
	td := t.TempDir()
	goFile := filepath.Join(td, "issue51939.go")
	if err := os.WriteFile(goFile, []byte(goSource), 0444); err != nil {
		t.Fatal(err)
	}
	outFile := filepath.Join(td, "issue51939.exe")
	goTool := testenv.GoToolPath(t)
	cmd := testenv.Command(t, goTool, "build", "-o", outFile, goFile)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	ef, err := elf.Open(outFile)
	if err != nil {
		t.Fatal(err)
	}

	for _, s := range ef.Sections {
		if s.Flags&elf.SHF_ALLOC == 0 && s.Addr != 0 {
			t.Errorf("section %s should not allocated with addr %x", s.Name, s.Addr)
		}
	}
}

func TestFlagR(t *testing.T) {
	// Test that using the -R flag to specify a (large) alignment generates
	// a working binary.
	// (Test only on ELF for now. The alignment allowed differs from platform
	// to platform.)
	testenv.MustHaveGoBuild(t)
	t.Parallel()
	tmpdir := t.TempDir()
	src := filepath.Join(tmpdir, "x.go")
	if err := os.WriteFile(src, []byte(goSource), 0444); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(tmpdir, "x.exe")

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-R=0x100000", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v, output:\n%s", err, out)
	}

	cmd = testenv.Command(t, exe)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("executable failed to run: %v\n%s", err, out)
	}
}
