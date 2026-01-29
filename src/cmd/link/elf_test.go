// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd

package main

import (
	"cmd/internal/buildid"
	"cmd/internal/hash"
	"cmd/link/internal/ld"
	"debug/elf"
	"encoding/binary"
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
	"unsafe"
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

var goSourceWithData = `
package main
var globalVar = 42
func main() { println(&globalVar) }
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
	gopathEnv := "GOPATH=" + gopath
	env := append(os.Environ(), gopathEnv)

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module elf_test\n"), 0666); err != nil {
		t.Fatal(err)
	}

	asmFile := filepath.Join(dir, "x.s")
	if err := os.WriteFile(asmFile, []byte(asmSource), 0444); err != nil {
		t.Fatal(err)
	}

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

	cmd := goCmd(t, "build")
	cmd.Dir = dir
	cmd.Env = append(cmd.Env, gopathEnv)
	t.Logf("%s build", testenv.GoToolPath(t))
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
	gopathEnv := "GOPATH=" + gopath
	env := append(os.Environ(), gopathEnv)

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module elf_test\n"), 0666); err != nil {
		t.Fatal(err)
	}

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

	t.Logf("%s build", testenv.GoToolPath(t))
	cmd := goCmd(t, "build")
	cmd.Dir = dir
	cmd.Env = append(cmd.Env, gopathEnv)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
}

func TestGNUBuildID(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()
	goFile := filepath.Join(tmpdir, "notes.go")
	if err := os.WriteFile(goFile, []byte(goSource), 0444); err != nil {
		t.Fatal(err)
	}

	// Use a specific Go buildid for testing.
	const gobuildid = "testbuildid"
	h := hash.Sum32([]byte(gobuildid))
	gobuildidHash := string(h[:20])

	tests := []struct{ name, ldflags, expect string }{
		{"default", "", gobuildidHash},
		{"gobuildid", "-B=gobuildid", gobuildidHash},
		{"specific", "-B=0x0123456789abcdef", "\x01\x23\x45\x67\x89\xab\xcd\xef"},
		{"none", "-B=none", ""},
	}
	if testenv.HasCGO() && runtime.GOOS != "solaris" && runtime.GOOS != "illumos" {
		// Solaris ld doesn't support --build-id. So we don't
		// add it in external linking mode.
		for _, test := range tests {
			t1 := test
			t1.name += "_external"
			t1.ldflags += " -linkmode=external"
			tests = append(tests, t1)
		}
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exe := filepath.Join(tmpdir, test.name)
			cmd := goCmd(t, "build", "-ldflags=-buildid="+gobuildid+" "+test.ldflags, "-o", exe, goFile)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("%v: %v:\n%s", cmd.Args, err, out)
			}
			gnuBuildID, err := buildid.ReadELFNote(exe, string(ld.ELF_NOTE_BUILDINFO_NAME), ld.ELF_NOTE_BUILDINFO_TAG)
			if err != nil {
				t.Fatalf("can't read GNU build ID")
			}
			if string(gnuBuildID) != test.expect {
				t.Errorf("build id mismatch: got %x, want %x", gnuBuildID, test.expect)
			}
		})
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
	// sha1sum of "gopher"
	id := "0xf4e8cd51ce8bae2996dc3b74639cdeaa1f7fee5f"
	cmd := goCmd(t, "build", "-o", outFile, "-ldflags", "-B "+id, goFile)
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

	var linkmodes []string
	if platform.InternalLinkPIESupported(runtime.GOOS, runtime.GOARCH) {
		linkmodes = append(linkmodes, "internal")
	}
	linkmodes = append(linkmodes, "external")

	for _, linkmode := range linkmodes {
		t.Run(fmt.Sprintf("TestPieSize-%v", linkmode), func(t *testing.T) {
			t.Parallel()

			dir := t.TempDir()

			writeGo(t, dir)

			binexe := filepath.Join(dir, "exe")
			binpie := filepath.Join(dir, "pie")
			binexe += linkmode
			binpie += linkmode

			build := func(bin, mode string) error {
				cmd := goCmd(t, "build", "-o", bin, "-buildmode="+mode, "-ldflags=-linkmode="+linkmode)
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
	cmd := goCmd(t, "build", "-o", outFile, goFile)
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

	cmd := goCmd(t, "build", "-ldflags=-R=0x100000", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v, output:\n%s", err, out)
	}

	cmd = testenv.Command(t, exe)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("executable failed to run: %v\n%s", err, out)
	}
}

func TestFlagD(t *testing.T) {
	// Test that using the -D flag to specify data section address generates
	// a working binary with data at the specified address.
	t.Parallel()
	testFlagD(t, "0x10000000", "", 0x10000000)
}

func TestFlagDUnaligned(t *testing.T) {
	// Test that using the -D flag with an unaligned address errors out
	t.Parallel()
	testFlagDError(t, "0x10000123", "", "invalid -D value 0x10000123")
}

func TestFlagDWithR(t *testing.T) {
	// Test that using the -D flag with -R flag errors on unaligned address.
	t.Parallel()
	testFlagDError(t, "0x30001234", "8192", "invalid -D value 0x30001234")
}

func testFlagD(t *testing.T, dataAddr string, roundQuantum string, expectedAddr uint64) {
	testenv.MustHaveGoBuild(t)
	tmpdir := t.TempDir()
	src := filepath.Join(tmpdir, "x.go")
	if err := os.WriteFile(src, []byte(goSourceWithData), 0444); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(tmpdir, "x.exe")

	// Build linker flags
	ldflags := "-D=" + dataAddr
	if roundQuantum != "" {
		ldflags += " -R=" + roundQuantum
	}

	cmd := goCmd(t, "build", "-ldflags="+ldflags, "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v, output:\n%s", err, out)
	}

	cmd = testenv.Command(t, exe)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("executable failed to run: %v\n%s", err, out)
	}

	ef, err := elf.Open(exe)
	if err != nil {
		t.Fatalf("open elf file failed: %v", err)
	}
	defer ef.Close()

	// Find the first data-related section to verify segment placement
	var firstDataSection *elf.Section
	for _, sec := range ef.Sections {
		if sec.Type == elf.SHT_PROGBITS || sec.Type == elf.SHT_NOBITS {
			// These sections are writable, allocated at runtime, but not executable
			// nor TLS.
			isWrite := sec.Flags&elf.SHF_WRITE != 0
			isExec := sec.Flags&elf.SHF_EXECINSTR != 0
			isAlloc := sec.Flags&elf.SHF_ALLOC != 0
			isTLS := sec.Flags&elf.SHF_TLS != 0

			if isWrite && !isExec && isAlloc && !isTLS {
				if firstDataSection == nil || sec.Addr < firstDataSection.Addr {
					firstDataSection = sec
				}
			}
		}
	}

	if firstDataSection == nil {
		t.Fatalf("can't find any writable data sections")
	}
	if firstDataSection.Addr != expectedAddr {
		t.Errorf("data section starts at 0x%x for section %s, expected 0x%x",
			firstDataSection.Addr, firstDataSection.Name, expectedAddr)
	}
}

func testFlagDError(t *testing.T, dataAddr string, roundQuantum string, expectedError string) {
	testenv.MustHaveGoBuild(t)
	tmpdir := t.TempDir()
	src := filepath.Join(tmpdir, "x.go")
	if err := os.WriteFile(src, []byte(goSourceWithData), 0444); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(tmpdir, "x.exe")

	// Build linker flags
	ldflags := "-D=" + dataAddr
	if roundQuantum != "" {
		ldflags += " -R=" + roundQuantum
	}

	cmd := goCmd(t, "build", "-ldflags="+ldflags, "-o", exe, src)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected build to fail with unaligned data address, but it succeeded")
	}
	if !strings.Contains(string(out), expectedError) {
		t.Errorf("expected error message to contain %q, got:\n%s", expectedError, out)
	}
}

func TestELFHeadersSorted(t *testing.T) {
	for _, buildmode := range []string{"exe", "pie"} {
		t.Run(buildmode, func(t *testing.T) {
			testELFHeadersSorted(t, buildmode)
		})
	}
}

func testELFHeadersSorted(t *testing.T, buildmode string) {
	testenv.MustHaveGoBuild(t)

	// We can only test this for internal linking mode.
	// For external linking the external linker will
	// decide how to sort the sections.
	testenv.MustInternalLink(t, testenv.NoSpecialBuildTypes)
	if buildmode == "pie" {
		testenv.MustInternalLinkPIE(t)
	}

	t.Parallel()

	tmpdir := t.TempDir()
	src := filepath.Join(tmpdir, "x.go")
	if err := os.WriteFile(src, []byte(goSourceWithData), 0o444); err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(tmpdir, "x.exe")
	cmd := goCmd(t, "build", "-buildmode="+buildmode, "-ldflags=-linkmode=internal", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v, output:\n%s", err, out)
	}

	// Check that the first section header is all zeroes.
	f, err := os.Open(exe)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	var ident [elf.EI_NIDENT]byte
	if _, err := f.Read(ident[:]); err != nil {
		t.Fatal(err)
	}

	var bo binary.ByteOrder
	switch elf.Data(ident[elf.EI_DATA]) {
	case elf.ELFDATA2LSB:
		bo = binary.LittleEndian
	case elf.ELFDATA2MSB:
		bo = binary.BigEndian
	default:
		t.Fatalf("unrecognized data encoding %d", ident[elf.EI_DATA])
	}

	var shoff int64
	var shsize int
	switch elf.Class(ident[elf.EI_CLASS]) {
	case elf.ELFCLASS32:
		var hdr elf.Header32
		data := make([]byte, unsafe.Sizeof(hdr))
		if _, err := f.ReadAt(data, 0); err != nil {
			t.Fatal(err)
		}
		shoff = int64(bo.Uint32(data[unsafe.Offsetof(hdr.Shoff):]))
		shsize = int(unsafe.Sizeof(elf.Section32{}))

	case elf.ELFCLASS64:
		var hdr elf.Header64
		data := make([]byte, unsafe.Sizeof(hdr))
		if _, err := f.ReadAt(data, 0); err != nil {
			t.Fatal(err)
		}
		shoff = int64(bo.Uint64(data[unsafe.Offsetof(hdr.Shoff):]))
		shsize = int(unsafe.Sizeof(elf.Section64{}))

	default:
		t.Fatalf("unrecognized class %d", ident[elf.EI_CLASS])
	}

	if shoff > 0 {
		data := make([]byte, shsize)
		if _, err := f.ReadAt(data, shoff); err != nil {
			t.Fatal(err)
		}
		for i, c := range data {
			if c != 0 {
				t.Errorf("section header 0 byte %d is %d, should be zero", i, c)
			}
		}
	}

	ef, err := elf.NewFile(f)
	if err != nil {
		t.Fatal(err)
	}
	defer ef.Close()

	// After the first zero section header,
	// we should see allocated sections,
	// then unallocated sections.
	// The allocated sections should be sorted by address.
	i := 1
	lastAddr := uint64(0)
	for i < len(ef.Sections) {
		sec := ef.Sections[i]
		if sec.Flags&elf.SHF_ALLOC == 0 {
			break
		}
		if sec.Addr < lastAddr {
			t.Errorf("section %d %q address %#x less than previous address %#x", i, sec.Name, sec.Addr, lastAddr)
		}
		lastAddr = sec.Addr
		i++
	}

	firstUnalc := i
	for i < len(ef.Sections) {
		sec := ef.Sections[i]
		if sec.Flags&elf.SHF_ALLOC != 0 {
			t.Errorf("allocated section %d %q follows first unallocated section %d %q", i, sec.Name, firstUnalc, ef.Sections[firstUnalc].Name)
		}
		i++
	}
}
