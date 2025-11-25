// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"debug/macho"
	"errors"
	"internal/platform"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"

	imacho "cmd/internal/macho"
	"cmd/internal/objfile"
	"cmd/internal/sys"
)

var AuthorPaidByTheColumnInch struct {
	fog int `text:"London. Michaelmas term lately over, and the Lord Chancellor sitting in Lincoln’s Inn Hall. Implacable November weather. As much mud in the streets as if the waters had but newly retired from the face of the earth, and it would not be wonderful to meet a Megalosaurus, forty feet long or so, waddling like an elephantine lizard up Holborn Hill. Smoke lowering down from chimney-pots, making a soft black drizzle, with flakes of soot in it as big as full-grown snowflakes—gone into mourning, one might imagine, for the death of the sun. Dogs, undistinguishable in mire. Horses, scarcely better; splashed to their very blinkers. Foot passengers, jostling one another’s umbrellas in a general infection of ill temper, and losing their foot-hold at street-corners, where tens of thousands of other foot passengers have been slipping and sliding since the day broke (if this day ever broke), adding new deposits to the crust upon crust of mud, sticking at those points tenaciously to the pavement, and accumulating at compound interest.  	Fog everywhere. Fog up the river, where it flows among green aits and meadows; fog down the river, where it rolls defiled among the tiers of shipping and the waterside pollutions of a great (and dirty) city. Fog on the Essex marshes, fog on the Kentish heights. Fog creeping into the cabooses of collier-brigs; fog lying out on the yards and hovering in the rigging of great ships; fog drooping on the gunwales of barges and small boats. Fog in the eyes and throats of ancient Greenwich pensioners, wheezing by the firesides of their wards; fog in the stem and bowl of the afternoon pipe of the wrathful skipper, down in his close cabin; fog cruelly pinching the toes and fingers of his shivering little ‘prentice boy on deck. Chance people on the bridges peeping over the parapets into a nether sky of fog, with fog all round them, as if they were up in a balloon and hanging in the misty clouds.  	Gas looming through the fog in divers places in the streets, much as the sun may, from the spongey fields, be seen to loom by husbandman and ploughboy. Most of the shops lighted two hours before their time—as the gas seems to know, for it has a haggard and unwilling look.  	The raw afternoon is rawest, and the dense fog is densest, and the muddy streets are muddiest near that leaden-headed old obstruction, appropriate ornament for the threshold of a leaden-headed old corporation, Temple Bar. And hard by Temple Bar, in Lincoln’s Inn Hall, at the very heart of the fog, sits the Lord High Chancellor in his High Court of Chancery."`

	wind int `text:"It was grand to see how the wind awoke, and bent the trees, and drove the rain before it like a cloud of smoke; and to hear the solemn thunder, and to see the lightning; and while thinking with awe of the tremendous powers by which our little lives are encompassed, to consider how beneficent they are, and how upon the smallest flower and leaf there was already a freshness poured from all this seeming rage, which seemed to make creation new again."`

	jarndyce int `text:"Jarndyce and Jarndyce drones on. This scarecrow of a suit has, over the course of time, become so complicated, that no man alive knows what it means. The parties to it understand it least; but it has been observed that no two Chancery lawyers can talk about it for five minutes, without coming to a total disagreement as to all the premises. Innumerable children have been born into the cause; innumerable young people have married into it; innumerable old people have died out of it. Scores of persons have deliriously found themselves made parties in Jarndyce and Jarndyce, without knowing how or why; whole families have inherited legendary hatreds with the suit. The little plaintiff or defendant, who was promised a new rocking-horse when Jarndyce and Jarndyce should be settled, has grown up, possessed himself of a real horse, and trotted away into the other world. Fair wards of court have faded into mothers and grandmothers; a long procession of Chancellors has come in and gone out; the legion of bills in the suit have been transformed into mere bills of mortality; there are not three Jarndyces left upon the earth perhaps, since old Tom Jarndyce in despair blew his brains out at a coffee-house in Chancery Lane; but Jarndyce and Jarndyce still drags its dreary length before the Court, perennially hopeless."`

	principle int `text:"The one great principle of the English law is, to make business for itself. There is no other principle distinctly, certainly, and consistently maintained through all its narrow turnings. Viewed by this light it becomes a coherent scheme, and not the monstrous maze the laity are apt to think it. Let them but once clearly perceive that its grand principle is to make business for itself at their expense, and surely they will cease to grumble."`
}

func TestLargeSymName(t *testing.T) {
	// The compiler generates a symbol name using the string form of the
	// type. This tests that the linker can read symbol names larger than
	// the bufio buffer. Issue #15104.
	_ = AuthorPaidByTheColumnInch
}

func TestIssue21703(t *testing.T) {
	t.Parallel()

	testenv.MustHaveGoBuild(t)
	// N.B. the build below explicitly doesn't pass through
	// -asan/-msan/-race, so we don't care about those.
	testenv.MustInternalLink(t, testenv.NoSpecialBuildTypes)

	const source = `
package main
const X = "\n!\n"
func main() {}
`

	tmpdir := t.TempDir()
	main := filepath.Join(tmpdir, "main.go")

	err := os.WriteFile(main, []byte(source), 0666)
	if err != nil {
		t.Fatalf("failed to write main.go: %v\n", err)
	}

	importcfgfile := filepath.Join(tmpdir, "importcfg")
	testenv.WriteImportcfg(t, importcfgfile, nil, main)

	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-importcfg="+importcfgfile, "-p=main", "main.go")
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to compile main.go: %v, output: %s\n", err, out)
	}

	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "link", "-importcfg="+importcfgfile, "main.o")
	cmd.Dir = tmpdir
	out, err = cmd.CombinedOutput()
	if err != nil {
		if runtime.GOOS == "android" && runtime.GOARCH == "arm64" {
			testenv.SkipFlaky(t, 58806)
		}
		t.Fatalf("failed to link main.o: %v, output: %s\n", err, out)
	}
}

// TestIssue28429 ensures that the linker does not attempt to link
// sections not named *.o. Such sections may be used by a build system
// to, for example, save facts produced by a modular static analysis
// such as golang.org/x/tools/go/analysis.
func TestIssue28429(t *testing.T) {
	t.Parallel()

	testenv.MustHaveGoBuild(t)
	// N.B. go build below explicitly doesn't pass through
	// -asan/-msan/-race, so we don't care about those.
	testenv.MustInternalLink(t, testenv.NoSpecialBuildTypes)

	tmpdir := t.TempDir()

	write := func(name, content string) {
		err := os.WriteFile(filepath.Join(tmpdir, name), []byte(content), 0666)
		if err != nil {
			t.Fatal(err)
		}
	}

	runGo := func(args ...string) {
		cmd := testenv.Command(t, testenv.GoToolPath(t), args...)
		cmd.Dir = tmpdir
		out, err := cmd.CombinedOutput()
		if err != nil {
			if len(args) >= 2 && args[1] == "link" && runtime.GOOS == "android" && runtime.GOARCH == "arm64" {
				testenv.SkipFlaky(t, 58806)
			}
			t.Fatalf("'go %s' failed: %v, output: %s",
				strings.Join(args, " "), err, out)
		}
	}

	// Compile a main package.
	write("main.go", "package main; func main() {}")
	importcfgfile := filepath.Join(tmpdir, "importcfg")
	testenv.WriteImportcfg(t, importcfgfile, nil, filepath.Join(tmpdir, "main.go"))
	runGo("tool", "compile", "-importcfg="+importcfgfile, "-p=main", "main.go")
	runGo("tool", "pack", "c", "main.a", "main.o")

	// Add an extra section with a short, non-.o name.
	// This simulates an alternative build system.
	write(".facts", "this is not an object file")
	runGo("tool", "pack", "r", "main.a", ".facts")

	// Verify that the linker does not attempt
	// to compile the extra section.
	runGo("tool", "link", "-importcfg="+importcfgfile, "main.a")
}

func TestUnresolved(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()

	write := func(name, content string) {
		err := os.WriteFile(filepath.Join(tmpdir, name), []byte(content), 0666)
		if err != nil {
			t.Fatal(err)
		}
	}

	// Test various undefined references. Because of issue #29852,
	// this used to give confusing error messages because the
	// linker would find an undefined reference to "zero" created
	// by the runtime package.

	write("go.mod", "module testunresolved\n")
	write("main.go", `package main

func main() {
        x()
}

func x()
`)
	write("main.s", `
TEXT ·x(SB),0,$0
        MOVD zero<>(SB), AX
        MOVD zero(SB), AX
        MOVD ·zero(SB), AX
        RET
`)
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build")
	cmd.Dir = tmpdir
	cmd.Env = append(os.Environ(),
		"GOARCH=amd64", "GOOS=linux", "GOPATH="+filepath.Join(tmpdir, "_gopath"))
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected build to fail, but it succeeded")
	}
	out = regexp.MustCompile("(?m)^#.*\n").ReplaceAll(out, nil)
	got := string(out)
	want := `main.x: relocation target zero not defined
main.x: relocation target zero not defined
main.x: relocation target main.zero not defined
`
	if want != got {
		t.Fatalf("want:\n%sgot:\n%s", want, got)
	}
}

func TestIssue33979(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	// N.B. go build below explicitly doesn't pass through
	// -asan/-msan/-race, so we don't care about those.
	testenv.MustInternalLink(t, testenv.SpecialBuildTypes{Cgo: true})

	t.Parallel()

	tmpdir := t.TempDir()

	write := func(name, content string) {
		err := os.WriteFile(filepath.Join(tmpdir, name), []byte(content), 0666)
		if err != nil {
			t.Fatal(err)
		}
	}

	run := func(name string, args ...string) string {
		cmd := testenv.Command(t, name, args...)
		cmd.Dir = tmpdir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("'go %s' failed: %v, output: %s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	runGo := func(args ...string) string {
		return run(testenv.GoToolPath(t), args...)
	}

	// Test object with undefined reference that was not generated
	// by Go, resulting in an SXREF symbol being loaded during linking.
	// Because of issue #33979, the SXREF symbol would be found during
	// error reporting, resulting in confusing error messages.

	write("main.go", `package main
func main() {
        x()
}
func x()
`)
	// The following assembly must work on all architectures.
	write("x.s", `
TEXT ·x(SB),0,$0
        CALL foo(SB)
        RET
`)
	write("x.c", `
void undefined();

void foo() {
        undefined();
}
`)

	cc := strings.TrimSpace(runGo("env", "CC"))
	cflags := strings.Fields(runGo("env", "GOGCCFLAGS"))

	importcfgfile := filepath.Join(tmpdir, "importcfg")
	testenv.WriteImportcfg(t, importcfgfile, nil, "runtime")

	// Compile, assemble and pack the Go and C code.
	runGo("tool", "asm", "-p=main", "-gensymabis", "-o", "symabis", "x.s")
	runGo("tool", "compile", "-importcfg="+importcfgfile, "-symabis", "symabis", "-p=main", "-o", "x1.o", "main.go")
	runGo("tool", "asm", "-p=main", "-o", "x2.o", "x.s")
	run(cc, append(cflags, "-c", "-o", "x3.o", "x.c")...)
	runGo("tool", "pack", "c", "x.a", "x1.o", "x2.o", "x3.o")

	// Now attempt to link using the internal linker.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "link", "-importcfg="+importcfgfile, "-linkmode=internal", "x.a")
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected link to fail, but it succeeded")
	}
	re := regexp.MustCompile(`(?m)^main\(.*text\): relocation target undefined not defined$`)
	if !re.Match(out) {
		t.Fatalf("got:\n%q\nwant:\n%s", out, re)
	}
}

func TestBuildForTvOS(t *testing.T) {
	testenv.MustHaveCGO(t)
	testenv.MustHaveGoBuild(t)

	// Only run this on darwin, where we can cross build for tvOS.
	if runtime.GOOS != "darwin" {
		t.Skip("skipping on non-darwin platform")
	}
	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in -short mode with $GO_BUILDER_NAME empty")
	}
	if err := testenv.Command(t, "xcrun", "--help").Run(); err != nil {
		t.Skipf("error running xcrun, required for iOS cross build: %v", err)
	}

	t.Parallel()

	sdkPath, err := testenv.Command(t, "xcrun", "--sdk", "appletvos", "--show-sdk-path").Output()
	if err != nil {
		t.Skip("failed to locate appletvos SDK, skipping")
	}
	CC := []string{
		"clang",
		"-arch",
		"arm64",
		"-isysroot", strings.TrimSpace(string(sdkPath)),
		"-mtvos-version-min=12.0",
		"-fembed-bitcode",
	}
	CGO_LDFLAGS := []string{"-framework", "CoreFoundation"}
	lib := filepath.Join("testdata", "testBuildFortvOS", "lib.go")
	tmpDir := t.TempDir()

	ar := filepath.Join(tmpDir, "lib.a")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-buildmode=c-archive", "-o", ar, lib)
	env := []string{
		"CGO_ENABLED=1",
		"GOOS=ios",
		"GOARCH=arm64",
		"CC=" + strings.Join(CC, " "),
		"CGO_CFLAGS=", // ensure CGO_CFLAGS does not contain any flags. Issue #35459
		"CGO_LDFLAGS=" + strings.Join(CGO_LDFLAGS, " "),
	}
	cmd.Env = append(os.Environ(), env...)
	t.Logf("%q %v", env, cmd)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v: %v:\n%s", cmd.Args, err, out)
	}

	link := testenv.Command(t, CC[0], CC[1:]...)
	link.Args = append(link.Args, CGO_LDFLAGS...)
	link.Args = append(link.Args, "-o", filepath.Join(tmpDir, "a.out")) // Avoid writing to package directory.
	link.Args = append(link.Args, ar, filepath.Join("testdata", "testBuildFortvOS", "main.m"))
	t.Log(link)
	if out, err := link.CombinedOutput(); err != nil {
		t.Fatalf("%v: %v:\n%s", link.Args, err, out)
	}
}

var testXFlagSrc = `
package main
var X = "hello"
var Z = [99999]int{99998:12345} // make it large enough to be mmaped
func main() { println(X) }
`

func TestXFlag(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "main.go")
	err := os.WriteFile(src, []byte(testXFlagSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-X=main.X=meow", "-o", filepath.Join(tmpdir, "main"), src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("%v: %v:\n%s", cmd.Args, err, out)
	}
}

var trivialSrc = `
package main
func main() { }
`

func TestMachOBuildVersion(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "main.go")
	err := os.WriteFile(src, []byte(trivialSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(tmpdir, "main")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-linkmode=internal", "-o", exe, src)
	cmd.Env = append(os.Environ(),
		"CGO_ENABLED=0",
		"GOOS=darwin",
		"GOARCH=amd64",
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v: %v:\n%s", cmd.Args, err, out)
	}
	exef, err := os.Open(exe)
	if err != nil {
		t.Fatal(err)
	}
	defer exef.Close()
	exem, err := macho.NewFile(exef)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	checkMin := func(ver uint32) {
		major, minor, patch := (ver>>16)&0xff, (ver>>8)&0xff, (ver>>0)&0xff
		if major < 12 {
			t.Errorf("LC_BUILD_VERSION version %d.%d.%d < 12.0.0", major, minor, patch)
		}
	}
	for _, cmd := range exem.Loads {
		raw := cmd.Raw()
		type_ := exem.ByteOrder.Uint32(raw)
		if type_ != imacho.LC_BUILD_VERSION {
			continue
		}
		osVer := exem.ByteOrder.Uint32(raw[12:])
		checkMin(osVer)
		sdkVer := exem.ByteOrder.Uint32(raw[16:])
		checkMin(sdkVer)
		found = true
		break
	}
	if !found {
		t.Errorf("no LC_BUILD_VERSION load command found")
	}
}

func TestMachOUUID(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if runtime.GOOS != "darwin" {
		t.Skip("this is only for darwin")
	}

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "main.go")
	err := os.WriteFile(src, []byte(trivialSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	extractUUID := func(exe string) string {
		exem, err := macho.Open(exe)
		if err != nil {
			t.Fatal(err)
		}
		defer exem.Close()
		for _, cmd := range exem.Loads {
			raw := cmd.Raw()
			type_ := exem.ByteOrder.Uint32(raw)
			if type_ != imacho.LC_UUID {
				continue
			}
			return string(raw[8:24])
		}
		return ""
	}

	tests := []struct{ name, ldflags, expect string }{
		{"default", "", "gobuildid"},
		{"gobuildid", "-B=gobuildid", "gobuildid"},
		{"specific", "-B=0x0123456789ABCDEF0123456789ABCDEF", "\x01\x23\x45\x67\x89\xAB\xCD\xEF\x01\x23\x45\x67\x89\xAB\xCD\xEF"},
		{"none", "-B=none", ""},
	}
	if testenv.HasCGO() {
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
			cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags="+test.ldflags, "-o", exe, src)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("%v: %v:\n%s", cmd.Args, err, out)
			}
			uuid := extractUUID(exe)
			if test.expect == "gobuildid" {
				// Go buildid is not known in source code. Check UUID is present,
				// and satisfies UUIDv3.
				if uuid == "" {
					t.Fatal("expect nonempty UUID, got empty")
				}
				// The version number is the high 4 bits of byte 6.
				if uuid[6]>>4 != 3 {
					t.Errorf("expect v3 UUID, got %X (version %d)", uuid, uuid[6]>>4)
				}
			} else if uuid != test.expect {
				t.Errorf("UUID mismatch: got %X, want %X", uuid, test.expect)
			}
		})
	}
}

const Issue34788src = `

package blah

func Blah(i int) int {
	a := [...]int{1, 2, 3, 4, 5, 6, 7, 8}
	return a[i&7]
}
`

func TestIssue34788Android386TLSSequence(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This is a cross-compilation test, so it doesn't make
	// sense to run it on every GOOS/GOARCH combination. Limit
	// the test to amd64 + darwin/linux.
	if runtime.GOARCH != "amd64" ||
		(runtime.GOOS != "darwin" && runtime.GOOS != "linux") {
		t.Skip("skipping on non-{linux,darwin}/amd64 platform")
	}

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "blah.go")
	err := os.WriteFile(src, []byte(Issue34788src), 0666)
	if err != nil {
		t.Fatal(err)
	}

	obj := filepath.Join(tmpdir, "blah.o")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-p=blah", "-o", obj, src)
	cmd.Env = append(os.Environ(), "GOARCH=386", "GOOS=android")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("failed to compile blah.go: %v, output: %s\n", err, out)
	}

	// Run objdump on the resulting object.
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "objdump", obj)
	out, oerr := cmd.CombinedOutput()
	if oerr != nil {
		t.Fatalf("failed to objdump blah.o: %v, output: %s\n", oerr, out)
	}

	// Sift through the output; we should not be seeing any R_TLS_LE relocs.
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "R_TLS_LE") {
			t.Errorf("objdump output contains unexpected R_TLS_LE reloc: %s", line)
		}
	}
}

const testStrictDupGoSrc = `
package main
func f()
func main() { f() }
`

const testStrictDupAsmSrc1 = `
#include "textflag.h"
TEXT	·f(SB), NOSPLIT|DUPOK, $0-0
	RET
`

const testStrictDupAsmSrc2 = `
#include "textflag.h"
TEXT	·f(SB), NOSPLIT|DUPOK, $0-0
	JMP	0(PC)
`

const testStrictDupAsmSrc3 = `
#include "textflag.h"
GLOBL ·rcon(SB), RODATA|DUPOK, $64
`

const testStrictDupAsmSrc4 = `
#include "textflag.h"
GLOBL ·rcon(SB), RODATA|DUPOK, $32
`

func TestStrictDup(t *testing.T) {
	// Check that -strictdups flag works.
	testenv.MustHaveGoBuild(t)

	asmfiles := []struct {
		fname   string
		payload string
	}{
		{"a", testStrictDupAsmSrc1},
		{"b", testStrictDupAsmSrc2},
		{"c", testStrictDupAsmSrc3},
		{"d", testStrictDupAsmSrc4},
	}

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "x.go")
	err := os.WriteFile(src, []byte(testStrictDupGoSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}
	for _, af := range asmfiles {
		src = filepath.Join(tmpdir, af.fname+".s")
		err = os.WriteFile(src, []byte(af.payload), 0666)
		if err != nil {
			t.Fatal(err)
		}
	}
	src = filepath.Join(tmpdir, "go.mod")
	err = os.WriteFile(src, []byte("module teststrictdup\n"), 0666)
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-strictdups=1")
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("linking with -strictdups=1 failed: %v\n%s", err, string(out))
	}
	if !bytes.Contains(out, []byte("mismatched payload")) {
		t.Errorf("unexpected output:\n%s", out)
	}

	cmd = testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-strictdups=2")
	cmd.Dir = tmpdir
	out, err = cmd.CombinedOutput()
	if err == nil {
		t.Errorf("linking with -strictdups=2 did not fail")
	}
	// NB: on amd64 we get the 'new length' error, on arm64 the 'different
	// contents' error.
	if !(bytes.Contains(out, []byte("mismatched payload: new length")) ||
		bytes.Contains(out, []byte("mismatched payload: same length but different contents"))) ||
		!bytes.Contains(out, []byte("mismatched payload: different sizes")) {
		t.Errorf("unexpected output:\n%s", out)
	}
}

const testFuncAlignSrc = `
package main
import (
	"fmt"
)
func alignPc()
var alignPcFnAddr uintptr

func main() {
	if alignPcFnAddr % 512 != 0 {
		fmt.Printf("expected 512 bytes alignment, got %v\n", alignPcFnAddr)
	} else {
		fmt.Printf("PASS")
	}
}
`

var testFuncAlignAsmSources = map[string]string{
	"arm64": `
#include "textflag.h"

TEXT	·alignPc(SB),NOSPLIT, $0-0
	MOVD	$2, R0
	PCALIGN	$512
	MOVD	$3, R1
	RET

GLOBL	·alignPcFnAddr(SB),RODATA,$8
DATA	·alignPcFnAddr(SB)/8,$·alignPc(SB)
`,
	"loong64": `
#include "textflag.h"

TEXT	·alignPc(SB),NOSPLIT, $0-0
	MOVV	$2, R4
	PCALIGN	$512
	MOVV	$3, R5
	RET

GLOBL	·alignPcFnAddr(SB),RODATA,$8
DATA	·alignPcFnAddr(SB)/8,$·alignPc(SB)
`,
}

// TestFuncAlign verifies that the address of a function can be aligned
// with a specific value on arm64 and loong64.
func TestFuncAlign(t *testing.T) {
	testFuncAlignAsmSrc := testFuncAlignAsmSources[runtime.GOARCH]
	if len(testFuncAlignAsmSrc) == 0 || runtime.GOOS != "linux" {
		t.Skip("skipping on non-linux/{arm64,loong64} platform")
	}
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "go.mod")
	err := os.WriteFile(src, []byte("module cmd/link/TestFuncAlign/falign"), 0666)
	if err != nil {
		t.Fatal(err)
	}
	src = filepath.Join(tmpdir, "falign.go")
	err = os.WriteFile(src, []byte(testFuncAlignSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}
	src = filepath.Join(tmpdir, "falign.s")
	err = os.WriteFile(src, []byte(testFuncAlignAsmSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", "falign")
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("build failed: %v", err)
	}
	cmd = testenv.Command(t, tmpdir+"/falign")
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Errorf("failed to run with err %v, output: %s", err, out)
	}
	if string(out) != "PASS" {
		t.Errorf("unexpected output: %s\n", out)
	}
}

const testFuncAlignOptionSrc = `
package main
//go:noinline
func foo() {
}
//go:noinline
func bar() {
}
//go:noinline
func baz() {
}
func main() {
	foo()
	bar()
	baz()
}
`

// TestFuncAlignOption verifies that the -funcalign option changes the function alignment
func TestFuncAlignOption(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "falign.go")
	err := os.WriteFile(src, []byte(testFuncAlignOptionSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	alignTest := func(align uint64) {
		exeName := "falign.exe"
		cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-funcalign="+strconv.FormatUint(align, 10), "-o", exeName, "falign.go")
		cmd.Dir = tmpdir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("build failed: %v \n%s", err, out)
		}
		exe := filepath.Join(tmpdir, exeName)
		cmd = testenv.Command(t, exe)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Errorf("failed to run with err %v, output: %s", err, out)
		}

		// Check function alignment
		f, err := objfile.Open(exe)
		if err != nil {
			t.Fatalf("failed to open file:%v\n", err)
		}
		defer f.Close()

		fname := map[string]bool{"_main.foo": false,
			"_main.bar": false,
			"_main.baz": false}
		syms, err := f.Symbols()
		for _, s := range syms {
			fn := s.Name
			if _, ok := fname[fn]; !ok {
				fn = "_" + s.Name
				if _, ok := fname[fn]; !ok {
					continue
				}
			}
			if s.Addr%align != 0 {
				t.Fatalf("unaligned function: %s %x. Expected alignment: %d\n", fn, s.Addr, align)
			}
			fname[fn] = true
		}
		for k, v := range fname {
			if !v {
				t.Fatalf("function %s not found\n", k)
			}
		}
	}
	alignTest(16)
	alignTest(32)
}

const testTrampSrc = `
package main
import "fmt"
func main() {
	fmt.Println("hello")

	defer func(){
		if e := recover(); e == nil {
			panic("did not panic")
		}
	}()
	f1()
}

// Test deferreturn trampolines. See issue #39049.
func f1() { defer f2() }
func f2() { panic("XXX") }
`

func TestTrampoline(t *testing.T) {
	// Test that trampoline insertion works as expected.
	// For stress test, we set -debugtramp=2 flag, which sets a very low
	// threshold for trampoline generation, and essentially all cross-package
	// calls will use trampolines.
	buildmodes := []string{"default"}
	switch runtime.GOARCH {
	case "arm", "arm64", "ppc64", "loong64":
	case "ppc64le":
		// Trampolines are generated differently when internal linking PIE, test them too.
		buildmodes = append(buildmodes, "pie")
	default:
		t.Skipf("trampoline insertion is not implemented on %s", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "hello.go")
	err := os.WriteFile(src, []byte(testTrampSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(tmpdir, "hello.exe")

	for _, mode := range buildmodes {
		cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-buildmode="+mode, "-ldflags=-debugtramp=2", "-o", exe, src)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("build (%s) failed: %v\n%s", mode, err, out)
		}
		cmd = testenv.Command(t, exe)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Errorf("executable failed to run (%s): %v\n%s", mode, err, out)
		}
		if string(out) != "hello\n" {
			t.Errorf("unexpected output (%s):\n%s", mode, out)
		}

		out, err = testenv.Command(t, testenv.GoToolPath(t), "tool", "nm", exe).CombinedOutput()
		if err != nil {
			t.Errorf("nm failure: %s\n%s\n", err, string(out))
		}
		if ok, _ := regexp.Match("T runtime.deferreturn(\\+0)?-tramp0", out); !ok {
			t.Errorf("Trampoline T runtime.deferreturn(+0)?-tramp0 is missing")
		}
	}
}

const testTrampCgoSrc = `
package main

// #include <stdio.h>
// void CHello() { printf("hello\n"); fflush(stdout); }
import "C"

func main() {
	C.CHello()
}
`

func TestTrampolineCgo(t *testing.T) {
	// Test that trampoline insertion works for cgo code.
	// For stress test, we set -debugtramp=2 flag, which sets a very low
	// threshold for trampoline generation, and essentially all cross-package
	// calls will use trampolines.
	buildmodes := []string{"default"}
	switch runtime.GOARCH {
	case "arm", "arm64", "ppc64", "loong64":
	case "ppc64le":
		// Trampolines are generated differently when internal linking PIE, test them too.
		buildmodes = append(buildmodes, "pie")
	default:
		t.Skipf("trampoline insertion is not implemented on %s", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "hello.go")
	err := os.WriteFile(src, []byte(testTrampCgoSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(tmpdir, "hello.exe")

	for _, mode := range buildmodes {
		cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-buildmode="+mode, "-ldflags=-debugtramp=2", "-o", exe, src)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("build (%s) failed: %v\n%s", mode, err, out)
		}
		cmd = testenv.Command(t, exe)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Errorf("executable failed to run (%s): %v\n%s", mode, err, out)
		}
		if string(out) != "hello\n" && string(out) != "hello\r\n" {
			t.Errorf("unexpected output (%s):\n%s", mode, out)
		}

		// Test internal linking mode.

		if !testenv.CanInternalLink(true) {
			continue
		}
		cmd = testenv.Command(t, testenv.GoToolPath(t), "build", "-buildmode="+mode, "-ldflags=-debugtramp=2 -linkmode=internal", "-o", exe, src)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("build (%s) failed: %v\n%s", mode, err, out)
		}
		cmd = testenv.Command(t, exe)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Errorf("executable failed to run (%s): %v\n%s", mode, err, out)
		}
		if string(out) != "hello\n" && string(out) != "hello\r\n" {
			t.Errorf("unexpected output (%s):\n%s", mode, out)
		}
	}
}

func TestIndexMismatch(t *testing.T) {
	// Test that index mismatch will cause a link-time error (not run-time error).
	// This shouldn't happen with "go build". We invoke the compiler and the linker
	// manually, and try to "trick" the linker with an inconsistent object file.
	testenv.MustHaveGoBuild(t)
	// N.B. the build below explicitly doesn't pass through
	// -asan/-msan/-race, so we don't care about those.
	testenv.MustInternalLink(t, testenv.NoSpecialBuildTypes)

	t.Parallel()

	tmpdir := t.TempDir()

	aSrc := filepath.Join("testdata", "testIndexMismatch", "a.go")
	bSrc := filepath.Join("testdata", "testIndexMismatch", "b.go")
	mSrc := filepath.Join("testdata", "testIndexMismatch", "main.go")
	aObj := filepath.Join(tmpdir, "a.o")
	mObj := filepath.Join(tmpdir, "main.o")
	exe := filepath.Join(tmpdir, "main.exe")

	importcfgFile := filepath.Join(tmpdir, "runtime.importcfg")
	testenv.WriteImportcfg(t, importcfgFile, nil, "runtime")
	importcfgWithAFile := filepath.Join(tmpdir, "witha.importcfg")
	testenv.WriteImportcfg(t, importcfgWithAFile, map[string]string{"a": aObj}, "runtime")

	// Build a program with main package importing package a.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-importcfg="+importcfgFile, "-p=a", "-o", aObj, aSrc)
	t.Log(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compiling a.go failed: %v\n%s", err, out)
	}
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-importcfg="+importcfgWithAFile, "-p=main", "-I", tmpdir, "-o", mObj, mSrc)
	t.Log(cmd)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compiling main.go failed: %v\n%s", err, out)
	}
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "link", "-importcfg="+importcfgWithAFile, "-L", tmpdir, "-o", exe, mObj)
	t.Log(cmd)
	out, err = cmd.CombinedOutput()
	if err != nil {
		if runtime.GOOS == "android" && runtime.GOARCH == "arm64" {
			testenv.SkipFlaky(t, 58806)
		}
		t.Errorf("linking failed: %v\n%s", err, out)
	}

	// Now, overwrite a.o with the object of b.go. This should
	// result in an index mismatch.
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-importcfg="+importcfgFile, "-p=a", "-o", aObj, bSrc)
	t.Log(cmd)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compiling a.go failed: %v\n%s", err, out)
	}
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "link", "-importcfg="+importcfgWithAFile, "-L", tmpdir, "-o", exe, mObj)
	t.Log(cmd)
	out, err = cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("linking didn't fail")
	}
	if !bytes.Contains(out, []byte("fingerprint mismatch")) {
		t.Errorf("did not see expected error message. out:\n%s", out)
	}
}

func TestPErsrcBinutils(t *testing.T) {
	// Test that PE rsrc section is handled correctly (issue 39658).
	testenv.MustHaveGoBuild(t)

	if (runtime.GOARCH != "386" && runtime.GOARCH != "amd64") || runtime.GOOS != "windows" {
		// This test is limited to amd64 and 386, because binutils is limited as such
		t.Skipf("this is only for windows/amd64 and windows/386")
	}

	t.Parallel()

	tmpdir := t.TempDir()

	pkgdir := filepath.Join("testdata", "pe-binutils")
	exe := filepath.Join(tmpdir, "a.exe")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe)
	cmd.Dir = pkgdir
	// cmd.Env = append(os.Environ(), "GOOS=windows", "GOARCH=amd64") // uncomment if debugging in a cross-compiling environment
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building failed: %v, output:\n%s", err, out)
	}

	// Check that the binary contains the rsrc data
	b, err := os.ReadFile(exe)
	if err != nil {
		t.Fatalf("reading output failed: %v", err)
	}
	if !bytes.Contains(b, []byte("Hello Gophers!")) {
		t.Fatalf("binary does not contain expected content")
	}
}

func TestPErsrcLLVM(t *testing.T) {
	// Test that PE rsrc section is handled correctly (issue 39658).
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS != "windows" {
		t.Skipf("this is a windows-only test")
	}

	t.Parallel()

	tmpdir := t.TempDir()

	pkgdir := filepath.Join("testdata", "pe-llvm")
	exe := filepath.Join(tmpdir, "a.exe")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe)
	cmd.Dir = pkgdir
	// cmd.Env = append(os.Environ(), "GOOS=windows", "GOARCH=amd64") // uncomment if debugging in a cross-compiling environment
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building failed: %v, output:\n%s", err, out)
	}

	// Check that the binary contains the rsrc data
	b, err := os.ReadFile(exe)
	if err != nil {
		t.Fatalf("reading output failed: %v", err)
	}
	if !bytes.Contains(b, []byte("resname RCDATA a.rc")) {
		t.Fatalf("binary does not contain expected content")
	}
}

func TestContentAddressableSymbols(t *testing.T) {
	// Test that the linker handles content-addressable symbols correctly.
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	src := filepath.Join("testdata", "testHashedSyms", "p.go")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "run", src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("command %s failed: %v\n%s", cmd, err, out)
	}
}

func TestReadOnly(t *testing.T) {
	// Test that read-only data is indeed read-only.
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	src := filepath.Join("testdata", "testRO", "x.go")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "run", src)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Errorf("running test program did not fail. output:\n%s", out)
	}
}

const testIssue38554Src = `
package main

type T [10<<20]byte

//go:noinline
func f() T {
	return T{} // compiler will make a large stmp symbol, but not used.
}

func main() {
	x := f()
	println(x[1])
}
`

func TestIssue38554(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "x.go")
	err := os.WriteFile(src, []byte(testIssue38554Src), 0666)
	if err != nil {
		t.Fatalf("failed to write source file: %v", err)
	}
	exe := filepath.Join(tmpdir, "x.exe")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe, src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}

	fi, err := os.Stat(exe)
	if err != nil {
		t.Fatalf("failed to stat output file: %v", err)
	}

	// The test program is not much different from a helloworld, which is
	// typically a little over 1 MB. We allow 5 MB. If the bad stmp is live,
	// it will be over 10 MB.
	const want = 5 << 20
	if got := fi.Size(); got > want {
		t.Errorf("binary too big: got %d, want < %d", got, want)
	}
}

const testIssue42396src = `
package main

//go:noinline
//go:nosplit
func callee(x int) {
}

func main() {
	callee(9)
}
`

func TestIssue42396(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if !platform.RaceDetectorSupported(runtime.GOOS, runtime.GOARCH) {
		t.Skip("no race detector support")
	}

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "main.go")
	err := os.WriteFile(src, []byte(testIssue42396src), 0666)
	if err != nil {
		t.Fatalf("failed to write source file: %v", err)
	}
	exe := filepath.Join(tmpdir, "main.exe")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-gcflags=-race", "-o", exe, src)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("build unexpectedly succeeded")
	}

	// Check to make sure that we see a reasonable error message
	// and not a panic.
	if strings.Contains(string(out), "panic:") {
		t.Fatalf("build should not fail with panic:\n%s", out)
	}
	const want = "reference to undefined builtin"
	if !strings.Contains(string(out), want) {
		t.Fatalf("error message incorrect: expected it to contain %q but instead got:\n%s\n", want, out)
	}
}

const testLargeRelocSrc = `
package main

var x = [1<<25]byte{1<<23: 23, 1<<24: 24}

var addr = [...]*byte{
	&x[1<<23-1],
	&x[1<<23],
	&x[1<<23+1],
	&x[1<<24-1],
	&x[1<<24],
	&x[1<<24+1],
}

func main() {
	// check relocations in instructions
	check(x[1<<23-1], 0)
	check(x[1<<23], 23)
	check(x[1<<23+1], 0)
	check(x[1<<24-1], 0)
	check(x[1<<24], 24)
	check(x[1<<24+1], 0)

	// check absolute address relocations in data
	check(*addr[0], 0)
	check(*addr[1], 23)
	check(*addr[2], 0)
	check(*addr[3], 0)
	check(*addr[4], 24)
	check(*addr[5], 0)
}

func check(x, y byte) {
	if x != y {
		panic("FAIL")
	}
}
`

func TestLargeReloc(t *testing.T) {
	// Test that large relocation addend is handled correctly.
	// In particular, on darwin/arm64 when external linking,
	// Mach-O relocation has only 24-bit addend. See issue #42738.
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "x.go")
	err := os.WriteFile(src, []byte(testLargeRelocSrc), 0666)
	if err != nil {
		t.Fatalf("failed to write source file: %v", err)
	}
	cmd := testenv.Command(t, testenv.GoToolPath(t), "run", src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("build failed: %v. output:\n%s", err, out)
	}

	if testenv.HasCGO() { // currently all targets that support cgo can external link
		cmd = testenv.Command(t, testenv.GoToolPath(t), "run", "-ldflags=-linkmode=external", src)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("build failed: %v. output:\n%s", err, out)
		}
	}
}

func TestUnlinkableObj(t *testing.T) {
	// Test that the linker emits an error with unlinkable object.
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	if true /* was buildcfg.Experiment.Unified */ {
		t.Skip("TODO(mdempsky): Fix ICE when importing unlinkable objects for GOEXPERIMENT=unified")
	}

	tmpdir := t.TempDir()

	xSrc := filepath.Join(tmpdir, "x.go")
	pSrc := filepath.Join(tmpdir, "p.go")
	xObj := filepath.Join(tmpdir, "x.o")
	pObj := filepath.Join(tmpdir, "p.o")
	exe := filepath.Join(tmpdir, "x.exe")
	importcfgfile := filepath.Join(tmpdir, "importcfg")
	testenv.WriteImportcfg(t, importcfgfile, map[string]string{"p": pObj})
	err := os.WriteFile(xSrc, []byte("package main\nimport _ \"p\"\nfunc main() {}\n"), 0666)
	if err != nil {
		t.Fatalf("failed to write source file: %v", err)
	}
	err = os.WriteFile(pSrc, []byte("package p\n"), 0666)
	if err != nil {
		t.Fatalf("failed to write source file: %v", err)
	}
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-importcfg="+importcfgfile, "-o", pObj, pSrc) // without -p
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile p.go failed: %v. output:\n%s", err, out)
	}
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-importcfg="+importcfgfile, "-p=main", "-o", xObj, xSrc)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile x.go failed: %v. output:\n%s", err, out)
	}
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "link", "-importcfg="+importcfgfile, "-o", exe, xObj)
	out, err = cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("link did not fail")
	}
	if !bytes.Contains(out, []byte("unlinkable object")) {
		t.Errorf("did not see expected error message. out:\n%s", out)
	}

	// It is okay to omit -p for (only) main package.
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-importcfg="+importcfgfile, "-p=p", "-o", pObj, pSrc)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile p.go failed: %v. output:\n%s", err, out)
	}
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-importcfg="+importcfgfile, "-o", xObj, xSrc) // without -p
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile failed: %v. output:\n%s", err, out)
	}

	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "link", "-importcfg="+importcfgfile, "-o", exe, xObj)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Errorf("link failed: %v. output:\n%s", err, out)
	}
}

func TestExtLinkCmdlineDeterminism(t *testing.T) {
	// Test that we pass flags in deterministic order to the external linker
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t) // this test requires -linkmode=external
	t.Parallel()

	// test source code, with some cgo exports
	testSrc := `
package main
import "C"
//export F1
func F1() {}
//export F2
func F2() {}
//export F3
func F3() {}
func main() {}
`

	tmpdir := t.TempDir()
	src := filepath.Join(tmpdir, "x.go")
	if err := os.WriteFile(src, []byte(testSrc), 0666); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(tmpdir, "x.exe")

	// Use a deterministic tmp directory so the temporary file paths are
	// deterministic.
	linktmp := filepath.Join(tmpdir, "linktmp")
	if err := os.Mkdir(linktmp, 0777); err != nil {
		t.Fatal(err)
	}

	// Link with -v -linkmode=external to see the flags we pass to the
	// external linker.
	ldflags := "-ldflags=-v -linkmode=external -tmpdir=" + linktmp
	var out0 []byte
	for i := 0; i < 5; i++ {
		cmd := testenv.Command(t, testenv.GoToolPath(t), "build", ldflags, "-o", exe, src)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("build failed: %v, output:\n%s", err, out)
		}
		if err := os.Remove(exe); err != nil {
			t.Fatal(err)
		}

		// extract the "host link" invocation
		j := bytes.Index(out, []byte("\nhost link:"))
		if j == -1 {
			t.Fatalf("host link step not found, output:\n%s", out)
		}
		out = out[j+1:]
		k := bytes.Index(out, []byte("\n"))
		if k == -1 {
			t.Fatalf("no newline after host link, output:\n%s", out)
		}
		out = out[:k]

		// filter out output file name, which is passed by the go
		// command and is nondeterministic.
		fs := bytes.Fields(out)
		for i, f := range fs {
			if bytes.Equal(f, []byte(`"-o"`)) && i+1 < len(fs) {
				fs[i+1] = []byte("a.out")
				break
			}
		}
		out = bytes.Join(fs, []byte{' '})

		if i == 0 {
			out0 = out
			continue
		}
		if !bytes.Equal(out0, out) {
			t.Fatalf("output differ:\n%s\n==========\n%s", out0, out)
		}
	}
}

// TestResponseFile tests that creating a response file to pass to the
// external linker works correctly.
func TestResponseFile(t *testing.T) {
	t.Parallel()

	testenv.MustHaveGoBuild(t)

	// This test requires -linkmode=external. Currently all
	// systems that support cgo support -linkmode=external.
	testenv.MustHaveCGO(t)

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "x.go")
	if err := os.WriteFile(src, []byte(`package main; import "C"; func main() {}`), 0666); err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", "output", "x.go")
	cmd.Dir = tmpdir

	// Add enough arguments to push cmd/link into creating a response file.
	var sb strings.Builder
	sb.WriteString(`'-ldflags=all="-extldflags=`)
	for i := 0; i < sys.ExecArgLengthLimit/len("-g"); i++ {
		if i > 0 {
			sb.WriteString(" ")
		}
		sb.WriteString("-g")
	}
	sb.WriteString(`"'`)
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GOFLAGS="+sb.String())

	out, err := cmd.CombinedOutput()
	if len(out) > 0 {
		t.Logf("%s", out)
	}
	if err != nil {
		t.Error(err)
	}
}

func TestDynimportVar(t *testing.T) {
	// Test that we can access dynamically imported variables.
	// Currently darwin only.
	if runtime.GOOS != "darwin" {
		t.Skip("skip on non-darwin platform")
	}

	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	t.Parallel()

	tmpdir := t.TempDir()
	exe := filepath.Join(tmpdir, "a.exe")
	src := filepath.Join("testdata", "dynimportvar", "main.go")

	for _, mode := range []string{"internal", "external"} {
		cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-linkmode="+mode, "-o", exe, src)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("build (linkmode=%s) failed: %v\n%s", mode, err, out)
		}
		cmd = testenv.Command(t, exe)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Errorf("executable failed to run (%s): %v\n%s", mode, err, out)
		}
	}
}

const helloSrc = `
package main
var X = 42
var Y int
func main() { println("hello", X, Y) }
`

func TestFlagS(t *testing.T) {
	// Test that the -s flag strips the symbol table.
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()
	exe := filepath.Join(tmpdir, "a.exe")
	src := filepath.Join(tmpdir, "a.go")
	err := os.WriteFile(src, []byte(helloSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	modes := []string{"auto"}
	if testenv.HasCGO() {
		modes = append(modes, "external")
	}

	// check a text symbol, a data symbol, and a BSS symbol
	syms := []string{"main.main", "main.X", "main.Y"}

	for _, mode := range modes {
		cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-s -linkmode="+mode, "-o", exe, src)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("build (linkmode=%s) failed: %v\n%s", mode, err, out)
		}
		cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "nm", exe)
		out, err = cmd.CombinedOutput()
		if err != nil {
			if _, ok := errors.AsType[*exec.ExitError](err); !ok {
				// Error exit is fine as it may have no symbols.
				// On darwin we need to emit dynamic symbol references so it
				// actually has some symbols, and nm succeeds.
				t.Errorf("(mode=%s) go tool nm failed: %v\n%s", mode, err, out)
			}
		}
		for _, s := range syms {
			if bytes.Contains(out, []byte(s)) {
				t.Errorf("(mode=%s): unexpected symbol %s", mode, s)
			}
		}
	}
}

func TestRandLayout(t *testing.T) {
	// Test that the -randlayout flag randomizes function order and
	// generates a working binary.
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join(tmpdir, "hello.go")
	err := os.WriteFile(src, []byte(trivialSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	var syms [2]string
	for i, seed := range []string{"123", "456"} {
		exe := filepath.Join(tmpdir, "hello"+seed+".exe")
		cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-randlayout="+seed, "-o", exe, src)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("seed=%v: build failed: %v\n%s", seed, err, out)
		}
		cmd = testenv.Command(t, exe)
		err = cmd.Run()
		if err != nil {
			t.Fatalf("seed=%v: executable failed to run: %v\n%s", seed, err, out)
		}
		cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "nm", exe)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("seed=%v: fail to run \"go tool nm\": %v\n%s", seed, err, out)
		}
		syms[i] = string(out)
	}
	if syms[0] == syms[1] {
		t.Errorf("randlayout with different seeds produced same layout:\n%s\n===\n\n%s", syms[0], syms[1])
	}
}

func TestCheckLinkname(t *testing.T) {
	// Test that code containing blocked linknames does not build.
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	tmpdir := t.TempDir()

	tests := []struct {
		src string
		ok  bool
	}{
		// use (instantiation) of public API is ok
		{"ok.go", true},
		// push linkname is ok
		{"push.go", true},
		// using a linknamed variable to reference an assembly
		// function in the same package is ok
		{"textvar", true},
		// pull linkname of blocked symbol is not ok
		{"coro.go", false},
		{"coro_var.go", false},
		// assembly reference is not ok
		{"coro_asm", false},
		// pull-only linkname is not ok
		{"coro2.go", false},
		// pull linkname of a builtin symbol is not ok
		{"builtin.go", false},
		{"addmoduledata.go", false},
		{"freegc.go", false},
		// legacy bad linkname is ok, for now
		{"fastrand.go", true},
		{"badlinkname.go", true},
	}
	for _, test := range tests {
		test := test
		t.Run(test.src, func(t *testing.T) {
			t.Parallel()
			src := "./testdata/linkname/" + test.src
			exe := filepath.Join(tmpdir, test.src+".exe")
			cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe, src)
			out, err := cmd.CombinedOutput()
			if test.ok && err != nil {
				t.Errorf("build failed unexpectedly: %v:\n%s", err, out)
			}
			if !test.ok && err == nil {
				t.Errorf("build succeeded unexpectedly: %v:\n%s", err, out)
			}
		})
	}
}

func TestLinknameBSS(t *testing.T) {
	// Test that the linker chooses the right one as the definition
	// for linknamed variables. See issue #72032.
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	tmpdir := t.TempDir()

	src := filepath.Join("testdata", "linkname", "sched.go")
	exe := filepath.Join(tmpdir, "sched.exe")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe, src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build failed unexpectedly: %v:\n%s", err, out)
	}

	// Check the symbol size.
	f, err := objfile.Open(exe)
	if err != nil {
		t.Fatalf("fail to open executable: %v", err)
	}
	defer f.Close()
	syms, err := f.Symbols()
	if err != nil {
		t.Fatalf("fail to get symbols: %v", err)
	}
	found := false
	for _, s := range syms {
		if s.Name == "runtime.sched" || s.Name == "_runtime.sched" {
			found = true
			if s.Size < 100 {
				// As of Go 1.25 (Mar 2025), runtime.sched has 6848 bytes on
				// darwin/arm64. It should always be larger than 100 bytes on
				// all platforms.
				t.Errorf("runtime.sched symbol size too small: want > 100, got %d", s.Size)
			}
		}
	}
	if !found {
		t.Errorf("runtime.sched symbol not found")
	}

	// Executable should run.
	cmd = testenv.Command(t, exe)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Errorf("executable failed to run: %v\n%s", err, out)
	}
}
