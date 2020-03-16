package main

import (
	"bufio"
	"bytes"
	"debug/macho"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
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

	const source = `
package main
const X = "\n!\n"
func main() {}
`

	tmpdir, err := ioutil.TempDir("", "issue21703")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v\n", err)
	}
	defer os.RemoveAll(tmpdir)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "main.go"), []byte(source), 0666)
	if err != nil {
		t.Fatalf("failed to write main.go: %v\n", err)
	}

	cmd := exec.Command(testenv.GoToolPath(t), "tool", "compile", "main.go")
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to compile main.go: %v, output: %s\n", err, out)
	}

	cmd = exec.Command(testenv.GoToolPath(t), "tool", "link", "main.o")
	cmd.Dir = tmpdir
	out, err = cmd.CombinedOutput()
	if err != nil {
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

	tmpdir, err := ioutil.TempDir("", "issue28429-")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	write := func(name, content string) {
		err := ioutil.WriteFile(filepath.Join(tmpdir, name), []byte(content), 0666)
		if err != nil {
			t.Fatal(err)
		}
	}

	runGo := func(args ...string) {
		cmd := exec.Command(testenv.GoToolPath(t), args...)
		cmd.Dir = tmpdir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("'go %s' failed: %v, output: %s",
				strings.Join(args, " "), err, out)
		}
	}

	// Compile a main package.
	write("main.go", "package main; func main() {}")
	runGo("tool", "compile", "-p", "main", "main.go")
	runGo("tool", "pack", "c", "main.a", "main.o")

	// Add an extra section with a short, non-.o name.
	// This simulates an alternative build system.
	write(".facts", "this is not an object file")
	runGo("tool", "pack", "r", "main.a", ".facts")

	// Verify that the linker does not attempt
	// to compile the extra section.
	runGo("tool", "link", "main.a")
}

func TestUnresolved(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	tmpdir, err := ioutil.TempDir("", "unresolved-")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	write := func(name, content string) {
		err := ioutil.WriteFile(filepath.Join(tmpdir, name), []byte(content), 0666)
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
	cmd := exec.Command(testenv.GoToolPath(t), "build")
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

func TestBuildForTvOS(t *testing.T) {
	testenv.MustHaveCGO(t)
	testenv.MustHaveGoBuild(t)

	// Only run this on darwin/amd64, where we can cross build for tvOS.
	if runtime.GOARCH != "amd64" || runtime.GOOS != "darwin" {
		t.Skip("skipping on non-darwin/amd64 platform")
	}
	if testing.Short() && os.Getenv("GO_BUILDER_NAME") == "" {
		t.Skip("skipping in -short mode with $GO_BUILDER_NAME empty")
	}
	if err := exec.Command("xcrun", "--help").Run(); err != nil {
		t.Skipf("error running xcrun, required for iOS cross build: %v", err)
	}

	sdkPath, err := exec.Command("xcrun", "--sdk", "appletvos", "--show-sdk-path").Output()
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
		"-framework", "CoreFoundation",
	}
	lib := filepath.Join("testdata", "lib.go")
	tmpDir, err := ioutil.TempDir("", "go-link-TestBuildFortvOS")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	ar := filepath.Join(tmpDir, "lib.a")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-buildmode=c-archive", "-o", ar, lib)
	cmd.Env = append(os.Environ(),
		"CGO_ENABLED=1",
		"GOOS=darwin",
		"GOARCH=arm64",
		"CC="+strings.Join(CC, " "),
		"CGO_CFLAGS=", // ensure CGO_CFLAGS does not contain any flags. Issue #35459
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v: %v:\n%s", cmd.Args, err, out)
	}

	link := exec.Command(CC[0], CC[1:]...)
	link.Args = append(link.Args, ar, filepath.Join("testdata", "main.m"))
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

	tmpdir, err := ioutil.TempDir("", "TestXFlag")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	src := filepath.Join(tmpdir, "main.go")
	err = ioutil.WriteFile(src, []byte(testXFlagSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(testenv.GoToolPath(t), "build", "-ldflags=-X=main.X=meow", "-o", filepath.Join(tmpdir, "main"), src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("%v: %v:\n%s", cmd.Args, err, out)
	}
}

var testMacOSVersionSrc = `
package main
func main() { }
`

func TestMacOSVersion(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	tmpdir, err := ioutil.TempDir("", "TestMacOSVersion")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	src := filepath.Join(tmpdir, "main.go")
	err = ioutil.WriteFile(src, []byte(testMacOSVersionSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(tmpdir, "main")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-ldflags=-linkmode=internal", "-o", exe, src)
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
	exem, err := macho.NewFile(exef)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	const LC_VERSION_MIN_MACOSX = 0x24
	checkMin := func(ver uint32) {
		major, minor := (ver>>16)&0xff, (ver>>8)&0xff
		if major != 10 || minor < 9 {
			t.Errorf("LC_VERSION_MIN_MACOSX version %d.%d < 10.9", major, minor)
		}
	}
	for _, cmd := range exem.Loads {
		raw := cmd.Raw()
		type_ := exem.ByteOrder.Uint32(raw)
		if type_ != LC_VERSION_MIN_MACOSX {
			continue
		}
		osVer := exem.ByteOrder.Uint32(raw[8:])
		checkMin(osVer)
		sdkVer := exem.ByteOrder.Uint32(raw[12:])
		checkMin(sdkVer)
		found = true
		break
	}
	if !found {
		t.Errorf("no LC_VERSION_MIN_MACOSX load command found")
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

	tmpdir, err := ioutil.TempDir("", "TestIssue34788Android386TLSSequence")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	src := filepath.Join(tmpdir, "blah.go")
	err = ioutil.WriteFile(src, []byte(Issue34788src), 0666)
	if err != nil {
		t.Fatal(err)
	}

	obj := filepath.Join(tmpdir, "blah.o")
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "compile", "-o", obj, src)
	cmd.Env = append(os.Environ(), "GOARCH=386", "GOOS=android")
	if out, err := cmd.CombinedOutput(); err != nil {
		if err != nil {
			t.Fatalf("failed to compile blah.go: %v, output: %s\n", err, out)
		}
	}

	// Run objdump on the resulting object.
	cmd = exec.Command(testenv.GoToolPath(t), "tool", "objdump", obj)
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

func TestStrictDup(t *testing.T) {
	// Check that -strictdups flag works.
	testenv.MustHaveGoBuild(t)

	tmpdir, err := ioutil.TempDir("", "TestStrictDup")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	src := filepath.Join(tmpdir, "x.go")
	err = ioutil.WriteFile(src, []byte(testStrictDupGoSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}
	src = filepath.Join(tmpdir, "a.s")
	err = ioutil.WriteFile(src, []byte(testStrictDupAsmSrc1), 0666)
	if err != nil {
		t.Fatal(err)
	}
	src = filepath.Join(tmpdir, "b.s")
	err = ioutil.WriteFile(src, []byte(testStrictDupAsmSrc2), 0666)
	if err != nil {
		t.Fatal(err)
	}
	src = filepath.Join(tmpdir, "go.mod")
	err = ioutil.WriteFile(src, []byte("module teststrictdup\n"), 0666)
	if err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(testenv.GoToolPath(t), "build", "-ldflags=-strictdups=1")
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("linking with -strictdups=1 failed: %v", err)
	}
	if !bytes.Contains(out, []byte("mismatched payload")) {
		t.Errorf("unexpected output:\n%s", out)
	}

	cmd = exec.Command(testenv.GoToolPath(t), "build", "-ldflags=-strictdups=2")
	cmd.Dir = tmpdir
	out, err = cmd.CombinedOutput()
	if err == nil {
		t.Errorf("linking with -strictdups=2 did not fail")
	}
	if !bytes.Contains(out, []byte("mismatched payload")) {
		t.Errorf("unexpected output:\n%s", out)
	}
}
