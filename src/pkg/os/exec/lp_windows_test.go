// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"text/template"
)

type lookPathTest struct {
	PATH      string
	PATHEXT   string
	files     []string
	searchFor string
	fails     bool // test is expected to fail
}

// PrefixPATH returns p.PATH with every element prefixed by prefix.
func (t lookPathTest) PrefixPATH(prefix string) string {
	a := strings.SplitN(t.PATH, ";", -1)
	for i := range a {
		a[i] = filepath.Join(prefix, a[i])
	}
	return strings.Join(a, ";")
}

var lookPathTests = []lookPathTest{
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.exe`, `p2\a.exe`, `p2\a`},
		searchFor: `a`,
	},
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1.dir;p2.dir`,
		files:     []string{`p1.dir\a`, `p2.dir\a.exe`},
		searchFor: `a`,
	},
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `a.exe`,
	},
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.exe`, `p2\b.exe`},
		searchFor: `b`,
	},
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\b`, `p2\a`},
		searchFor: `a`,
		fails:     true, // TODO(brainman): do not know why this fails
	},
	// If the command name specifies a path, the shell searches
	// the specified path for an executable file matching
	// the command name. If a match is found, the external
	// command (the executable file) executes.
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `p2\a`,
	},
	// If the command name specifies a path, the shell searches
	// the specified path for an executable file matching the command
	// name. ... If no match is found, the shell reports an error
	// and command processing completes.
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\b.exe`, `p2\a.exe`},
		searchFor: `p2\b`,
		fails:     true,
	},
	// If the command name does not specify a path, the shell
	// searches the current directory for an executable file
	// matching the command name. If a match is found, the external
	// command (the executable file) executes.
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`a`, `p1\a.exe`, `p2\a.exe`},
		searchFor: `a`,
	},
	// The shell now searches each directory specified by the
	// PATH environment variable, in the order listed, for an
	// executable file matching the command name. If a match
	// is found, the external command (the executable file) executes.
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `a`,
	},
	// The shell now searches each directory specified by the
	// PATH environment variable, in the order listed, for an
	// executable file matching the command name. If no match
	// is found, the shell reports an error and command processing
	// completes.
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `b`,
		fails:     true,
	},
	// If the command name includes a file extension, the shell
	// searches each directory for the exact file name specified
	// by the command name.
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `a.exe`,
	},
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `a.com`,
		fails:     true, // includes extension and not exact file name match
	},
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1`,
		files:     []string{`p1\a.exe.exe`},
		searchFor: `a.exe`,
	},
	{
		PATHEXT:   `.COM;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `a.exe`,
	},
	// If the command name does not include a file extension, the shell
	// adds the extensions listed in the PATHEXT environment variable,
	// one by one, and searches the directory for that file name. Note
	// that the shell tries all possible file extensions in a specific
	// directory before moving on to search the next directory
	// (if there is one).
	{
		PATHEXT:   `.COM;.EXE`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.bat`, `p2\a.exe`},
		searchFor: `a`,
	},
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.bat`, `p2\a.exe`},
		searchFor: `a`,
	},
	{
		PATHEXT:   `.COM;.EXE;.BAT`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.bat`, `p1\a.exe`, `p2\a.bat`, `p2\a.exe`},
		searchFor: `a`,
	},
	{
		PATHEXT:   `.COM`,
		PATH:      `p1;p2`,
		files:     []string{`p1\a.bat`, `p2\a.exe`},
		searchFor: `a`,
		fails:     true, // tried all extensions in PATHEXT, but none matches
	},
}

func updateEnv(env []string, name, value string) []string {
	for i, e := range env {
		if strings.HasPrefix(strings.ToUpper(e), name+"=") {
			env[i] = name + "=" + value
			return env
		}
	}
	return append(env, name+"="+value)
}

func installExe(t *testing.T, dest, src string) {
	fsrc, err := os.Open(src)
	if err != nil {
		t.Fatal("os.Open failed: ", err)
	}
	defer fsrc.Close()
	fdest, err := os.Create(dest)
	if err != nil {
		t.Fatal("os.Create failed: ", err)
	}
	defer fdest.Close()
	_, err = io.Copy(fdest, fsrc)
	if err != nil {
		t.Fatal("io.Copy failed: ", err)
	}
}

func installBat(t *testing.T, dest string) {
	f, err := os.Create(dest)
	if err != nil {
		t.Fatalf("failed to create batch file: %v", err)
	}
	defer f.Close()
	fmt.Fprintf(f, "@echo %s\n", dest)
}

func installProg(t *testing.T, dest, srcExe string) {
	err := os.MkdirAll(filepath.Dir(dest), 0700)
	if err != nil {
		t.Fatal("os.MkdirAll failed: ", err)
	}
	if strings.ToLower(filepath.Ext(dest)) == ".bat" {
		installBat(t, dest)
		return
	}
	installExe(t, dest, srcExe)
}

func runProg(t *testing.T, test lookPathTest, env []string, dir string, args ...string) (string, error) {
	cmd := Command(args[0], args[1:]...)
	cmd.Env = env
	cmd.Dir = dir
	args[0] = filepath.Base(args[0])
	cmdText := fmt.Sprintf("%q command", strings.Join(args, " "))
	out, err := cmd.CombinedOutput()
	if (err != nil) != test.fails {
		if test.fails {
			t.Fatalf("test=%+v: %s succeeded, but expected to fail", test, cmdText)
		}
		t.Fatalf("test=%+v: %s failed, but expected to succeed: %v - %v", test, cmdText, err, string(out))
	}
	if err != nil {
		return "", fmt.Errorf("test=%+v: %s failed: %v - %v", test, cmdText, err, string(out))
	}
	// normalise program output
	p := string(out)
	// trim terminating \r and \n that batch file outputs
	for len(p) > 0 && (p[len(p)-1] == '\n' || p[len(p)-1] == '\r') {
		p = p[:len(p)-1]
	}
	if !filepath.IsAbs(p) {
		return p, nil
	}
	if p[:len(dir)] != dir {
		t.Fatalf("test=%+v: %s output is wrong: %q must have %q prefix", test, cmdText, p, dir)
	}
	return p[len(dir)+1:], nil
}

func testLookPath(t *testing.T, test lookPathTest, tmpdir, lookpathExe, printpathExe string) {
	// Create files listed in test.files in tmp directory.
	for i := range test.files {
		installProg(t, filepath.Join(tmpdir, test.files[i]), printpathExe)
	}
	// Create environment with test.PATH and test.PATHEXT set.
	env := os.Environ()
	env = updateEnv(env, "PATH", test.PrefixPATH(tmpdir))
	env = updateEnv(env, "PATHEXT", test.PATHEXT)
	// Run "cmd.exe /c test.searchFor" with new environment and
	// work directory set. All candidates are copies of printpath.exe.
	// These will output their program paths when run.
	should, errCmd := runProg(t, test, env, tmpdir, "cmd", "/c", test.searchFor)
	// Run the lookpath program with new environment and work directory set.
	have, errLP := runProg(t, test, env, tmpdir, lookpathExe, test.searchFor)
	// Compare results.
	if errCmd == nil && errLP == nil {
		// both succeeded
		if should != have {
			//			t.Fatalf("test=%+v failed: expected to find %v, but found %v", test, should, have)
			t.Fatalf("test=%+v failed: expected to find %q, but found %q", test, should, have)
		}
		return
	}
	if errCmd != nil && errLP != nil {
		// both failed -> continue
		return
	}
	if errCmd != nil {
		t.Fatal(errCmd)
	}
	if errLP != nil {
		t.Fatal(errLP)
	}
}

func buildExe(t *testing.T, templ, dir, name string) string {
	srcname := name + ".go"
	f, err := os.Create(filepath.Join(dir, srcname))
	if err != nil {
		t.Fatalf("failed to create source: %v", err)
	}
	err = template.Must(template.New("template").Parse(templ)).Execute(f, nil)
	f.Close()
	if err != nil {
		t.Fatalf("failed to execute template: %v", err)
	}
	outname := name + ".exe"
	cmd := Command("go", "build", "-o", outname, srcname)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build executable: %v - %v", err, string(out))
	}
	return filepath.Join(dir, outname)
}

func TestLookPath(t *testing.T) {
	tmp, err := ioutil.TempDir("", "TestLookPath")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}
	defer os.RemoveAll(tmp)

	// Create a Go program that uses LookPath to find executable passed as command line parameter.
	lookpathExe := buildExe(t, lookpathSrc, tmp, "lookpath")

	// Create a Go program that prints its own path.
	printpathExe := buildExe(t, printpathSrc, tmp, "printpath")

	// Run all tests.
	for i, test := range lookPathTests {
		dir := filepath.Join(tmp, "d"+strconv.Itoa(i))
		err := os.Mkdir(dir, 0700)
		if err != nil {
			t.Fatal("Mkdir failed: ", err)
		}
		testLookPath(t, test, dir, lookpathExe, printpathExe)
	}
}

const lookpathSrc = `
package main

import (
	"fmt"
	"os"
	"os/exec"
)

func main() {
	p, err := exec.LookPath(os.Args[1])
	if err != nil {
		fmt.Printf("LookPath failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Print(p)
}
`

const printpathSrc = `
package main

import (
	"fmt"
	"os"
	"syscall"
	"unicode/utf16"
	"unsafe"
)

func getMyName() (string, error) {
	var sysproc = syscall.MustLoadDLL("kernel32.dll").MustFindProc("GetModuleFileNameW")
	b := make([]uint16, syscall.MAX_PATH)
	r, _, err := sysproc.Call(0, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)))
	n := uint32(r)
	if n == 0 {
		return "", err
	}
	return string(utf16.Decode(b[0:n])), nil
}

func main() {
	path, err := getMyName()
	if err != nil {
		fmt.Printf("getMyName failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Print(path)
}
`
