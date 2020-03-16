// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Use an external test to avoid os/exec -> internal/testenv -> os/exec
// circular dependency.

package exec_test

import (
	"fmt"
	"internal/testenv"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

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

type lookPathTest struct {
	rootDir   string
	PATH      string
	PATHEXT   string
	files     []string
	searchFor string
	fails     bool // test is expected to fail
}

func (test lookPathTest) runProg(t *testing.T, env []string, args ...string) (string, error) {
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Env = env
	cmd.Dir = test.rootDir
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
	if p[:len(test.rootDir)] != test.rootDir {
		t.Fatalf("test=%+v: %s output is wrong: %q must have %q prefix", test, cmdText, p, test.rootDir)
	}
	return p[len(test.rootDir)+1:], nil
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

func createEnv(dir, PATH, PATHEXT string) []string {
	env := os.Environ()
	env = updateEnv(env, "PATHEXT", PATHEXT)
	// Add dir in front of every directory in the PATH.
	dirs := filepath.SplitList(PATH)
	for i := range dirs {
		dirs[i] = filepath.Join(dir, dirs[i])
	}
	path := strings.Join(dirs, ";")
	env = updateEnv(env, "PATH", os.Getenv("SystemRoot")+"/System32;"+path)
	return env
}

// createFiles copies srcPath file into multiply files.
// It uses dir as prefix for all destination files.
func createFiles(t *testing.T, dir string, files []string, srcPath string) {
	for _, f := range files {
		installProg(t, filepath.Join(dir, f), srcPath)
	}
}

func (test lookPathTest) run(t *testing.T, tmpdir, printpathExe string) {
	test.rootDir = tmpdir
	createFiles(t, test.rootDir, test.files, printpathExe)
	env := createEnv(test.rootDir, test.PATH, test.PATHEXT)
	// Run "cmd.exe /c test.searchFor" with new environment and
	// work directory set. All candidates are copies of printpath.exe.
	// These will output their program paths when run.
	should, errCmd := test.runProg(t, env, "cmd", "/c", test.searchFor)
	// Run the lookpath program with new environment and work directory set.
	env = append(env, "GO_WANT_HELPER_PROCESS=1")
	have, errLP := test.runProg(t, env, os.Args[0], "-test.run=TestHelperProcess", "--", "lookpath", test.searchFor)
	// Compare results.
	if errCmd == nil && errLP == nil {
		// both succeeded
		if should != have {
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

func TestLookPath(t *testing.T) {
	tmp, err := ioutil.TempDir("", "TestLookPath")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}
	defer os.RemoveAll(tmp)

	printpathExe := buildPrintPathExe(t, tmp)

	// Run all tests.
	for i, test := range lookPathTests {
		dir := filepath.Join(tmp, "d"+strconv.Itoa(i))
		err := os.Mkdir(dir, 0700)
		if err != nil {
			t.Fatal("Mkdir failed: ", err)
		}
		test.run(t, dir, printpathExe)
	}
}

type commandTest struct {
	PATH  string
	files []string
	dir   string
	arg0  string
	want  string
	fails bool // test is expected to fail
}

func (test commandTest) isSuccess(rootDir, output string, err error) error {
	if err != nil {
		return fmt.Errorf("test=%+v: exec: %v %v", test, err, output)
	}
	path := output
	if path[:len(rootDir)] != rootDir {
		return fmt.Errorf("test=%+v: %q must have %q prefix", test, path, rootDir)
	}
	path = path[len(rootDir)+1:]
	if path != test.want {
		return fmt.Errorf("test=%+v: want %q, got %q", test, test.want, path)
	}
	return nil
}

func (test commandTest) runOne(rootDir string, env []string, dir, arg0 string) error {
	cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcess", "--", "exec", dir, arg0)
	cmd.Dir = rootDir
	cmd.Env = env
	output, err := cmd.CombinedOutput()
	err = test.isSuccess(rootDir, string(output), err)
	if (err != nil) != test.fails {
		if test.fails {
			return fmt.Errorf("test=%+v: succeeded, but expected to fail", test)
		}
		return err
	}
	return nil
}

func (test commandTest) run(t *testing.T, rootDir, printpathExe string) {
	createFiles(t, rootDir, test.files, printpathExe)
	PATHEXT := `.COM;.EXE;.BAT`
	env := createEnv(rootDir, test.PATH, PATHEXT)
	env = append(env, "GO_WANT_HELPER_PROCESS=1")
	err := test.runOne(rootDir, env, test.dir, test.arg0)
	if err != nil {
		t.Error(err)
	}
}

var commandTests = []commandTest{
	// testing commands with no slash, like `a.exe`
	{
		// should find a.exe in current directory
		files: []string{`a.exe`},
		arg0:  `a.exe`,
		want:  `a.exe`,
	},
	{
		// like above, but add PATH in attempt to break the test
		PATH:  `p2;p`,
		files: []string{`a.exe`, `p\a.exe`, `p2\a.exe`},
		arg0:  `a.exe`,
		want:  `a.exe`,
	},
	{
		// like above, but use "a" instead of "a.exe" for command
		PATH:  `p2;p`,
		files: []string{`a.exe`, `p\a.exe`, `p2\a.exe`},
		arg0:  `a`,
		want:  `a.exe`,
	},
	// testing commands with slash, like `.\a.exe`
	{
		// should find p\a.exe
		files: []string{`p\a.exe`},
		arg0:  `p\a.exe`,
		want:  `p\a.exe`,
	},
	{
		// like above, but adding `.` in front of executable should still be OK
		files: []string{`p\a.exe`},
		arg0:  `.\p\a.exe`,
		want:  `p\a.exe`,
	},
	{
		// like above, but with PATH added in attempt to break it
		PATH:  `p2`,
		files: []string{`p\a.exe`, `p2\a.exe`},
		arg0:  `p\a.exe`,
		want:  `p\a.exe`,
	},
	{
		// like above, but make sure .exe is tried even for commands with slash
		PATH:  `p2`,
		files: []string{`p\a.exe`, `p2\a.exe`},
		arg0:  `p\a`,
		want:  `p\a.exe`,
	},
	// tests commands, like `a.exe`, with c.Dir set
	{
		// should not find a.exe in p, because LookPath(`a.exe`) will fail
		files: []string{`p\a.exe`},
		dir:   `p`,
		arg0:  `a.exe`,
		want:  `p\a.exe`,
		fails: true,
	},
	{
		// LookPath(`a.exe`) will find `.\a.exe`, but prefixing that with
		// dir `p\a.exe` will refer to a non-existent file
		files: []string{`a.exe`, `p\not_important_file`},
		dir:   `p`,
		arg0:  `a.exe`,
		want:  `a.exe`,
		fails: true,
	},
	{
		// like above, but making test succeed by installing file
		// in referred destination (so LookPath(`a.exe`) will still
		// find `.\a.exe`, but we successfully execute `p\a.exe`)
		files: []string{`a.exe`, `p\a.exe`},
		dir:   `p`,
		arg0:  `a.exe`,
		want:  `p\a.exe`,
	},
	{
		// like above, but add PATH in attempt to break the test
		PATH:  `p2;p`,
		files: []string{`a.exe`, `p\a.exe`, `p2\a.exe`},
		dir:   `p`,
		arg0:  `a.exe`,
		want:  `p\a.exe`,
	},
	{
		// like above, but use "a" instead of "a.exe" for command
		PATH:  `p2;p`,
		files: []string{`a.exe`, `p\a.exe`, `p2\a.exe`},
		dir:   `p`,
		arg0:  `a`,
		want:  `p\a.exe`,
	},
	{
		// finds `a.exe` in the PATH regardless of dir set
		// because LookPath returns full path in that case
		PATH:  `p2;p`,
		files: []string{`p\a.exe`, `p2\a.exe`},
		dir:   `p`,
		arg0:  `a.exe`,
		want:  `p2\a.exe`,
	},
	// tests commands, like `.\a.exe`, with c.Dir set
	{
		// should use dir when command is path, like ".\a.exe"
		files: []string{`p\a.exe`},
		dir:   `p`,
		arg0:  `.\a.exe`,
		want:  `p\a.exe`,
	},
	{
		// like above, but with PATH added in attempt to break it
		PATH:  `p2`,
		files: []string{`p\a.exe`, `p2\a.exe`},
		dir:   `p`,
		arg0:  `.\a.exe`,
		want:  `p\a.exe`,
	},
	{
		// like above, but make sure .exe is tried even for commands with slash
		PATH:  `p2`,
		files: []string{`p\a.exe`, `p2\a.exe`},
		dir:   `p`,
		arg0:  `.\a`,
		want:  `p\a.exe`,
	},
}

func TestCommand(t *testing.T) {
	tmp, err := ioutil.TempDir("", "TestCommand")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}
	defer os.RemoveAll(tmp)

	printpathExe := buildPrintPathExe(t, tmp)

	// Run all tests.
	for i, test := range commandTests {
		dir := filepath.Join(tmp, "d"+strconv.Itoa(i))
		err := os.Mkdir(dir, 0700)
		if err != nil {
			t.Fatal("Mkdir failed: ", err)
		}
		test.run(t, dir, printpathExe)
	}
}

// buildPrintPathExe creates a Go program that prints its own path.
// dir is a temp directory where executable will be created.
// The function returns full path to the created program.
func buildPrintPathExe(t *testing.T, dir string) string {
	const name = "printpath"
	srcname := name + ".go"
	err := ioutil.WriteFile(filepath.Join(dir, srcname), []byte(printpathSrc), 0644)
	if err != nil {
		t.Fatalf("failed to create source: %v", err)
	}
	if err != nil {
		t.Fatalf("failed to execute template: %v", err)
	}
	outname := name + ".exe"
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", outname, srcname)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build executable: %v - %v", err, string(out))
	}
	return filepath.Join(dir, outname)
}

const printpathSrc = `
package main

import (
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
		os.Stderr.Write([]byte("getMyName failed: " + err.Error() + "\n"))
		os.Exit(1)
	}
	os.Stdout.Write([]byte(path))
}
`
