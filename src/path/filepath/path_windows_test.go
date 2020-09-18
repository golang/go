// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath_test

import (
	"flag"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime/debug"
	"strings"
	"testing"
)

func TestWinSplitListTestsAreValid(t *testing.T) {
	comspec := os.Getenv("ComSpec")
	if comspec == "" {
		t.Fatal("%ComSpec% must be set")
	}

	for ti, tt := range winsplitlisttests {
		testWinSplitListTestIsValid(t, ti, tt, comspec)
	}
}

func testWinSplitListTestIsValid(t *testing.T, ti int, tt SplitListTest,
	comspec string) {

	const (
		cmdfile             = `printdir.cmd`
		perm    os.FileMode = 0700
	)

	tmp, err := ioutil.TempDir("", "testWinSplitListTestIsValid")
	if err != nil {
		t.Fatalf("TempDir failed: %v", err)
	}
	defer os.RemoveAll(tmp)

	for i, d := range tt.result {
		if d == "" {
			continue
		}
		if cd := filepath.Clean(d); filepath.VolumeName(cd) != "" ||
			cd[0] == '\\' || cd == ".." || (len(cd) >= 3 && cd[0:3] == `..\`) {
			t.Errorf("%d,%d: %#q refers outside working directory", ti, i, d)
			return
		}
		dd := filepath.Join(tmp, d)
		if _, err := os.Stat(dd); err == nil {
			t.Errorf("%d,%d: %#q already exists", ti, i, d)
			return
		}
		if err = os.MkdirAll(dd, perm); err != nil {
			t.Errorf("%d,%d: MkdirAll(%#q) failed: %v", ti, i, dd, err)
			return
		}
		fn, data := filepath.Join(dd, cmdfile), []byte("@echo "+d+"\r\n")
		if err = ioutil.WriteFile(fn, data, perm); err != nil {
			t.Errorf("%d,%d: WriteFile(%#q) failed: %v", ti, i, fn, err)
			return
		}
	}

	// on some systems, SystemRoot is required for cmd to work
	systemRoot := os.Getenv("SystemRoot")

	for i, d := range tt.result {
		if d == "" {
			continue
		}
		exp := []byte(d + "\r\n")
		cmd := &exec.Cmd{
			Path: comspec,
			Args: []string{`/c`, cmdfile},
			Env:  []string{`Path=` + systemRoot + "/System32;" + tt.list, `SystemRoot=` + systemRoot},
			Dir:  tmp,
		}
		out, err := cmd.CombinedOutput()
		switch {
		case err != nil:
			t.Errorf("%d,%d: execution error %v\n%q", ti, i, err, out)
			return
		case !reflect.DeepEqual(out, exp):
			t.Errorf("%d,%d: expected %#q, got %#q", ti, i, exp, out)
			return
		default:
			// unshadow cmdfile in next directory
			err = os.Remove(filepath.Join(tmp, d, cmdfile))
			if err != nil {
				t.Fatalf("Remove test command failed: %v", err)
			}
		}
	}
}

func TestWindowsEvalSymlinks(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmpDir, err := ioutil.TempDir("", "TestWindowsEvalSymlinks")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// /tmp may itself be a symlink! Avoid the confusion, although
	// it means trusting the thing we're testing.
	tmpDir, err = filepath.EvalSymlinks(tmpDir)
	if err != nil {
		t.Fatal(err)
	}

	if len(tmpDir) < 3 {
		t.Fatalf("tmpDir path %q is too short", tmpDir)
	}
	if tmpDir[1] != ':' {
		t.Fatalf("tmpDir path %q must have drive letter in it", tmpDir)
	}
	test := EvalSymlinksTest{"test/linkabswin", tmpDir[:3]}

	// Create the symlink farm using relative paths.
	testdirs := append(EvalSymlinksTestDirs, test)
	for _, d := range testdirs {
		var err error
		path := simpleJoin(tmpDir, d.path)
		if d.dest == "" {
			err = os.Mkdir(path, 0755)
		} else {
			err = os.Symlink(d.dest, path)
		}
		if err != nil {
			t.Fatal(err)
		}
	}

	path := simpleJoin(tmpDir, test.path)

	testEvalSymlinks(t, path, test.dest)

	testEvalSymlinksAfterChdir(t, path, ".", test.dest)

	testEvalSymlinksAfterChdir(t,
		path,
		filepath.VolumeName(tmpDir)+".",
		test.dest)

	testEvalSymlinksAfterChdir(t,
		simpleJoin(tmpDir, "test"),
		simpleJoin("..", test.path),
		test.dest)

	testEvalSymlinksAfterChdir(t, tmpDir, test.path, test.dest)
}

// TestEvalSymlinksCanonicalNames verify that EvalSymlinks
// returns "canonical" path names on windows.
func TestEvalSymlinksCanonicalNames(t *testing.T) {
	tmp, err := ioutil.TempDir("", "evalsymlinkcanonical")
	if err != nil {
		t.Fatal("creating temp dir:", err)
	}
	defer os.RemoveAll(tmp)

	// ioutil.TempDir might return "non-canonical" name.
	cTmpName, err := filepath.EvalSymlinks(tmp)
	if err != nil {
		t.Errorf("EvalSymlinks(%q) error: %v", tmp, err)
	}

	dirs := []string{
		"test",
		"test/dir",
		"testing_long_dir",
		"TEST2",
	}

	for _, d := range dirs {
		dir := filepath.Join(cTmpName, d)
		err := os.Mkdir(dir, 0755)
		if err != nil {
			t.Fatal(err)
		}
		cname, err := filepath.EvalSymlinks(dir)
		if err != nil {
			t.Errorf("EvalSymlinks(%q) error: %v", dir, err)
			continue
		}
		if dir != cname {
			t.Errorf("EvalSymlinks(%q) returns %q, but should return %q", dir, cname, dir)
			continue
		}
		// test non-canonical names
		test := strings.ToUpper(dir)
		p, err := filepath.EvalSymlinks(test)
		if err != nil {
			t.Errorf("EvalSymlinks(%q) error: %v", test, err)
			continue
		}
		if p != cname {
			t.Errorf("EvalSymlinks(%q) returns %q, but should return %q", test, p, cname)
			continue
		}
		// another test
		test = strings.ToLower(dir)
		p, err = filepath.EvalSymlinks(test)
		if err != nil {
			t.Errorf("EvalSymlinks(%q) error: %v", test, err)
			continue
		}
		if p != cname {
			t.Errorf("EvalSymlinks(%q) returns %q, but should return %q", test, p, cname)
			continue
		}
	}
}

// checkVolume8dot3Setting runs "fsutil 8dot3name query c:" command
// (where c: is vol parameter) to discover "8dot3 name creation state".
// The state is combination of 2 flags. The global flag controls if it
// is per volume or global setting:
//   0 - Enable 8dot3 name creation on all volumes on the system
//   1 - Disable 8dot3 name creation on all volumes on the system
//   2 - Set 8dot3 name creation on a per volume basis
//   3 - Disable 8dot3 name creation on all volumes except the system volume
// If global flag is set to 2, then per-volume flag needs to be examined:
//   0 - Enable 8dot3 name creation on this volume
//   1 - Disable 8dot3 name creation on this volume
// checkVolume8dot3Setting verifies that "8dot3 name creation" flags
// are set to 2 and 0, if enabled parameter is true, or 2 and 1, if enabled
// is false. Otherwise checkVolume8dot3Setting returns error.
func checkVolume8dot3Setting(vol string, enabled bool) error {
	// It appears, on some systems "fsutil 8dot3name query ..." command always
	// exits with error. Ignore exit code, and look at fsutil output instead.
	out, _ := exec.Command("fsutil", "8dot3name", "query", vol).CombinedOutput()
	// Check that system has "Volume level setting" set.
	expected := "The registry state of NtfsDisable8dot3NameCreation is 2, the default (Volume level setting)"
	if !strings.Contains(string(out), expected) {
		// Windows 10 version of fsutil has different output message.
		expectedWindow10 := "The registry state is: 2 (Per volume setting - the default)"
		if !strings.Contains(string(out), expectedWindow10) {
			return fmt.Errorf("fsutil output should contain %q, but is %q", expected, string(out))
		}
	}
	// Now check the volume setting.
	expected = "Based on the above two settings, 8dot3 name creation is %s on %s"
	if enabled {
		expected = fmt.Sprintf(expected, "enabled", vol)
	} else {
		expected = fmt.Sprintf(expected, "disabled", vol)
	}
	if !strings.Contains(string(out), expected) {
		return fmt.Errorf("unexpected fsutil output: %q", string(out))
	}
	return nil
}

func setVolume8dot3Setting(vol string, enabled bool) error {
	cmd := []string{"fsutil", "8dot3name", "set", vol}
	if enabled {
		cmd = append(cmd, "0")
	} else {
		cmd = append(cmd, "1")
	}
	// It appears, on some systems "fsutil 8dot3name set ..." command always
	// exits with error. Ignore exit code, and look at fsutil output instead.
	out, _ := exec.Command(cmd[0], cmd[1:]...).CombinedOutput()
	if string(out) != "\r\nSuccessfully set 8dot3name behavior.\r\n" {
		// Windows 10 version of fsutil has different output message.
		expectedWindow10 := "Successfully %s 8dot3name generation on %s\r\n"
		if enabled {
			expectedWindow10 = fmt.Sprintf(expectedWindow10, "enabled", vol)
		} else {
			expectedWindow10 = fmt.Sprintf(expectedWindow10, "disabled", vol)
		}
		if string(out) != expectedWindow10 {
			return fmt.Errorf("%v command failed: %q", cmd, string(out))
		}
	}
	return nil
}

var runFSModifyTests = flag.Bool("run_fs_modify_tests", false, "run tests which modify filesystem parameters")

// This test assumes registry state of NtfsDisable8dot3NameCreation is 2,
// the default (Volume level setting).
func TestEvalSymlinksCanonicalNamesWith8dot3Disabled(t *testing.T) {
	if !*runFSModifyTests {
		t.Skip("skipping test that modifies file system setting; enable with -run_fs_modify_tests")
	}
	tempVol := filepath.VolumeName(os.TempDir())
	if len(tempVol) != 2 {
		t.Fatalf("unexpected temp volume name %q", tempVol)
	}

	err := checkVolume8dot3Setting(tempVol, true)
	if err != nil {
		t.Fatal(err)
	}
	err = setVolume8dot3Setting(tempVol, false)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err := setVolume8dot3Setting(tempVol, true)
		if err != nil {
			t.Fatal(err)
		}
		err = checkVolume8dot3Setting(tempVol, true)
		if err != nil {
			t.Fatal(err)
		}
	}()
	err = checkVolume8dot3Setting(tempVol, false)
	if err != nil {
		t.Fatal(err)
	}
	TestEvalSymlinksCanonicalNames(t)
}

func TestToNorm(t *testing.T) {
	stubBase := func(path string) (string, error) {
		vol := filepath.VolumeName(path)
		path = path[len(vol):]

		if strings.Contains(path, "/") {
			return "", fmt.Errorf("invalid path is given to base: %s", vol+path)
		}

		if path == "" || path == "." || path == `\` {
			return "", fmt.Errorf("invalid path is given to base: %s", vol+path)
		}

		i := strings.LastIndexByte(path, filepath.Separator)
		if i == len(path)-1 { // trailing '\' is invalid
			return "", fmt.Errorf("invalid path is given to base: %s", vol+path)
		}
		if i == -1 {
			return strings.ToUpper(path), nil
		}

		return strings.ToUpper(path[i+1:]), nil
	}

	// On this test, toNorm should be same as string.ToUpper(filepath.Clean(path)) except empty string.
	tests := []struct {
		arg  string
		want string
	}{
		{"", ""},
		{".", "."},
		{"./foo/bar", `FOO\BAR`},
		{"/", `\`},
		{"/foo/bar", `\FOO\BAR`},
		{"/foo/bar/baz/qux", `\FOO\BAR\BAZ\QUX`},
		{"foo/bar", `FOO\BAR`},
		{"C:/foo/bar", `C:\FOO\BAR`},
		{"C:foo/bar", `C:FOO\BAR`},
		{"c:/foo/bar", `C:\FOO\BAR`},
		{"C:/foo/bar", `C:\FOO\BAR`},
		{"C:/foo/bar/", `C:\FOO\BAR`},
		{`C:\foo\bar`, `C:\FOO\BAR`},
		{`C:\foo/bar\`, `C:\FOO\BAR`},
		{"C:/ふー/バー", `C:\ふー\バー`},
	}

	for _, test := range tests {
		got, err := filepath.ToNorm(test.arg, stubBase)
		if err != nil {
			t.Errorf("toNorm(%s) failed: %v\n", test.arg, err)
		} else if got != test.want {
			t.Errorf("toNorm(%s) returns %s, but %s expected\n", test.arg, got, test.want)
		}
	}

	testPath := `{{tmp}}\test\foo\bar`

	testsDir := []struct {
		wd   string
		arg  string
		want string
	}{
		// test absolute paths
		{".", `{{tmp}}\test\foo\bar`, `{{tmp}}\test\foo\bar`},
		{".", `{{tmp}}\.\test/foo\bar`, `{{tmp}}\test\foo\bar`},
		{".", `{{tmp}}\test\..\test\foo\bar`, `{{tmp}}\test\foo\bar`},
		{".", `{{tmp}}\TEST\FOO\BAR`, `{{tmp}}\test\foo\bar`},

		// test relative paths begin with drive letter
		{`{{tmp}}\test`, `{{tmpvol}}.`, `{{tmpvol}}.`},
		{`{{tmp}}\test`, `{{tmpvol}}..`, `{{tmpvol}}..`},
		{`{{tmp}}\test`, `{{tmpvol}}foo\bar`, `{{tmpvol}}foo\bar`},
		{`{{tmp}}\test`, `{{tmpvol}}.\foo\bar`, `{{tmpvol}}foo\bar`},
		{`{{tmp}}\test`, `{{tmpvol}}foo\..\foo\bar`, `{{tmpvol}}foo\bar`},
		{`{{tmp}}\test`, `{{tmpvol}}FOO\BAR`, `{{tmpvol}}foo\bar`},

		// test relative paths begin with '\'
		{"{{tmp}}", `{{tmpnovol}}\test\foo\bar`, `{{tmpnovol}}\test\foo\bar`},
		{"{{tmp}}", `{{tmpnovol}}\.\test\foo\bar`, `{{tmpnovol}}\test\foo\bar`},
		{"{{tmp}}", `{{tmpnovol}}\test\..\test\foo\bar`, `{{tmpnovol}}\test\foo\bar`},
		{"{{tmp}}", `{{tmpnovol}}\TEST\FOO\BAR`, `{{tmpnovol}}\test\foo\bar`},

		// test relative paths begin without '\'
		{`{{tmp}}\test`, ".", `.`},
		{`{{tmp}}\test`, "..", `..`},
		{`{{tmp}}\test`, `foo\bar`, `foo\bar`},
		{`{{tmp}}\test`, `.\foo\bar`, `foo\bar`},
		{`{{tmp}}\test`, `foo\..\foo\bar`, `foo\bar`},
		{`{{tmp}}\test`, `FOO\BAR`, `foo\bar`},
	}

	tmp, err := ioutil.TempDir("", "testToNorm")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err := os.RemoveAll(tmp)
		if err != nil {
			t.Fatal(err)
		}
	}()

	// ioutil.TempDir might return "non-canonical" name.
	ctmp, err := filepath.EvalSymlinks(tmp)
	if err != nil {
		t.Fatal(err)
	}

	err = os.MkdirAll(strings.ReplaceAll(testPath, "{{tmp}}", ctmp), 0777)
	if err != nil {
		t.Fatal(err)
	}

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err := os.Chdir(cwd)
		if err != nil {
			t.Fatal(err)
		}
	}()

	tmpVol := filepath.VolumeName(ctmp)
	if len(tmpVol) != 2 {
		t.Fatalf("unexpected temp volume name %q", tmpVol)
	}

	tmpNoVol := ctmp[len(tmpVol):]

	replacer := strings.NewReplacer("{{tmp}}", ctmp, "{{tmpvol}}", tmpVol, "{{tmpnovol}}", tmpNoVol)

	for _, test := range testsDir {
		wd := replacer.Replace(test.wd)
		arg := replacer.Replace(test.arg)
		want := replacer.Replace(test.want)

		if test.wd == "." {
			err := os.Chdir(cwd)
			if err != nil {
				t.Error(err)

				continue
			}
		} else {
			err := os.Chdir(wd)
			if err != nil {
				t.Error(err)

				continue
			}
		}

		got, err := filepath.ToNorm(arg, filepath.NormBase)
		if err != nil {
			t.Errorf("toNorm(%s) failed: %v (wd=%s)\n", arg, err, wd)
		} else if got != want {
			t.Errorf("toNorm(%s) returns %s, but %s expected (wd=%s)\n", arg, got, want, wd)
		}
	}
}

func TestUNC(t *testing.T) {
	// Test that this doesn't go into an infinite recursion.
	// See golang.org/issue/15879.
	defer debug.SetMaxStack(debug.SetMaxStack(1e6))
	filepath.Glob(`\\?\c:\*`)
}

func testWalkMklink(t *testing.T, linktype string) {
	output, _ := exec.Command("cmd", "/c", "mklink", "/?").Output()
	if !strings.Contains(string(output), fmt.Sprintf(" /%s ", linktype)) {
		t.Skipf(`skipping test; mklink does not supports /%s parameter`, linktype)
	}
	testWalkSymlink(t, func(target, link string) error {
		output, err := exec.Command("cmd", "/c", "mklink", "/"+linktype, link, target).CombinedOutput()
		if err != nil {
			return fmt.Errorf(`"mklink /%s %v %v" command failed: %v\n%v`, linktype, link, target, err, string(output))
		}
		return nil
	})
}

func TestWalkDirectoryJunction(t *testing.T) {
	testenv.MustHaveSymlink(t)
	testWalkMklink(t, "J")
}

func TestWalkDirectorySymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)
	testWalkMklink(t, "D")
}

func TestNTNamespaceSymlink(t *testing.T) {
	output, _ := exec.Command("cmd", "/c", "mklink", "/?").Output()
	if !strings.Contains(string(output), " /J ") {
		t.Skip("skipping test because mklink command does not support junctions")
	}

	tmpdir, err := ioutil.TempDir("", "TestNTNamespaceSymlink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	// Make sure tmpdir is not a symlink, otherwise tests will fail.
	tmpdir, err = filepath.EvalSymlinks(tmpdir)
	if err != nil {
		t.Fatal(err)
	}

	vol := filepath.VolumeName(tmpdir)
	output, err = exec.Command("cmd", "/c", "mountvol", vol, "/L").CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run mountvol %v /L: %v %q", vol, err, output)
	}
	target := strings.Trim(string(output), " \n\r")

	dirlink := filepath.Join(tmpdir, "dirlink")
	output, err = exec.Command("cmd", "/c", "mklink", "/J", dirlink, target).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run mklink %v %v: %v %q", dirlink, target, err, output)
	}

	got, err := filepath.EvalSymlinks(dirlink)
	if err != nil {
		t.Fatal(err)
	}
	if want := vol + `\`; got != want {
		t.Errorf(`EvalSymlinks(%q): got %q, want %q`, dirlink, got, want)
	}

	// Make sure we have sufficient privilege to run mklink command.
	testenv.MustHaveSymlink(t)

	file := filepath.Join(tmpdir, "file")
	err = ioutil.WriteFile(file, []byte(""), 0666)
	if err != nil {
		t.Fatal(err)
	}

	target += file[len(filepath.VolumeName(file)):]

	filelink := filepath.Join(tmpdir, "filelink")
	output, err = exec.Command("cmd", "/c", "mklink", filelink, target).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run mklink %v %v: %v %q", filelink, target, err, output)
	}

	got, err = filepath.EvalSymlinks(filelink)
	if err != nil {
		t.Fatal(err)
	}
	if want := file; got != want {
		t.Errorf(`EvalSymlinks(%q): got %q, want %q`, filelink, got, want)
	}
}
