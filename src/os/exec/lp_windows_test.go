// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Use an external test to avoid os/exec -> internal/testenv -> os/exec
// circular dependency.

package exec_test

import (
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func init() {
	registerHelperCommand("printpath", cmdPrintPath)
}

func cmdPrintPath(args ...string) {
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Executable: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(exe)
}

// makePATH returns a PATH variable referring to the
// given directories relative to a root directory.
//
// The empty string results in an empty entry.
// Paths beginning with . are kept as relative entries.
func makePATH(root string, dirs []string) string {
	paths := make([]string, 0, len(dirs))
	for _, d := range dirs {
		switch {
		case d == "":
			paths = append(paths, "")
		case d == "." || (len(d) >= 2 && d[0] == '.' && os.IsPathSeparator(d[1])):
			paths = append(paths, filepath.Clean(d))
		default:
			paths = append(paths, filepath.Join(root, d))
		}
	}
	return strings.Join(paths, string(os.PathListSeparator))
}

// installProgs creates executable files (or symlinks to executable files) at
// multiple destination paths. It uses root as prefix for all destination files.
func installProgs(t *testing.T, root string, files []string) {
	for _, f := range files {
		dstPath := filepath.Join(root, f)

		dir := filepath.Dir(dstPath)
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}

		if os.IsPathSeparator(f[len(f)-1]) {
			continue // directory and PATH entry only.
		}
		if strings.EqualFold(filepath.Ext(f), ".bat") {
			installBat(t, dstPath)
		} else {
			installExe(t, dstPath)
		}
	}
}

// installExe installs a copy of the test executable
// at the given location, creating directories as needed.
//
// (We use a copy instead of just a symlink to ensure that os.Executable
// always reports an unambiguous path, regardless of how it is implemented.)
func installExe(t *testing.T, dstPath string) {
	src, err := os.Open(exePath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	dst, err := os.OpenFile(dstPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o777)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dst.Close(); err != nil {
			t.Fatal(err)
		}
	}()

	_, err = io.Copy(dst, src)
	if err != nil {
		t.Fatal(err)
	}
}

// installBat creates a batch file at dst that prints its own
// path when run.
func installBat(t *testing.T, dstPath string) {
	dst, err := os.OpenFile(dstPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o777)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dst.Close(); err != nil {
			t.Fatal(err)
		}
	}()

	if _, err := fmt.Fprintf(dst, "@echo %s\r\n", dstPath); err != nil {
		t.Fatal(err)
	}
}

type lookPathTest struct {
	name            string
	PATHEXT         string // empty to use default
	files           []string
	PATH            []string // if nil, use all parent directories from files
	searchFor       string
	want            string
	wantErr         error
	skipCmdExeCheck bool // if true, do not check want against the behavior of cmd.exe
}

var lookPathTests = []lookPathTest{
	{
		name:      "first match",
		files:     []string{`p1\a.exe`, `p2\a.exe`, `p2\a`},
		searchFor: `a`,
		want:      `p1\a.exe`,
	},
	{
		name:      "dirs with extensions",
		files:     []string{`p1.dir\a`, `p2.dir\a.exe`},
		searchFor: `a`,
		want:      `p2.dir\a.exe`,
	},
	{
		name:      "first with extension",
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `a.exe`,
		want:      `p1\a.exe`,
	},
	{
		name:      "specific name",
		files:     []string{`p1\a.exe`, `p2\b.exe`},
		searchFor: `b`,
		want:      `p2\b.exe`,
	},
	{
		name:      "no extension",
		files:     []string{`p1\b`, `p2\a`},
		searchFor: `a`,
		wantErr:   exec.ErrNotFound,
	},
	{
		name:      "directory, no extension",
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `p2\a`,
		want:      `p2\a.exe`,
	},
	{
		name:      "no match",
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `b`,
		wantErr:   exec.ErrNotFound,
	},
	{
		name:      "no match with dir",
		files:     []string{`p1\b.exe`, `p2\a.exe`},
		searchFor: `p2\b`,
		wantErr:   exec.ErrNotFound,
	},
	{
		name:      "extensionless file in CWD ignored",
		files:     []string{`a`, `p1\a.exe`, `p2\a.exe`},
		searchFor: `a`,
		want:      `p1\a.exe`,
	},
	{
		name:      "extensionless file in PATH ignored",
		files:     []string{`p1\a`, `p2\a.exe`},
		searchFor: `a`,
		want:      `p2\a.exe`,
	},
	{
		name:      "specific extension",
		files:     []string{`p1\a.exe`, `p2\a.bat`},
		searchFor: `a.bat`,
		want:      `p2\a.bat`,
	},
	{
		name:      "mismatched extension",
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `a.com`,
		wantErr:   exec.ErrNotFound,
	},
	{
		name:      "doubled extension",
		files:     []string{`p1\a.exe.exe`},
		searchFor: `a.exe`,
		want:      `p1\a.exe.exe`,
	},
	{
		name:      "extension not in PATHEXT",
		PATHEXT:   `.COM;.BAT`,
		files:     []string{`p1\a.exe`, `p2\a.exe`},
		searchFor: `a.exe`,
		want:      `p1\a.exe`,
	},
	{
		name:      "first allowed by PATHEXT",
		PATHEXT:   `.COM;.EXE`,
		files:     []string{`p1\a.bat`, `p2\a.exe`},
		searchFor: `a`,
		want:      `p2\a.exe`,
	},
	{
		name:      "first directory containing a PATHEXT match",
		PATHEXT:   `.COM;.EXE;.BAT`,
		files:     []string{`p1\a.bat`, `p2\a.exe`},
		searchFor: `a`,
		want:      `p1\a.bat`,
	},
	{
		name:      "first PATHEXT entry",
		PATHEXT:   `.COM;.EXE;.BAT`,
		files:     []string{`p1\a.bat`, `p1\a.exe`, `p2\a.bat`, `p2\a.exe`},
		searchFor: `a`,
		want:      `p1\a.exe`,
	},
	{
		name:      "ignore dir with PATHEXT extension",
		files:     []string{`a.exe\`},
		searchFor: `a`,
		wantErr:   exec.ErrNotFound,
	},
	{
		name:      "ignore empty PATH entry",
		files:     []string{`a.bat`, `p\a.bat`},
		PATH:      []string{`p`},
		searchFor: `a`,
		want:      `p\a.bat`,
		// If cmd.exe is too old it might not respect NoDefaultCurrentDirectoryInExePath,
		// so skip that check.
		skipCmdExeCheck: true,
	},
	{
		name:      "return ErrDot if found by a different absolute path",
		files:     []string{`p1\a.bat`, `p2\a.bat`},
		PATH:      []string{`.\p1`, `p2`},
		searchFor: `a`,
		want:      `p1\a.bat`,
		wantErr:   exec.ErrDot,
	},
	{
		name:      "suppress ErrDot if also found in absolute path",
		files:     []string{`p1\a.bat`, `p2\a.bat`},
		PATH:      []string{`.\p1`, `p1`, `p2`},
		searchFor: `a`,
		want:      `p1\a.bat`,
	},
}

func TestLookPathWindows(t *testing.T) {
	// Not parallel: uses Chdir and Setenv.

	// We are using the "printpath" command mode to test exec.Command here,
	// so we won't be calling helperCommand to resolve it.
	// That may cause it to appear to be unused.
	maySkipHelperCommand("printpath")

	// Before we begin, find the absolute path to cmd.exe.
	// In non-short mode, we will use it to check the ground truth
	// of the test's "want" field.
	cmdExe, err := exec.LookPath("cmd")
	if err != nil {
		t.Fatal(err)
	}

	for _, tt := range lookPathTests {
		t.Run(tt.name, func { t ->
			if tt.want == "" && tt.wantErr == nil {
				t.Fatalf("test must specify either want or wantErr")
			}

			root := t.TempDir()
			installProgs(t, root, tt.files)

			if tt.PATHEXT != "" {
				t.Setenv("PATHEXT", tt.PATHEXT)
				t.Logf("set PATHEXT=%s", tt.PATHEXT)
			}

			var pathVar string
			if tt.PATH == nil {
				paths := make([]string, 0, len(tt.files))
				for _, f := range tt.files {
					dir := filepath.Join(root, filepath.Dir(f))
					if !slices.Contains(paths, dir) {
						paths = append(paths, dir)
					}
				}
				pathVar = strings.Join(paths, string(os.PathListSeparator))
			} else {
				pathVar = makePATH(root, tt.PATH)
			}
			t.Setenv("PATH", pathVar)
			t.Logf("set PATH=%s", pathVar)

			chdir(t, root)

			if !testing.Short() && !(tt.skipCmdExeCheck || errors.Is(tt.wantErr, exec.ErrDot)) {
				// Check that cmd.exe, which is our source of ground truth,
				// agrees that our test case is correct.
				cmd := testenv.Command(t, cmdExe, "/c", tt.searchFor, "printpath")
				out, err := cmd.Output()
				if err == nil {
					gotAbs := strings.TrimSpace(string(out))
					wantAbs := ""
					if tt.want != "" {
						wantAbs = filepath.Join(root, tt.want)
					}
					if gotAbs != wantAbs {
						// cmd.exe disagrees. Probably the test case is wrong?
						t.Fatalf("%v\n\tresolved to %s\n\twant %s", cmd, gotAbs, wantAbs)
					}
				} else if tt.wantErr == nil {
					if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
						t.Fatalf("%v: %v\n%s", cmd, err, ee.Stderr)
					}
					t.Fatalf("%v: %v", cmd, err)
				}
			}

			got, err := exec.LookPath(tt.searchFor)
			if filepath.IsAbs(got) {
				got, err = filepath.Rel(root, got)
				if err != nil {
					t.Fatal(err)
				}
			}
			if got != tt.want {
				t.Errorf("LookPath(%#q) = %#q; want %#q", tt.searchFor, got, tt.want)
			}
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("LookPath(%#q): %v; want %v", tt.searchFor, err, tt.wantErr)
			}
		})
	}
}

type commandTest struct {
	name       string
	PATH       []string
	files      []string
	dir        string
	arg0       string
	want       string
	wantPath   string // the resolved c.Path, if different from want
	wantErrDot bool
	wantRunErr error
}

var commandTests = []commandTest{
	// testing commands with no slash, like `a.exe`
	{
		name:       "current directory",
		files:      []string{`a.exe`},
		PATH:       []string{"."},
		arg0:       `a.exe`,
		want:       `a.exe`,
		wantErrDot: true,
	},
	{
		name:       "with extra PATH",
		files:      []string{`a.exe`, `p\a.exe`, `p2\a.exe`},
		PATH:       []string{".", "p2", "p"},
		arg0:       `a.exe`,
		want:       `a.exe`,
		wantErrDot: true,
	},
	{
		name:       "with extra PATH and no extension",
		files:      []string{`a.exe`, `p\a.exe`, `p2\a.exe`},
		PATH:       []string{".", "p2", "p"},
		arg0:       `a`,
		want:       `a.exe`,
		wantErrDot: true,
	},
	// testing commands with slash, like `.\a.exe`
	{
		name:  "with dir",
		files: []string{`p\a.exe`},
		PATH:  []string{"."},
		arg0:  `p\a.exe`,
		want:  `p\a.exe`,
	},
	{
		name:  "with explicit dot",
		files: []string{`p\a.exe`},
		PATH:  []string{"."},
		arg0:  `.\p\a.exe`,
		want:  `p\a.exe`,
	},
	{
		name:  "with irrelevant PATH",
		files: []string{`p\a.exe`, `p2\a.exe`},
		PATH:  []string{".", "p2"},
		arg0:  `p\a.exe`,
		want:  `p\a.exe`,
	},
	{
		name:  "with slash and no extension",
		files: []string{`p\a.exe`, `p2\a.exe`},
		PATH:  []string{".", "p2"},
		arg0:  `p\a`,
		want:  `p\a.exe`,
	},
	// tests commands, like `a.exe`, with c.Dir set
	{
		// should not find a.exe in p, because LookPath(`a.exe`) will fail when
		// called by Command (before Dir is set), and that error is sticky.
		name:       "not found before Dir",
		files:      []string{`p\a.exe`},
		PATH:       []string{"."},
		dir:        `p`,
		arg0:       `a.exe`,
		want:       `p\a.exe`,
		wantRunErr: exec.ErrNotFound,
	},
	{
		// LookPath(`a.exe`) will resolve to `.\a.exe`, but prefixing that with
		// dir `p\a.exe` will refer to a non-existent file
		name:       "resolved before Dir",
		files:      []string{`a.exe`, `p\not_important_file`},
		PATH:       []string{"."},
		dir:        `p`,
		arg0:       `a.exe`,
		want:       `a.exe`,
		wantErrDot: true,
		wantRunErr: fs.ErrNotExist,
	},
	{
		// like above, but making test succeed by installing file
		// in referred destination (so LookPath(`a.exe`) will still
		// find `.\a.exe`, but we successfully execute `p\a.exe`)
		name:       "relative to Dir",
		files:      []string{`a.exe`, `p\a.exe`},
		PATH:       []string{"."},
		dir:        `p`,
		arg0:       `a.exe`,
		want:       `p\a.exe`,
		wantErrDot: true,
	},
	{
		// like above, but add PATH in attempt to break the test
		name:       "relative to Dir with extra PATH",
		files:      []string{`a.exe`, `p\a.exe`, `p2\a.exe`},
		PATH:       []string{".", "p2", "p"},
		dir:        `p`,
		arg0:       `a.exe`,
		want:       `p\a.exe`,
		wantErrDot: true,
	},
	{
		// like above, but use "a" instead of "a.exe" for command
		name:       "relative to Dir with extra PATH and no extension",
		files:      []string{`a.exe`, `p\a.exe`, `p2\a.exe`},
		PATH:       []string{".", "p2", "p"},
		dir:        `p`,
		arg0:       `a`,
		want:       `p\a.exe`,
		wantErrDot: true,
	},
	{
		// finds `a.exe` in the PATH regardless of Dir because Command resolves the
		// full path (using LookPath) before Dir is set.
		name:  "from PATH with no match in Dir",
		files: []string{`p\a.exe`, `p2\a.exe`},
		PATH:  []string{".", "p2", "p"},
		dir:   `p`,
		arg0:  `a.exe`,
		want:  `p2\a.exe`,
	},
	// tests commands, like `.\a.exe`, with c.Dir set
	{
		// should use dir when command is path, like ".\a.exe"
		name:  "relative to Dir with explicit dot",
		files: []string{`p\a.exe`},
		PATH:  []string{"."},
		dir:   `p`,
		arg0:  `.\a.exe`,
		want:  `p\a.exe`,
	},
	{
		// like above, but with PATH added in attempt to break it
		name:  "relative to Dir with dot and extra PATH",
		files: []string{`p\a.exe`, `p2\a.exe`},
		PATH:  []string{".", "p2"},
		dir:   `p`,
		arg0:  `.\a.exe`,
		want:  `p\a.exe`,
	},
	{
		// LookPath(".\a") will fail before Dir is set, and that error is sticky.
		name:  "relative to Dir with dot and extra PATH and no extension",
		files: []string{`p\a.exe`, `p2\a.exe`},
		PATH:  []string{".", "p2"},
		dir:   `p`,
		arg0:  `.\a`,
		want:  `p\a.exe`,
	},
	{
		// LookPath(".\a") will fail before Dir is set, and that error is sticky.
		name:  "relative to Dir with different extension",
		files: []string{`a.exe`, `p\a.bat`},
		PATH:  []string{"."},
		dir:   `p`,
		arg0:  `.\a`,
		want:  `p\a.bat`,
	},
}

func TestCommand(t *testing.T) {
	// Not parallel: uses Chdir and Setenv.

	// We are using the "printpath" command mode to test exec.Command here,
	// so we won't be calling helperCommand to resolve it.
	// That may cause it to appear to be unused.
	maySkipHelperCommand("printpath")

	for _, tt := range commandTests {
		t.Run(tt.name, func { t ->
			if tt.PATH == nil {
				t.Fatalf("test must specify PATH")
			}

			root := t.TempDir()
			installProgs(t, root, tt.files)

			pathVar := makePATH(root, tt.PATH)
			t.Setenv("PATH", pathVar)
			t.Logf("set PATH=%s", pathVar)

			chdir(t, root)

			cmd := exec.Command(tt.arg0, "printpath")
			cmd.Dir = filepath.Join(root, tt.dir)
			if tt.wantErrDot {
				if errors.Is(cmd.Err, exec.ErrDot) {
					cmd.Err = nil
				} else {
					t.Fatalf("cmd.Err = %v; want ErrDot", cmd.Err)
				}
			}

			out, err := cmd.Output()
			if err != nil {
				if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
					t.Logf("%v: %v\n%s", cmd, err, ee.Stderr)
				} else {
					t.Logf("%v: %v", cmd, err)
				}
				if !errors.Is(err, tt.wantRunErr) {
					t.Errorf("want %v", tt.wantRunErr)
				}
				return
			}

			got := strings.TrimSpace(string(out))
			if filepath.IsAbs(got) {
				got, err = filepath.Rel(root, got)
				if err != nil {
					t.Fatal(err)
				}
			}
			if got != tt.want {
				t.Errorf("\nran  %#q\nwant %#q", got, tt.want)
			}

			gotPath := cmd.Path
			wantPath := tt.wantPath
			if wantPath == "" {
				if strings.Contains(tt.arg0, `\`) {
					wantPath = tt.arg0
				} else if tt.wantErrDot {
					wantPath = strings.TrimPrefix(tt.want, tt.dir+`\`)
				} else {
					wantPath = filepath.Join(root, tt.want)
				}
			}
			if gotPath != wantPath {
				t.Errorf("\ncmd.Path = %#q\nwant       %#q", gotPath, wantPath)
			}
		})
	}
}

func TestAbsCommandWithDoubledExtension(t *testing.T) {
	t.Parallel()

	// We expect that ".com" is always included in PATHEXT, but it may also be
	// found in the import path of a Go package. If it is at the root of the
	// import path, the resulting executable may be named like "example.com.exe".
	//
	// Since "example.com" looks like a proper executable name, it is probably ok
	// for exec.Command to try to run it directly without re-resolving it.
	// However, exec.LookPath should try a little harder to figure it out.

	comPath := filepath.Join(t.TempDir(), "example.com")
	batPath := comPath + ".bat"
	installBat(t, batPath)

	cmd := exec.Command(comPath)
	out, err := cmd.CombinedOutput()
	t.Logf("%v: %v\n%s", cmd, err, out)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Command(%#q).Run: %v\nwant fs.ErrNotExist", comPath, err)
	}

	resolved, err := exec.LookPath(comPath)
	if err != nil || resolved != batPath {
		t.Fatalf("LookPath(%#q) = %v, %v; want %#q, <nil>", comPath, resolved, err, batPath)
	}
}
