// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"bytes"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"io/fs"
	"net"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

// testMaybeRooted calls f in two subtests,
// one with a Root and one with a nil r.
func testMaybeRooted(t *testing.T, f func(t *testing.T, r *os.Root)) {
	t.Run("NoRoot", func(t *testing.T) {
		t.Chdir(t.TempDir())
		f(t, nil)
	})
	t.Run("InRoot", func(t *testing.T) {
		t.Chdir(t.TempDir())
		r, err := os.OpenRoot(".")
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		f(t, r)
	})
}

// makefs creates a test filesystem layout and returns the path to its root.
//
// Each entry in the slice is a file, directory, or symbolic link to create:
//
//   - "d/": directory d
//   - "f": file f with contents f
//   - "a => b": symlink a with target b
//
// The directory containing the filesystem is always named ROOT.
// $ABS is replaced with the absolute path of the directory containing the filesystem.
//
// Parent directories are automatically created as needed.
//
// makefs calls t.Skip if the layout contains features not supported by the current GOOS.
func makefs(t *testing.T, fs []string) string {
	root := path.Join(t.TempDir(), "ROOT")
	if err := os.Mkdir(root, 0o777); err != nil {
		t.Fatal(err)
	}
	for _, ent := range fs {
		ent = strings.ReplaceAll(ent, "$ABS", root)
		base, link, isLink := strings.Cut(ent, " => ")
		if isLink {
			if runtime.GOOS == "wasip1" && path.IsAbs(link) {
				t.Skip("absolute link targets not supported on " + runtime.GOOS)
			}
			if runtime.GOOS == "plan9" {
				t.Skip("symlinks not supported on " + runtime.GOOS)
			}
			ent = base
		}
		if err := os.MkdirAll(path.Join(root, path.Dir(base)), 0o777); err != nil {
			t.Fatal(err)
		}
		if isLink {
			if err := os.Symlink(link, path.Join(root, base)); err != nil {
				t.Fatal(err)
			}
		} else if strings.HasSuffix(ent, "/") {
			if err := os.MkdirAll(path.Join(root, ent), 0o777); err != nil {
				t.Fatal(err)
			}
		} else {
			if err := os.WriteFile(path.Join(root, ent), []byte(ent), 0o666); err != nil {
				t.Fatal(err)
			}
		}
	}
	return root
}

// A rootTest is a test case for os.Root.
type rootTest struct {
	name string

	// fs is the test filesystem layout. See makefs above.
	fs []string

	// open is the filename to access in the test.
	open string

	// target is the filename that we expect to be accessed, after resolving all symlinks.
	// For test cases where the operation fails due to an escaping path such as ../ROOT/x,
	// the target is the filename that should not have been opened.
	target string

	// ltarget is the filename that we expect to accessed, after resolving all symlinks
	// except the last one. This is the file we expect to be removed by Remove or statted
	// by Lstat.
	//
	// If the last path component in open is not a symlink, ltarget should be "".
	ltarget string

	// wantError is true if accessing the file should fail.
	wantError bool

	// alwaysFails is true if the open operation is expected to fail
	// even when using non-openat operations.
	//
	// This lets us check that tests that are expected to fail because (for example)
	// a path escapes the directory root will succeed when the escaping checks are not
	// performed.
	alwaysFails bool
}

// run sets up the test filesystem layout, os.OpenDirs the root, and calls f.
func (test *rootTest) run(t *testing.T, f func(t *testing.T, target string, d *os.Root)) {
	t.Run(test.name, func(t *testing.T) {
		root := makefs(t, test.fs)
		d, err := os.OpenRoot(root)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		// The target is a file that will be accessed,
		// or a file that should not be accessed
		// (because doing so escapes the root).
		target := test.target
		if test.target != "" {
			target = filepath.Join(root, test.target)
		}
		f(t, target, d)
	})
}

// errEndsTest checks the error result of a test,
// verifying that it succeeded or failed as expected.
//
// It returns true if the test is done due to encountering an expected error.
// false if the test should continue.
func errEndsTest(t *testing.T, err error, wantError bool, format string, args ...any) bool {
	t.Helper()
	if wantError {
		if err == nil {
			op := fmt.Sprintf(format, args...)
			t.Fatalf("%v = nil; want error", op)
		}
		return true
	} else {
		if err != nil {
			op := fmt.Sprintf(format, args...)
			t.Fatalf("%v = %v; want success", op, err)
		}
		return false
	}
}

var rootTestCases = []rootTest{{
	name:   "plain path",
	fs:     []string{},
	open:   "target",
	target: "target",
}, {
	name: "path in directory",
	fs: []string{
		"a/b/c/",
	},
	open:   "a/b/c/target",
	target: "a/b/c/target",
}, {
	name: "symlink",
	fs: []string{
		"link => target",
	},
	open:    "link",
	target:  "target",
	ltarget: "link",
}, {
	name: "symlink chain",
	fs: []string{
		"link => a/b/c/target",
		"a/b => e",
		"a/e => ../f",
		"f => g/h/i",
		"g/h/i => ..",
		"g/c/",
	},
	open:    "link",
	target:  "g/c/target",
	ltarget: "link",
}, {
	name: "path with dot",
	fs: []string{
		"a/b/",
	},
	open:   "./a/./b/./target",
	target: "a/b/target",
}, {
	name: "path with dotdot",
	fs: []string{
		"a/b/",
	},
	open:   "a/../a/b/../../a/b/../b/target",
	target: "a/b/target",
}, {
	name: "dotdot no symlink",
	fs: []string{
		"a/",
	},
	open:   "a/../target",
	target: "target",
}, {
	name: "dotdot after symlink",
	fs: []string{
		"a => b/c",
		"b/c/",
	},
	open: "a/../target",
	target: func() string {
		if runtime.GOOS == "windows" {
			// On Windows, the path is cleaned before symlink resolution.
			return "target"
		}
		return "b/target"
	}(),
}, {
	name: "dotdot before symlink",
	fs: []string{
		"a => b/c",
		"b/c/",
	},
	open:   "b/../a/target",
	target: "b/c/target",
}, {
	name: "symlink ends in dot",
	fs: []string{
		"a => b/.",
		"b/",
	},
	open:   "a/target",
	target: "b/target",
}, {
	name:        "directory does not exist",
	fs:          []string{},
	open:        "a/file",
	wantError:   true,
	alwaysFails: true,
}, {
	name:        "empty path",
	fs:          []string{},
	open:        "",
	wantError:   true,
	alwaysFails: true,
}, {
	name: "symlink cycle",
	fs: []string{
		"a => a",
	},
	open:        "a",
	ltarget:     "a",
	wantError:   true,
	alwaysFails: true,
}, {
	name:      "path escapes",
	fs:        []string{},
	open:      "../ROOT/target",
	target:    "target",
	wantError: true,
}, {
	name: "long path escapes",
	fs: []string{
		"a/",
	},
	open:      "a/../../ROOT/target",
	target:    "target",
	wantError: true,
}, {
	name: "absolute symlink",
	fs: []string{
		"link => $ABS/target",
	},
	open:      "link",
	ltarget:   "link",
	target:    "target",
	wantError: true,
}, {
	name: "relative symlink",
	fs: []string{
		"link => ../ROOT/target",
	},
	open:      "link",
	target:    "target",
	ltarget:   "link",
	wantError: true,
}, {
	name: "symlink chain escapes",
	fs: []string{
		"link => a/b/c/target",
		"a/b => e",
		"a/e => ../../ROOT",
		"c/",
	},
	open:      "link",
	target:    "c/target",
	ltarget:   "link",
	wantError: true,
}}

func TestRootOpen_File(t *testing.T) {
	want := []byte("target")
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			if target != "" {
				if err := os.WriteFile(target, want, 0o666); err != nil {
					t.Fatal(err)
				}
			}
			f, err := root.Open(test.open)
			if errEndsTest(t, err, test.wantError, "root.Open(%q)", test.open) {
				return
			}
			defer f.Close()
			got, err := io.ReadAll(f)
			if err != nil || !bytes.Equal(got, want) {
				t.Errorf(`Dir.Open(%q): read content %q, %v; want %q`, test.open, string(got), err, string(want))
			}
		})
	}
}

func TestRootOpen_Directory(t *testing.T) {
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			if target != "" {
				if err := os.Mkdir(target, 0o777); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(target+"/found", nil, 0o666); err != nil {
					t.Fatal(err)
				}
			}
			f, err := root.Open(test.open)
			if errEndsTest(t, err, test.wantError, "root.Open(%q)", test.open) {
				return
			}
			defer f.Close()
			got, err := f.Readdirnames(-1)
			if err != nil {
				t.Errorf(`Dir.Open(%q).Readdirnames: %v`, test.open, err)
			}
			if want := []string{"found"}; !slices.Equal(got, want) {
				t.Errorf(`Dir.Open(%q).Readdirnames: %q, want %q`, test.open, got, want)
			}
		})
	}
}

func TestRootCreate(t *testing.T) {
	want := []byte("target")
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			f, err := root.Create(test.open)
			if errEndsTest(t, err, test.wantError, "root.Create(%q)", test.open) {
				return
			}
			if _, err := f.Write(want); err != nil {
				t.Fatal(err)
			}
			f.Close()
			got, err := os.ReadFile(target)
			if err != nil {
				t.Fatalf(`reading file created with root.Create(%q): %v`, test.open, err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf(`reading file created with root.Create(%q): got %q; want %q`, test.open, got, want)
			}
		})
	}
}

func TestRootChmod(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("Chmod not supported on " + runtime.GOOS)
	}
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			if target != "" {
				// Create a file with no read/write permissions,
				// to ensure we can use Chmod on an inaccessible file.
				if err := os.WriteFile(target, nil, 0o000); err != nil {
					t.Fatal(err)
				}
			}
			if runtime.GOOS == "windows" {
				// On Windows, Chmod("symlink") affects the link, not its target.
				// See issue 71492.
				fi, err := root.Lstat(test.open)
				if err == nil && !fi.Mode().IsRegular() {
					t.Skip("https://go.dev/issue/71492")
				}
			}
			want := os.FileMode(0o666)
			err := root.Chmod(test.open, want)
			if errEndsTest(t, err, test.wantError, "root.Chmod(%q)", test.open) {
				return
			}
			st, err := os.Stat(target)
			if err != nil {
				t.Fatalf("os.Stat(%q) = %v", target, err)
			}
			if got := st.Mode(); got != want {
				t.Errorf("after root.Chmod(%q, %v): file mode = %v, want %v", test.open, want, got, want)
			}
		})
	}
}

func TestRootChtimes(t *testing.T) {
	// Don't check atimes if the fs is mounted noatime,
	// or on Plan 9 which does not permit changing atimes to arbitrary values.
	checkAtimes := !hasNoatime() && runtime.GOOS != "plan9"
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			if target != "" {
				if err := os.WriteFile(target, nil, 0o666); err != nil {
					t.Fatal(err)
				}
			}
			for _, times := range []struct {
				atime, mtime time.Time
			}{{
				atime: time.Now().Add(-1 * time.Minute),
				mtime: time.Now().Add(-1 * time.Minute),
			}, {
				atime: time.Now().Add(1 * time.Minute),
				mtime: time.Now().Add(1 * time.Minute),
			}, {
				atime: time.Time{},
				mtime: time.Now(),
			}, {
				atime: time.Now(),
				mtime: time.Time{},
			}} {
				switch runtime.GOOS {
				case "js", "plan9":
					times.atime = times.atime.Truncate(1 * time.Second)
					times.mtime = times.mtime.Truncate(1 * time.Second)
				case "illumos":
					times.atime = times.atime.Truncate(1 * time.Microsecond)
					times.mtime = times.mtime.Truncate(1 * time.Microsecond)
				}

				err := root.Chtimes(test.open, times.atime, times.mtime)
				if errEndsTest(t, err, test.wantError, "root.Chtimes(%q)", test.open) {
					return
				}
				st, err := os.Stat(target)
				if err != nil {
					t.Fatalf("os.Stat(%q) = %v", target, err)
				}
				if got := st.ModTime(); !times.mtime.IsZero() && !got.Equal(times.mtime) {
					t.Errorf("after root.Chtimes(%q, %v, %v): got mtime=%v, want %v", test.open, times.atime, times.mtime, got, times.mtime)
				}
				if checkAtimes {
					if got := os.Atime(st); !times.atime.IsZero() && !got.Equal(times.atime) {
						t.Errorf("after root.Chtimes(%q, %v, %v): got atime=%v, want %v", test.open, times.atime, times.mtime, got, times.atime)
					}
				}
			}
		})
	}
}

func TestRootMkdir(t *testing.T) {
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			wantError := test.wantError
			if !wantError {
				fi, err := os.Lstat(filepath.Join(root.Name(), test.open))
				if err == nil && fi.Mode().Type() == fs.ModeSymlink {
					// This case is trying to mkdir("some symlink"),
					// which is an error.
					wantError = true
				}
			}

			err := root.Mkdir(test.open, 0o777)
			if errEndsTest(t, err, wantError, "root.Create(%q)", test.open) {
				return
			}
			fi, err := os.Lstat(target)
			if err != nil {
				t.Fatalf(`stat file created with Root.Mkdir(%q): %v`, test.open, err)
			}
			if !fi.IsDir() {
				t.Fatalf(`stat file created with Root.Mkdir(%q): not a directory`, test.open)
			}
			if mode := fi.Mode(); mode&0o777 == 0 {
				// Issue #73559: We're not going to worry about the exact
				// mode bits (which will have been modified by umask),
				// but there should be mode bits.
				t.Fatalf(`stat file created with Root.Mkdir(%q): mode=%v, want non-zero`, test.open, mode)
			}
		})
	}
}

func TestRootOpenRoot(t *testing.T) {
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			if target != "" {
				if err := os.Mkdir(target, 0o777); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(target+"/f", nil, 0o666); err != nil {
					t.Fatal(err)
				}
			}
			rr, err := root.OpenRoot(test.open)
			if errEndsTest(t, err, test.wantError, "root.OpenRoot(%q)", test.open) {
				return
			}
			defer rr.Close()
			f, err := rr.Open("f")
			if err != nil {
				t.Fatalf(`root.OpenRoot(%q).Open("f") = %v`, test.open, err)
			}
			f.Close()
		})
	}
}

func TestRootRemoveFile(t *testing.T) {
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			wantError := test.wantError
			if test.ltarget != "" {
				// Remove doesn't follow symlinks in the final path component,
				// so it will successfully remove ltarget.
				wantError = false
				target = filepath.Join(root.Name(), test.ltarget)
			} else if target != "" {
				if err := os.WriteFile(target, nil, 0o666); err != nil {
					t.Fatal(err)
				}
			}

			err := root.Remove(test.open)
			if errEndsTest(t, err, wantError, "root.Remove(%q)", test.open) {
				return
			}
			_, err = os.Lstat(target)
			if !errors.Is(err, os.ErrNotExist) {
				t.Fatalf(`stat file removed with Root.Remove(%q): %v, want ErrNotExist`, test.open, err)
			}
		})
	}
}

func TestRootRemoveDirectory(t *testing.T) {
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			wantError := test.wantError
			if test.ltarget != "" {
				// Remove doesn't follow symlinks in the final path component,
				// so it will successfully remove ltarget.
				wantError = false
				target = filepath.Join(root.Name(), test.ltarget)
			} else if target != "" {
				if err := os.Mkdir(target, 0o777); err != nil {
					t.Fatal(err)
				}
			}

			err := root.Remove(test.open)
			if errEndsTest(t, err, wantError, "root.Remove(%q)", test.open) {
				return
			}
			_, err = os.Lstat(target)
			if !errors.Is(err, os.ErrNotExist) {
				t.Fatalf(`stat file removed with Root.Remove(%q): %v, want ErrNotExist`, test.open, err)
			}
		})
	}
}

func TestRootOpenFileAsRoot(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	if err := os.WriteFile(target, nil, 0o666); err != nil {
		t.Fatal(err)
	}
	r, err := os.OpenRoot(target)
	if err == nil {
		r.Close()
		t.Fatal("os.OpenRoot(file) succeeded; want failure")
	}
	r, err = os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	rr, err := r.OpenRoot("target")
	if err == nil {
		rr.Close()
		t.Fatal("Root.OpenRoot(file) succeeded; want failure")
	}
}

func TestRootStat(t *testing.T) {
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			const content = "content"
			if target != "" {
				if err := os.WriteFile(target, []byte(content), 0o666); err != nil {
					t.Fatal(err)
				}
			}

			fi, err := root.Stat(test.open)
			if errEndsTest(t, err, test.wantError, "root.Stat(%q)", test.open) {
				return
			}
			if got, want := fi.Name(), filepath.Base(test.open); got != want {
				t.Errorf("root.Stat(%q).Name() = %q, want %q", test.open, got, want)
			}
			if got, want := fi.Size(), int64(len(content)); got != want {
				t.Errorf("root.Stat(%q).Size() = %v, want %v", test.open, got, want)
			}
		})
	}
}

func TestRootLstat(t *testing.T) {
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			const content = "content"
			wantError := test.wantError
			if test.ltarget != "" {
				// Lstat will stat the final link, rather than following it.
				wantError = false
			} else if target != "" {
				if err := os.WriteFile(target, []byte(content), 0o666); err != nil {
					t.Fatal(err)
				}
			}

			fi, err := root.Lstat(test.open)
			if errEndsTest(t, err, wantError, "root.Stat(%q)", test.open) {
				return
			}
			if got, want := fi.Name(), filepath.Base(test.open); got != want {
				t.Errorf("root.Stat(%q).Name() = %q, want %q", test.open, got, want)
			}
			if test.ltarget == "" {
				if got := fi.Mode(); got&os.ModeSymlink != 0 {
					t.Errorf("root.Stat(%q).Mode() = %v, want non-symlink", test.open, got)
				}
				if got, want := fi.Size(), int64(len(content)); got != want {
					t.Errorf("root.Stat(%q).Size() = %v, want %v", test.open, got, want)
				}
			} else {
				if got := fi.Mode(); got&os.ModeSymlink == 0 {
					t.Errorf("root.Stat(%q).Mode() = %v, want symlink", test.open, got)
				}
			}
		})
	}
}

func TestRootReadlink(t *testing.T) {
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			const content = "content"
			wantError := test.wantError
			if test.ltarget != "" {
				// Readlink will read the final link, rather than following it.
				wantError = false
			} else {
				// Readlink fails on non-link targets.
				wantError = true
			}

			got, err := root.Readlink(test.open)
			if errEndsTest(t, err, wantError, "root.Readlink(%q)", test.open) {
				return
			}

			want, err := os.Readlink(filepath.Join(root.Name(), test.ltarget))
			if err != nil {
				t.Fatalf("os.Readlink(%q) = %v, want success", test.ltarget, err)
			}
			if got != want {
				t.Errorf("root.Readlink(%q) = %q, want %q", test.open, got, want)
			}
		})
	}
}

// TestRootRenameFrom tests renaming the test case target to a known-good path.
func TestRootRenameFrom(t *testing.T) {
	testRootMoveFrom(t, true)
}

// TestRootRenameFrom tests linking the test case target to a known-good path.
func TestRootLinkFrom(t *testing.T) {
	testenv.MustHaveLink(t)
	testRootMoveFrom(t, false)
}

func testRootMoveFrom(t *testing.T, rename bool) {
	want := []byte("target")
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			if target != "" {
				if err := os.WriteFile(target, want, 0o666); err != nil {
					t.Fatal(err)
				}
			}
			wantError := test.wantError
			var linkTarget string
			if test.ltarget != "" {
				// Rename will rename the link, not the file linked to.
				wantError = false
				var err error
				linkTarget, err = root.Readlink(test.ltarget)
				if err != nil {
					t.Fatalf("root.Readlink(%q) = %v, want success", test.ltarget, err)
				}

				// When GOOS=js, creating a hard link to a symlink fails.
				if !rename && runtime.GOOS == "js" {
					wantError = true
				}
			}

			const dstPath = "destination"

			// Plan 9 doesn't allow cross-directory renames.
			if runtime.GOOS == "plan9" && strings.Contains(test.open, "/") {
				wantError = true
			}

			var op string
			var err error
			if rename {
				op = "Rename"
				err = root.Rename(test.open, dstPath)
			} else {
				op = "Link"
				err = root.Link(test.open, dstPath)
			}
			if errEndsTest(t, err, wantError, "root.%v(%q, %q)", op, test.open, dstPath) {
				return
			}

			origPath := target
			if test.ltarget != "" {
				origPath = filepath.Join(root.Name(), test.ltarget)
			}
			_, err = os.Lstat(origPath)
			if rename {
				if !errors.Is(err, os.ErrNotExist) {
					t.Errorf("after renaming file, Lstat(%q) = %v, want ErrNotExist", origPath, err)
				}
			} else {
				if err != nil {
					t.Errorf("after linking file, error accessing original: %v", err)
				}
			}

			dstFullPath := filepath.Join(root.Name(), dstPath)
			if test.ltarget != "" {
				got, err := os.Readlink(dstFullPath)
				if err != nil || got != linkTarget {
					t.Errorf("os.Readlink(%q) = %q, %v, want %q", dstFullPath, got, err, linkTarget)
				}
			} else {
				got, err := os.ReadFile(dstFullPath)
				if err != nil || !bytes.Equal(got, want) {
					t.Errorf(`os.ReadFile(%q): read content %q, %v; want %q`, dstFullPath, string(got), err, string(want))
				}
				st, err := os.Lstat(dstFullPath)
				if err != nil || st.Mode()&fs.ModeSymlink != 0 {
					t.Errorf(`os.Lstat(%q) = %v, %v; want non-symlink`, dstFullPath, st.Mode(), err)
				}

			}
		})
	}
}

// TestRootRenameTo tests renaming a known-good path to the test case target.
func TestRootRenameTo(t *testing.T) {
	testRootMoveTo(t, true)
}

// TestRootLinkTo tests renaming a known-good path to the test case target.
func TestRootLinkTo(t *testing.T) {
	testenv.MustHaveLink(t)
	testRootMoveTo(t, true)
}

func testRootMoveTo(t *testing.T, rename bool) {
	want := []byte("target")
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			const srcPath = "source"
			if err := os.WriteFile(filepath.Join(root.Name(), srcPath), want, 0o666); err != nil {
				t.Fatal(err)
			}

			target = test.target
			wantError := test.wantError
			if test.ltarget != "" {
				// Rename will overwrite the final link rather than follow it.
				target = test.ltarget
				wantError = false
			}

			// Plan 9 doesn't allow cross-directory renames.
			if runtime.GOOS == "plan9" && strings.Contains(test.open, "/") {
				wantError = true
			}

			var err error
			var op string
			if rename {
				op = "Rename"
				err = root.Rename(srcPath, test.open)
			} else {
				op = "Link"
				err = root.Link(srcPath, test.open)
			}
			if errEndsTest(t, err, wantError, "root.%v(%q, %q)", op, srcPath, test.open) {
				return
			}

			_, err = os.Lstat(filepath.Join(root.Name(), srcPath))
			if rename {
				if !errors.Is(err, os.ErrNotExist) {
					t.Errorf("after renaming file, Lstat(%q) = %v, want ErrNotExist", srcPath, err)
				}
			} else {
				if err != nil {
					t.Errorf("after linking file, error accessing original: %v", err)
				}
			}

			got, err := os.ReadFile(filepath.Join(root.Name(), target))
			if err != nil || !bytes.Equal(got, want) {
				t.Errorf(`os.ReadFile(%q): read content %q, %v; want %q`, target, string(got), err, string(want))
			}
		})
	}
}

func TestRootSymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			wantError := test.wantError
			if test.ltarget != "" {
				// We can't create a symlink over an existing symlink.
				wantError = true
			}

			const wantTarget = "linktarget"
			err := root.Symlink(wantTarget, test.open)
			if errEndsTest(t, err, wantError, "root.Symlink(%q)", test.open) {
				return
			}
			got, err := os.Readlink(target)
			if err != nil || got != wantTarget {
				t.Fatalf("ReadLink(%q) = %q, %v; want %q, nil", target, got, err, wantTarget)
			}
		})
	}
}

// A rootConsistencyTest is a test case comparing os.Root behavior with
// the corresponding non-Root function.
//
// These tests verify that, for example, Root.Open("file/./") and os.Open("file/./")
// have the same result, although the specific result may vary by platform.
type rootConsistencyTest struct {
	name string

	// fs is the test filesystem layout. See makefs above.
	// fsFunc is called to modify the test filesystem, or replace it.
	fs     []string
	fsFunc func(t *testing.T, dir string) string

	// open is the filename to access in the test.
	open string

	// detailedErrorMismatch indicates that os.Root and the corresponding non-Root
	// function return different errors for this test.
	detailedErrorMismatch func(t *testing.T) bool
}

var rootConsistencyTestCases = []rootConsistencyTest{{
	name: "file",
	fs: []string{
		"target",
	},
	open: "target",
}, {
	name: "dir slash dot",
	fs: []string{
		"target/file",
	},
	open: "target/.",
}, {
	name: "dot",
	fs: []string{
		"file",
	},
	open: ".",
}, {
	name: "file slash dot",
	fs: []string{
		"target",
	},
	open: "target/.",
	detailedErrorMismatch: func(t *testing.T) bool {
		// FreeBSD returns EPERM in the non-Root case.
		return runtime.GOOS == "freebsd" && strings.HasPrefix(t.Name(), "TestRootConsistencyRemove")
	},
}, {
	name: "dir slash",
	fs: []string{
		"target/file",
	},
	open: "target/",
}, {
	name: "dot slash",
	fs: []string{
		"file",
	},
	open: "./",
}, {
	name: "file slash",
	fs: []string{
		"target",
	},
	open: "target/",
	detailedErrorMismatch: func(t *testing.T) bool {
		// os.Create returns ENOTDIR or EISDIR depending on the platform.
		return runtime.GOOS == "js"
	},
}, {
	name: "file in path",
	fs: []string{
		"file",
	},
	open: "file/target",
}, {
	name: "directory in path missing",
	open: "dir/target",
}, {
	name: "target does not exist",
	open: "target",
}, {
	name: "symlink slash",
	fs: []string{
		"target/file",
		"link => target",
	},
	open: "link/",
}, {
	name: "symlink slash dot",
	fs: []string{
		"target/file",
		"link => target",
	},
	open: "link/.",
}, {
	name: "file symlink slash",
	fs: []string{
		"target",
		"link => target",
	},
	open: "link/",
	detailedErrorMismatch: func(t *testing.T) bool {
		// os.Create returns ENOTDIR or EISDIR depending on the platform.
		return runtime.GOOS == "js"
	},
}, {
	name: "unresolved symlink",
	fs: []string{
		"link => target",
	},
	open: "link",
}, {
	name: "resolved symlink",
	fs: []string{
		"link => target",
		"target",
	},
	open: "link",
}, {
	name: "dotdot in path after symlink",
	fs: []string{
		"a => b/c",
		"b/c/",
		"b/target",
	},
	open: "a/../target",
}, {
	name: "long file name",
	open: strings.Repeat("a", 500),
}, {
	name: "unreadable directory",
	fs: []string{
		"dir/target",
	},
	fsFunc: func(t *testing.T, dir string) string {
		os.Chmod(filepath.Join(dir, "dir"), 0)
		t.Cleanup(func() {
			os.Chmod(filepath.Join(dir, "dir"), 0o700)
		})
		return dir
	},
	open: "dir/target",
}, {
	name: "unix domain socket target",
	fsFunc: func(t *testing.T, dir string) string {
		return tempDirWithUnixSocket(t, "a")
	},
	open: "a",
}, {
	name: "unix domain socket in path",
	fsFunc: func(t *testing.T, dir string) string {
		return tempDirWithUnixSocket(t, "a")
	},
	open: "a/b",
	detailedErrorMismatch: func(t *testing.T) bool {
		// On Windows, os.Root.Open returns "The directory name is invalid."
		// and os.Open returns "The file cannot be accessed by the system.".
		return runtime.GOOS == "windows"
	},
}, {
	name: "question mark",
	open: "?",
}, {
	name: "nul byte",
	open: "\x00",
}}

func tempDirWithUnixSocket(t *testing.T, name string) string {
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Error(err)
		}
	})
	addr, err := net.ResolveUnixAddr("unix", filepath.Join(dir, name))
	if err != nil {
		t.Skipf("net.ResolveUnixAddr: %v", err)
	}
	conn, err := net.ListenUnix("unix", addr)
	if err != nil {
		t.Skipf("net.ListenUnix: %v", err)
	}
	t.Cleanup(func() {
		conn.Close()
	})
	return dir
}

func (test rootConsistencyTest) run(t *testing.T, f func(t *testing.T, path string, r *os.Root) (string, error)) {
	if runtime.GOOS == "wasip1" {
		// On wasip, non-Root functions clean paths before opening them,
		// resulting in inconsistent behavior.
		// https://go.dev/issue/69509
		t.Skip("#69509: inconsistent results on wasip1")
	}

	t.Run(test.name, func(t *testing.T) {
		dir1 := makefs(t, test.fs)
		dir2 := makefs(t, test.fs)
		if test.fsFunc != nil {
			dir1 = test.fsFunc(t, dir1)
			dir2 = test.fsFunc(t, dir2)
		}

		r, err := os.OpenRoot(dir1)
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()

		res1, err1 := f(t, test.open, r)
		res2, err2 := f(t, dir2+"/"+test.open, nil)

		if res1 != res2 || ((err1 == nil) != (err2 == nil)) {
			t.Errorf("with root:    res=%v", res1)
			t.Errorf("              err=%v", err1)
			t.Errorf("without root: res=%v", res2)
			t.Errorf("              err=%v", err2)
			t.Errorf("want consistent results, got mismatch")
		}

		if err1 != nil || err2 != nil {
			underlyingError := func(how string, err error) error {
				switch e := err1.(type) {
				case *os.PathError:
					return e.Err
				case *os.LinkError:
					return e.Err
				default:
					t.Fatalf("%v, expected PathError or LinkError; got: %v", how, err)
				}
				return nil
			}
			e1 := underlyingError("with root", err1)
			e2 := underlyingError("without root", err1)
			detailedErrorMismatch := false
			if f := test.detailedErrorMismatch; f != nil {
				detailedErrorMismatch = f(t)
			}
			if runtime.GOOS == "plan9" {
				// Plan9 syscall errors aren't comparable.
				detailedErrorMismatch = true
			}
			if !detailedErrorMismatch && e1 != e2 {
				t.Errorf("with root:    err=%v", e1)
				t.Errorf("without root: err=%v", e2)
				t.Errorf("want consistent results, got mismatch")
			}
		}
	})
}

func TestRootConsistencyOpen(t *testing.T) {
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			var f *os.File
			var err error
			if r == nil {
				f, err = os.Open(path)
			} else {
				f, err = r.Open(path)
			}
			if err != nil {
				return "", err
			}
			defer f.Close()
			fi, err := f.Stat()
			if err == nil && !fi.IsDir() {
				b, err := io.ReadAll(f)
				return string(b), err
			} else {
				names, err := f.Readdirnames(-1)
				slices.Sort(names)
				return fmt.Sprintf("%q", names), err
			}
		})
	}
}

func TestRootConsistencyCreate(t *testing.T) {
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			var f *os.File
			var err error
			if r == nil {
				f, err = os.Create(path)
			} else {
				f, err = r.Create(path)
			}
			if err == nil {
				f.Write([]byte("file contents"))
				f.Close()
			}
			return "", err
		})
	}
}

func TestRootConsistencyChmod(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("Chmod not supported on " + runtime.GOOS)
	}
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			chmod := os.Chmod
			lstat := os.Lstat
			if r != nil {
				chmod = r.Chmod
				lstat = r.Lstat
			}

			var m1, m2 os.FileMode
			if err := chmod(path, 0o555); err != nil {
				return "chmod 0o555", err
			}
			fi, err := lstat(path)
			if err == nil {
				m1 = fi.Mode()
			}
			if err = chmod(path, 0o777); err != nil {
				return "chmod 0o777", err
			}
			fi, err = lstat(path)
			if err == nil {
				m2 = fi.Mode()
			}
			return fmt.Sprintf("%v %v", m1, m2), err
		})
	}
}

func TestRootConsistencyMkdir(t *testing.T) {
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			var err error
			if r == nil {
				err = os.Mkdir(path, 0o777)
			} else {
				err = r.Mkdir(path, 0o777)
			}
			return "", err
		})
	}
}

func TestRootConsistencyRemove(t *testing.T) {
	for _, test := range rootConsistencyTestCases {
		if test.open == "." || test.open == "./" {
			continue // can't remove the root itself
		}
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			var err error
			if r == nil {
				err = os.Remove(path)
			} else {
				err = r.Remove(path)
			}
			return "", err
		})
	}
}

func TestRootConsistencyStat(t *testing.T) {
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			var fi os.FileInfo
			var err error
			if r == nil {
				fi, err = os.Stat(path)
			} else {
				fi, err = r.Stat(path)
			}
			if err != nil {
				return "", err
			}
			return fmt.Sprintf("name:%q size:%v mode:%v isdir:%v", fi.Name(), fi.Size(), fi.Mode(), fi.IsDir()), nil
		})
	}
}

func TestRootConsistencyLstat(t *testing.T) {
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			var fi os.FileInfo
			var err error
			if r == nil {
				fi, err = os.Lstat(path)
			} else {
				fi, err = r.Lstat(path)
			}
			if err != nil {
				return "", err
			}
			return fmt.Sprintf("name:%q size:%v mode:%v isdir:%v", fi.Name(), fi.Size(), fi.Mode(), fi.IsDir()), nil
		})
	}
}

func TestRootConsistencyReadlink(t *testing.T) {
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			if r == nil {
				return os.Readlink(path)
			} else {
				return r.Readlink(path)
			}
		})
	}
}

func TestRootConsistencyRename(t *testing.T) {
	testRootConsistencyMove(t, true)
}

func TestRootConsistencyLink(t *testing.T) {
	testenv.MustHaveLink(t)
	testRootConsistencyMove(t, false)
}

func testRootConsistencyMove(t *testing.T, rename bool) {
	if runtime.GOOS == "plan9" {
		// This test depends on moving files between directories.
		t.Skip("Plan 9 does not support cross-directory renames")
	}
	// Run this test in two directions:
	// Renaming the test path to a known-good path (from),
	// and renaming a known-good path to the test path (to).
	for _, name := range []string{"from", "to"} {
		t.Run(name, func(t *testing.T) {
			for _, test := range rootConsistencyTestCases {
				if runtime.GOOS == "windows" {
					// On Windows, Rename("/path/to/.", x) succeeds,
					// because Windows cleans the path to just "/path/to".
					// Root.Rename(".", x) fails as expected.
					// Don't run this consistency test on Windows.
					if test.open == "." || test.open == "./" {
						continue
					}
				}

				test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
					var move func(oldname, newname string) error
					switch {
					case rename && r == nil:
						move = os.Rename
					case rename && r != nil:
						move = r.Rename
					case !rename && r == nil:
						move = os.Link
					case !rename && r != nil:
						move = r.Link
					}
					lstat := os.Lstat
					if r != nil {
						lstat = r.Lstat
					}

					otherPath := "other"
					if r == nil {
						otherPath = filepath.Join(t.TempDir(), otherPath)
					}

					var srcPath, dstPath string
					if name == "from" {
						srcPath = path
						dstPath = otherPath
					} else {
						srcPath = otherPath
						dstPath = path
					}

					if !rename {
						// When the source is a symlink, Root.Link creates
						// a hard link to the symlink.
						// os.Link does whatever the link syscall does,
						// which varies between operating systems and
						// their versions.
						// Skip running the consistency test when
						// the source is a symlink.
						fi, err := lstat(srcPath)
						if err == nil && fi.Mode()&os.ModeSymlink != 0 {
							return "", nil
						}
					}

					if err := move(srcPath, dstPath); err != nil {
						return "", err
					}
					fi, err := lstat(dstPath)
					if err != nil {
						t.Errorf("stat(%q) after successful copy: %v", dstPath, err)
						return "stat error", err
					}
					return fmt.Sprintf("name:%q size:%v mode:%v isdir:%v", fi.Name(), fi.Size(), fi.Mode(), fi.IsDir()), nil
				})
			}
		})
	}
}

func TestRootConsistencySymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			const target = "linktarget"
			var err error
			var got string
			if r == nil {
				err = os.Symlink(target, path)
				got, _ = os.Readlink(target)
			} else {
				err = r.Symlink(target, path)
				got, _ = r.Readlink(target)
			}
			return got, err
		})
	}
}

func TestRootRenameAfterOpen(t *testing.T) {
	switch runtime.GOOS {
	case "windows":
		t.Skip("renaming open files not supported on " + runtime.GOOS)
	case "js", "plan9":
		t.Skip("openat not supported on " + runtime.GOOS)
	case "wasip1":
		if os.Getenv("GOWASIRUNTIME") == "wazero" {
			t.Skip("wazero does not track renamed directories")
		}
	}

	dir := t.TempDir()

	// Create directory "a" and open it.
	if err := os.Mkdir(filepath.Join(dir, "a"), 0o777); err != nil {
		t.Fatal(err)
	}
	dirf, err := os.OpenRoot(filepath.Join(dir, "a"))
	if err != nil {
		t.Fatal(err)
	}
	defer dirf.Close()

	// Rename "a" => "b", and create "b/f".
	if err := os.Rename(filepath.Join(dir, "a"), filepath.Join(dir, "b")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "b/f"), []byte("hello"), 0o666); err != nil {
		t.Fatal(err)
	}

	// Open "f", and confirm that we see it.
	f, err := dirf.OpenFile("f", os.O_RDONLY, 0)
	if err != nil {
		t.Fatalf("reading file after renaming parent: %v", err)
	}
	defer f.Close()
	b, err := io.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(b), "hello"; got != want {
		t.Fatalf("file contents: %q, want %q", got, want)
	}

	// f.Name reflects the original path we opened the directory under (".../a"), not "b".
	if got, want := f.Name(), dirf.Name()+string(os.PathSeparator)+"f"; got != want {
		t.Errorf("f.Name() = %q, want %q", got, want)
	}
}

func TestRootNonPermissionMode(t *testing.T) {
	r, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := r.OpenFile("file", os.O_RDWR|os.O_CREATE, 0o1777); err == nil {
		t.Errorf("r.OpenFile(file, O_RDWR|O_CREATE, 0o1777) succeeded; want error")
	}
	if err := r.Mkdir("file", 0o1777); err == nil {
		t.Errorf("r.Mkdir(file, 0o1777) succeeded; want error")
	}
}

func TestRootUseAfterClose(t *testing.T) {
	r, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	r.Close()
	for _, test := range []struct {
		name string
		f    func(r *os.Root, filename string) error
	}{{
		name: "Open",
		f: func(r *os.Root, filename string) error {
			_, err := r.Open(filename)
			return err
		},
	}, {
		name: "Create",
		f: func(r *os.Root, filename string) error {
			_, err := r.Create(filename)
			return err
		},
	}, {
		name: "OpenFile",
		f: func(r *os.Root, filename string) error {
			_, err := r.OpenFile(filename, os.O_RDWR, 0o666)
			return err
		},
	}, {
		name: "OpenRoot",
		f: func(r *os.Root, filename string) error {
			_, err := r.OpenRoot(filename)
			return err
		},
	}, {
		name: "Mkdir",
		f: func(r *os.Root, filename string) error {
			return r.Mkdir(filename, 0o777)
		},
	}} {
		err := test.f(r, "target")
		pe, ok := err.(*os.PathError)
		if !ok || pe.Path != "target" || pe.Err != os.ErrClosed {
			t.Errorf(`r.%v = %v; want &PathError{Path: "target", Err: ErrClosed}`, test.name, err)
		}
	}
}

func TestRootConcurrentClose(t *testing.T) {
	r, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	ch := make(chan error, 1)
	go func() {
		defer close(ch)
		first := true
		for {
			f, err := r.OpenFile("file", os.O_RDWR|os.O_CREATE, 0o666)
			if err != nil {
				ch <- err
				return
			}
			if first {
				ch <- nil
				first = false
			}
			f.Close()
			if runtime.GOARCH == "wasm" {
				// TODO(go.dev/issue/71134) can lead to goroutine starvation.
				runtime.Gosched()
			}
		}
	}()
	if err := <-ch; err != nil {
		t.Errorf("OpenFile: %v, want success", err)
	}
	r.Close()
	if err := <-ch; !errors.Is(err, os.ErrClosed) {
		t.Errorf("OpenFile: %v, want ErrClosed", err)
	}
}

// TestRootRaceRenameDir attempts to escape a Root by renaming a path component mid-parse.
//
// We create a deeply nested directory:
//
//	base/a/a/a/a/ [...] /a
//
// And a path that descends into the tree, then returns to the top using ..:
//
//	base/a/a/a/a/ [...] /a/../../../ [..] /../a/f
//
// While opening this file, we rename base/a/a to base/b.
// A naive lookup operation will resolve the path to base/f.
func TestRootRaceRenameDir(t *testing.T) {
	dir := t.TempDir()
	r, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	const depth = 4

	os.MkdirAll(dir+"/base/"+strings.Repeat("/a", depth), 0o777)

	path := "base/" + strings.Repeat("a/", depth) + strings.Repeat("../", depth) + "a/f"
	os.WriteFile(dir+"/f", []byte("secret"), 0o666)
	os.WriteFile(dir+"/base/a/f", []byte("public"), 0o666)

	// Compute how long it takes to open the path in the common case.
	const tries = 10
	var total time.Duration
	for range tries {
		start := time.Now()
		f, err := r.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(f)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != "public" {
			t.Fatalf("read %q, want %q", b, "public")
		}
		f.Close()
		total += time.Since(start)
	}
	avg := total / tries

	// We're trying to exploit a race, so try this a number of times.
	for range 100 {
		// Start a goroutine to open the file.
		gotc := make(chan []byte)
		go func() {
			f, err := r.Open(path)
			if err != nil {
				gotc <- nil
			}
			defer f.Close()
			b, _ := io.ReadAll(f)
			gotc <- b
		}()

		// Wait for the open operation to partially complete,
		// and then rename a directory near the root.
		time.Sleep(avg / 4)
		if err := os.Rename(dir+"/base/a", dir+"/b"); err != nil {
			// Windows and Plan9 won't let us rename a directory if we have
			// an open handle for it, so an error here is expected.
			switch runtime.GOOS {
			case "windows", "plan9":
			default:
				t.Fatal(err)
			}
		}

		got := <-gotc
		os.Rename(dir+"/b", dir+"/base/a")
		if len(got) > 0 && string(got) != "public" {
			t.Errorf("read file: %q; want error or 'public'", got)
		}
	}
}

func TestRootSymlinkToRoot(t *testing.T) {
	dir := makefs(t, []string{
		"d/d => ..",
	})
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	if err := root.Mkdir("d/d/new", 0777); err != nil {
		t.Fatal(err)
	}
	f, err := root.Open("d/d")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	names, err := f.Readdirnames(-1)
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(names)
	if got, want := names, []string{"d", "new"}; !slices.Equal(got, want) {
		t.Errorf("root contains: %q, want %q", got, want)
	}
}

func TestOpenInRoot(t *testing.T) {
	dir := makefs(t, []string{
		"file",
		"link => ../ROOT/file",
	})
	f, err := os.OpenInRoot(dir, "file")
	if err != nil {
		t.Fatalf("OpenInRoot(`file`) = %v, want success", err)
	}
	f.Close()
	for _, name := range []string{
		"link",
		"../ROOT/file",
		dir + "/file",
	} {
		f, err := os.OpenInRoot(dir, name)
		if err == nil {
			f.Close()
			t.Fatalf("OpenInRoot(%q) = nil, want error", name)
		}
	}
}
