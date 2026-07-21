// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"internal/testenv"
	"io"
	"io/fs"
	"iter"
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
	root := filepath.Join(t.TempDir(), "ROOT")
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

// hasLink reports whether the test filesystem layout fs
// contains at least a single symlink.
func hasLink(fs []string) bool {
	for _, ent := range fs {
		isLink := strings.Contains(ent, " => ")
		if isLink {
			return true
		}
	}
	return false
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
		if hasLink(test.fs) {
			testenv.MustHaveSymlink(t)
		}
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
	name: "symlink dotdot slash",
	fs: []string{
		"link => ../",
	},
	open:      "link",
	ltarget:   "link",
	wantError: true,
}, {
	name: "symlink ending in slash",
	fs: []string{
		"dir/",
		"link => dir/",
	},
	open:   "link/target",
	target: "dir/target",
}, {
	name: "slash after symlink to file",
	fs: []string{
		"link => ../ROOT/target",
	},
	open:      "link/",
	target:    "target",
	wantError: true,
}, {
	name: "slash after symlink to dir",
	fs: []string{
		"link => ../ROOT/target",
		"target/",
	},
	open:      "link/",
	wantError: true,
}, {
	name: "symlink dotdot dotdot slash",
	fs: []string{
		"dir/link => ../../",
	},
	open:      "dir/link",
	ltarget:   "dir/link",
	wantError: true,
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
	name:      "path with dotdot slash",
	fs:        []string{},
	open:      "../",
	wantError: true,
}, {
	name: "path with dotdot dotdot slash",
	fs: []string{
		"a/",
	},
	open:      "a/../../",
	wantError: true,
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
			if test.ltarget != "" {
				// This case is trying to mkdir("some symlink"),
				// which is an error (but not an escape).
				wantError = true
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

func TestRootMkdirAll(t *testing.T) {
	for _, test := range rootTestCases {
		if test.name == "directory does not exist" {
			// Test expects error, mkdirall creates the missing directory.
			// TestRootMultiMkdirAll covers this case better anyway, just skip.
			continue
		}
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			wantError := test.wantError
			if test.ltarget != "" {
				// This case is trying to mkdir("some symlink"),
				// which is an error (but not an escape).
				wantError = true
			}

			err := root.MkdirAll(test.open, 0o777)
			if errEndsTest(t, err, wantError, "root.MkdirAll(%q)", test.open) {
				return
			}
			fi, err := os.Lstat(target)
			if err != nil {
				t.Fatalf(`stat file created with Root.MkdirAll(%q): %v`, test.open, err)
			}
			if !fi.IsDir() {
				t.Fatalf(`stat file created with Root.MkdirAll(%q): not a directory`, test.open)
			}
			if mode := fi.Mode(); mode&0o777 == 0 {
				// Issue #73559: We're not going to worry about the exact
				// mode bits (which will have been modified by umask),
				// but there should be mode bits.
				t.Fatalf(`stat file created with Root.MkdirAll(%q): mode=%v, want non-zero`, test.open, mode)
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

func TestRootRemoveAll(t *testing.T) {
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			if strings.HasSuffix(test.open, "/") {
				// The test is removing a file with a trailing /.
				// RemoveAll ignores trailing /s
				// If the file is a symlink, it will remove the symlink.
				fullname := filepath.Join(root.Name(), test.open)
				if st, err := os.Lstat(fullname); err == nil && st.Mode().Type() == fs.ModeSymlink {
					test.ltarget = test.open
				}
			}
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
				if err := os.WriteFile(filepath.Join(target, "file"), nil, 0o666); err != nil {
					t.Fatal(err)
				}
			}
			targetExists := true
			if _, err := root.Lstat(test.open); errors.Is(err, os.ErrNotExist) {
				// If the target doesn't exist, RemoveAll succeeds rather
				// than returning ErrNotExist.
				targetExists = false
				wantError = false
			}

			err := root.RemoveAll(test.open)
			if errEndsTest(t, err, wantError, "root.RemoveAll(%q)", test.open) {
				return
			}
			if !targetExists {
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

				// Windows allows creating a hard link to a file symlink,
				// but not to a directory symlink.
				//
				// This uses os.Stat to check the link target, because this
				// is easier than figuring out whether the link itself is a
				// directory link. The link was created with os.Symlink,
				// which creates directory links when the target is a directory,
				// so this is good enough for a test.
				if !rename && runtime.GOOS == "windows" {
					st, err := os.Stat(filepath.Join(root.Name(), test.ltarget))
					if err == nil && st.IsDir() {
						wantError = true
					}
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

			if runtime.GOOS == "windows" && strings.HasSuffix(test.open, "/") {
				// Windows will ignore trailing slashes in the rename/link target.
				p := strings.TrimSuffix(test.open, "/")
				st, err := root.Lstat(p)
				if err == nil && st.Mode().Type() == fs.ModeSymlink {
					test.ltarget = p
				}
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

	// check is called before the test starts, and may t.Skip if necessary.
	check func(t *testing.T)
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
	check: func(t *testing.T) {
		if runtime.GOOS == "linux" && strings.HasPrefix(t.Name(), "TestRootConsistencyRename/") {
			// Linux does not resolve "symlink" in rename("symlink/", "target").
			t.Skip("known inconsistency on linux")
		}
		if strings.HasPrefix(t.Name(), "TestRootConsistencyRemoveAll/") {
			// Root.RemoveAll and os.RemoveAll are not always consistent here.
			t.Skip("known inconsistency in RemoveAll")
		}
	},
}, {
	name: "symlink slash dot",
	fs: []string{
		"target/file",
		"link => target",
	},
	open: "link/.",
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
	name: "symlink to dir ends in slash",
	fs: []string{
		"dir/",
		"link => dir/",
	},
	open: "link",
}, {
	name: "symlink to file ends in slash",
	fs: []string{
		"file",
		"link => file/",
	},
	open: "link",
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
	check: func(t *testing.T) {
		if strings.HasPrefix(t.Name(), "TestRootConsistencyRemoveAll/") {
			switch runtime.GOOS {
			case "windows":
				// Root.RemoveAll notices that a/ is not a directory,
				// and returns success.
				// os.RemoveAll tries to open a/ and fails because
				// it is not a regular file.
				// The inconsistency here isn't worth fixing, so just skip this test.
				t.Skip("known inconsistency on windows")
			case "js":
				// GOOS=js behavior varies with what the underlying OS is.
				t.Skip("known inconsistency with GOOS=js")
			}
		}
	},
}, {
	name: "question mark",
	open: "?",
}, {
	name: "nul byte",
	open: "\x00",
}}

func tempDirWithUnixSocket(t *testing.T, name string) string {
	dir := t.TempDir()
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
		if test.check != nil {
			test.check(t)
		}

		if hasLink(test.fs) {
			testenv.MustHaveSymlink(t)
		}

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

func TestRootConsistencyMkdirAll(t *testing.T) {
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			var err error
			if r == nil {
				err = os.MkdirAll(path, 0o777)
			} else {
				err = r.MkdirAll(path, 0o777)
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

func TestRootConsistencyRemoveAll(t *testing.T) {
	for _, test := range rootConsistencyTestCases {
		if test.open == "." || test.open == "./" {
			continue // can't remove the root itself
		}
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			var err error
			if r == nil {
				err = os.RemoveAll(path)
			} else {
				err = r.RemoveAll(path)
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
	testenv.MustHaveSymlink(t)
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
	testenv.MustHaveSymlink(t)
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

func TestRootRemoveDot(t *testing.T) {
	dir := t.TempDir()
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	if err := root.Remove("."); err == nil {
		t.Errorf(`root.Remove(".") = %v, want error`, err)
	}
	if err := root.RemoveAll("."); err == nil {
		t.Errorf(`root.RemoveAll(".") = %v, want error`, err)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Error(`root.Remove(All)?(".") removed the root`)
	}
}

func TestRootWriteReadFile(t *testing.T) {
	dir := t.TempDir()
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	name := "filename"
	want := []byte("file contents")
	if err := root.WriteFile(name, want, 0o666); err != nil {
		t.Fatalf("root.WriteFile(%q, %q, 0o666) = %v; want nil", name, want, err)
	}

	got, err := root.ReadFile(name)
	if err != nil {
		t.Fatalf("root.ReadFile(%q) = %q, %v; want %q, nil", name, got, err, want)
	}
}

func TestRootName(t *testing.T) {
	dir := t.TempDir()
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	if got, want := root.Name(), dir; got != want {
		t.Errorf("root.Name() = %q, want %q", got, want)
	}

	f, err := root.Create("file")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if got, want := f.Name(), filepath.Join(dir, "file"); got != want {
		t.Errorf(`root.Create("file").Name() = %q, want %q`, got, want)
	}

	if err := root.Mkdir("dir", 0o777); err != nil {
		t.Fatal(err)
	}
	subroot, err := root.OpenRoot("dir")
	if err != nil {
		t.Fatal(err)
	}
	defer subroot.Close()
	if got, want := subroot.Name(), filepath.Join(dir, "dir"); got != want {
		t.Errorf(`root.OpenRoot("dir").Name() = %q, want %q`, got, want)
	}
}

// TestRootNoLstat verifies that we do not use lstat (possibly escaping the root)
// when reading directories in a Root.
func TestRootNoLstat(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("wasm lacks fstatat")
	}

	dir := makefs(t, []string{
		"subdir/",
	})
	const size = 42
	contents := strings.Repeat("x", size)
	if err := os.WriteFile(dir+"/subdir/file", []byte(contents), 0666); err != nil {
		t.Fatal(err)
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	test := func(name string, fn func(t *testing.T, f *os.File)) {
		t.Run(name, func(t *testing.T) {
			os.SetStatHook(t, func(f *os.File, name string) (os.FileInfo, error) {
				if f == nil {
					t.Errorf("unexpected Lstat(%q)", name)
				}
				return nil, nil
			})
			f, err := root.Open("subdir")
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			fn(t, f)
		})
	}

	checkFileInfo := func(t *testing.T, fi fs.FileInfo) {
		t.Helper()
		if got, want := fi.Name(), "file"; got != want {
			t.Errorf("FileInfo.Name() = %q, want %q", got, want)
		}
		if got, want := fi.Size(), int64(size); got != want {
			t.Errorf("FileInfo.Size() = %v, want %v", got, want)
		}
	}
	checkDirEntry := func(t *testing.T, d fs.DirEntry) {
		t.Helper()
		if got, want := d.Name(), "file"; got != want {
			t.Errorf("DirEntry.Name() = %q, want %q", got, want)
		}
		if got, want := d.IsDir(), false; got != want {
			t.Errorf("DirEntry.IsDir() = %v, want %v", got, want)
		}
		fi, err := d.Info()
		if err != nil {
			t.Fatalf("DirEntry.Info() = _, %v", err)
		}
		checkFileInfo(t, fi)
	}

	test("Stat", func(t *testing.T, subdir *os.File) {
		fi, err := subdir.Stat()
		if err != nil {
			t.Fatal(err)
		}
		if !fi.IsDir() {
			t.Fatalf(`Open("subdir").Stat().IsDir() = false, want true`)
		}
	})
	// File.ReadDir, returning []DirEntry
	test("ReadDirEntry", func(t *testing.T, subdir *os.File) {
		dirents, err := subdir.ReadDir(-1)
		if err != nil {
			t.Fatal(err)
		}
		if len(dirents) != 1 {
			t.Fatalf(`Open("subdir").ReadDir(-1) = {%v}, want {file}`, dirents)
		}
		checkDirEntry(t, dirents[0])
	})
	// File.Readdir, returning []FileInfo
	test("ReadFileInfo", func(t *testing.T, subdir *os.File) {
		fileinfos, err := subdir.Readdir(-1)
		if err != nil {
			t.Fatal(err)
		}
		if len(fileinfos) != 1 {
			t.Fatalf(`Open("subdir").Readdir(-1) = {%v}, want {file}`, fileinfos)
		}
		checkFileInfo(t, fileinfos[0])
	})
	// File.Readdirnames, returning []string
	test("Readdirnames", func(t *testing.T, subdir *os.File) {
		names, err := subdir.Readdirnames(-1)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := names, []string{"file"}; !slices.Equal(got, want) {
			t.Fatalf(`Open("subdir").Readdirnames(-1) = %q, want %q`, got, want)
		}
	})
}

// A rootMultiTest is state for testing an os.Root operation in one configuration among many.
// Each execution of a rootMultiTest varies in several ways:
//
//   - With or without an *os.Root, to check consistency between root/non-root operations.
//   - With a target that may be a file, directory, symlink, or entirely absent.
//   - With various paths referencing the target: "target", "DIR/../target", etc.
//   - When the target is a symlink, with various link target paths.
//
// For example, a single test execution might be:
// In an *os.Root, copy "source" to "DIR/../target".
// "source" is a file, and "target" is a symlink to "../ROOT/s_target". "s_target" is a directory.
// (In this case, we expect the test to fail due to the path escape in the symlink.)
type rootMultiTest struct {
	// dir is the directory containing the test.
	// dir will always contain a directory named "ROOT"
	// and a subdir named "ROOT/DIR".
	dir string

	// root is the *Root for the test. May be nil.
	root *os.Root

	// source and target are files acted on by the test.
	// target is always set; source is only set for tests which request two files.
	source testFileDesc
	target testFileDesc

	// sourcePath and targetPath are the paths which should be used to acceess
	// the source/target.
	sourcePath string
	targetPath string

	sourceInfo os.FileInfo
	targetInfo os.FileInfo

	// op is the operation being performed, used for reporting errors.
	op string
}

var testVerbose = flag.Bool("verbose", false, "verbose")

// A rootMultiTest function may return this error to disable
// the check that in-root and out-of-root functions have the same outcome.
var errSkipRootConsistencyCheck = errors.New("skip root consistency check")

// runRootMultiTest runs f in a variety of configurations.
// See above.
func runRootMultiTest(t *testing.T, f func(*testing.T, *rootMultiTest) (string, error)) {
	for target := range allTestFileDescs() {
		t.Run(target.String(), func(t *testing.T) {
			var source testFileDesc // unused
			runRootMultiTestDescs(t, source, target, f)
		})
	}
}

// runRootMultiTest2 runs f in a variety of configurations,
// with both source and target files.
// See above.
func runRootMultiTest2(t *testing.T, f func(*testing.T, *rootMultiTest) (string, error)) {
	// A "simple" desc is one which contains only direct references.
	// When not running the comprehensive (but slow) set of test variations,
	// we only test variations where at least one of source and target is simple.
	isSimple := func(desc testFileDesc) bool {
		if desc.ref.template != "BASE" {
			return false
		}
		if desc.kind == testFileSymlink && desc.target.ref.template != "BASE" {
			return false
		}
		return true
	}
	for source := range allTestFileDescs() {
		for target := range allTestFileDescs() {
			if !*rootComprehensive && !isSimple(source) && !isSimple(target) {
				continue
			}
			name := fmt.Sprintf("%s_to_%s", source, target)
			t.Run(name, func(t *testing.T) {
				runRootMultiTestDescs(t, source, target, f)
			})
		}
	}
}

// setOp sets the operation performed by the test (logged in errors).
//
// This currently assumes the operation will be a method of os.Root and a function in os
// (e.g., root.Open/os.Open).
func (test *rootMultiTest) setOp(format string, a ...any) {
	if test.root != nil {
		test.op = "root."
	} else {
		test.op = "os."
	}
	test.op += fmt.Sprintf(format, a...)
}

var errAny = errors.New("any error")

func (test *rootMultiTest) errorf(t *testing.T, format string, args ...any) {
	t.Errorf("%v:", test.op)
	t.Fatalf("  "+format, args...)
}

// wantError tests whether got matches want.
// If want is errAny, got may be any non-nil error.
func (test *rootMultiTest) wantError(t *testing.T, got, want error) {
	t.Helper()
	if errors.Is(got, want) || (got != nil && want == errAny) {
		return
	}
	t.Fatalf("%v:\ngot error:  %v\nwant error: %v", test.op, got, want)
}

func runRootMultiTestDescs(t *testing.T, source, target testFileDesc, f func(*testing.T, *rootMultiTest) (string, error)) {
	rootTest := newRootTest(t, source, target, true)
	osTest := newRootTest(t, source, target, false)

	initialContent := dirTreeContents(t, rootTest.dir)
	t.Cleanup(func() {
		if t.Failed() {
			t.Log("Initial directory contents:")
			for _, line := range initialContent {
				t.Logf("  %v", line)
			}
		}
	})

	rootResult, rootErr := f(t, rootTest)

	if runtime.GOOS == "darwin" {
		// Darwin appears to have a kernel bug which causes restrictions on paths
		// with a trailing / to not be applied during uncached path lookups.
		// These restrictions are applied during cached lookups, so the results
		// of operating on /-suffixed paths are inconsistent.
		//
		// An example of this Darwin behavior (as of 25.4.0) is:
		//   $ mkdir -p test/dir
		//   $ echo hello > test/file
		//   $ ln -s dir/../file test/link
		//   $ cat test/link/
		//   hello
		//   $ cat test/link/
		//   cat: test/link/: Not a directory
		//
		// Since Darwin isn't consistent with itself, we can't verify that we're
		// consistent with it.
		if rootTest.source.anySlashSuffix() || rootTest.target.anySlashSuffix() {
			return
		}
	}

	if runtime.GOOS == "wasip1" || runtime.GOOS == "js" {
		// WASI runtimes don't have any consistent behavior for handling paths with
		// a trailing /, so skip consistency tests for these paths.
		if rootTest.source.anySlashSuffix() || rootTest.target.anySlashSuffix() {
			return
		}
	}

	osResult, osErr := f(t, osTest)

	t.Cleanup(func() {
		if t.Failed() || !*testVerbose {
			return
		}
		rootContent := dirTreeContents(t, rootTest.dir)
		osContent := dirTreeContents(t, osTest.dir)
		t.Log("Initial directory contents:")
		for _, line := range initialContent {
			t.Logf("  %v", line)
		}
		t.Logf("%v:", rootTest.op)
		t.Logf("  result: %v", rootResult)
		t.Logf("  error: %v", rootErr)
		for _, line := range rootContent {
			t.Logf("  %v", line)
		}
		t.Logf("%v:", osTest.op)
		t.Logf("  result: %v", osResult)
		t.Logf("  error: %v", osErr)
		for _, line := range osContent {
			t.Logf("  %v", line)
		}
	})

	if errors.Is(rootErr, os.ErrPathEscapes) {
		// os.Root forbids this operation (and is therefore not consistent with
		// the non-root version).
		return
	}

	if rootErr == errSkipRootConsistencyCheck || osErr == errSkipRootConsistencyCheck {
		return
	}

	// Consistency check: Performing the same operation in and out of a root
	// should produce the same results.
	if rootResult != osResult {
		t.Errorf("inconsistent results in/out of root")
		t.Errorf("%v:", rootTest.op)
		t.Errorf("  result: %v", rootResult)
		t.Errorf("%v:", osTest.op)
		t.Errorf("  result: %v", osResult)
	}
	if (rootErr == nil) != (osErr == nil) {
		t.Errorf("inconsistent errors in/out of root")
		t.Errorf("%v:", rootTest.op)
		t.Errorf("  error: %v", rootErr)
		t.Errorf("%v:", osTest.op)
		t.Errorf("  error: %v", osErr)
	}

	// Filesystem consistency check: Same files in the same places.
	rootContent := dirTreeContents(t, rootTest.dir)
	osContent := dirTreeContents(t, osTest.dir)
	if !slices.Equal(rootContent, osContent) {
		t.Errorf("inconsistent filesystem after running in/out of root")
		t.Errorf("%v:", rootTest.op)
		for _, line := range rootContent {
			t.Errorf("  %v", line)
		}
		t.Errorf("%v:", osTest.op)
		for _, line := range osContent {
			t.Errorf("  %v", line)
		}
	}
}

func newRootTest(t *testing.T, source, target testFileDesc, inRoot bool) *rootMultiTest {
	dir := makefs(t, []string{
		"DIR/",
	})
	var root *os.Root
	if inRoot {
		var err error
		root, err = os.OpenRoot(dir)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			root.Close()
		})
	}
	test := &rootMultiTest{
		dir:    dir,
		root:   root,
		source: source,
		target: target,
	}
	createFile := func(name string, desc testFileDesc) (path string, fi os.FileInfo) {
		if desc.kind == testFileUnused {
			return "", nil
		}
		fi = desc.create(t, dir, name, name)
		path = desc.ref.path(dir, name)
		if !inRoot && !filepath.IsAbs(path) {
			path = dir + "/" + path
		}
		return path, fi
	}
	test.sourcePath, test.sourceInfo = createFile("source", source)
	test.targetPath, test.targetInfo = createFile("target", target)
	return test
}

// testFileKind is a kind of file.
type testFileKind int

const (
	testFileUnused  = testFileKind(iota)
	testFileAbsent  // file does not exist
	testFileFile    // regular file
	testFileDir     // directory
	testFileSymlink // symlink
	testFileMax

	// testFileError represents a path which fails during resolution,
	// such as "a/b" where "a" does not exist.
	testFileError
)

func (kind testFileKind) String() string {
	switch kind {
	case testFileUnused:
		return "unused"
	case testFileAbsent:
		return "absent"
	case testFileFile:
		return "file"
	case testFileDir:
		return "dir"
	case testFileSymlink:
		return "symlink"
	case testFileError:
		return "error"
	default:
		return fmt.Sprintf("testFileKind(%d)", kind)
	}
}

// testFileRef is a kind of reference to a file.
//
// Many path names can refer to the same file: f, ./f, /abs/path/to/f, somedir/../f, etc.
// A testFileRef describes some form of reference.
type testFileRef struct {
	// name is the name of the reference (not the file name).
	// These are a bit cryptic to keep test names short:
	// s (/ slash), p (.. parent), b (base), d (directory), r (root)
	name string

	// template is a template path.
	//
	// templates assume that the file is contained in a directory named "ROOT",
	// and that "ROOT/DIR" exists and is a directory.
	//
	// The string BASE in the template may be replaced with the file's basename.
	//
	// Absolute path templates start with /ROOT.
	template string

	// escapes indicates whether the path escapes the current directory.
	escapes bool
}

var testFileRefs = []testFileRef{
	{escapes: false, name: "b", template: "BASE"},
	{escapes: false, name: "bs", template: "BASE/"},
	{escapes: false, name: "dpb", template: "DIR/../BASE"},
	{escapes: false, name: "dpbs", template: "DIR/../BASE/"},
	{escapes: true, name: "prb", template: "../ROOT/BASE"},
	{escapes: true, name: "prbs", template: "../ROOT/BASE/"},
	{escapes: true, name: "srb", template: "/ROOT/BASE"},
	{escapes: true, name: "srbs", template: "/ROOT/BASE/"},
}

// testFileLimitedRefs is a smaller set of references which do not exercise path escapes
// (see allTestFileDescs).
var testFileLimitedRefs = testFileRefs[0:2]

// path creates a path using the template.
//
// dir is the absolute path to the root directory (which must be named "ROOT").
// base is the name of the target file within the root directory.
func (ref testFileRef) path(dir, base string) string {
	p := ref.template
	p = strings.ReplaceAll(p, "BASE", base)
	if trim, ok := strings.CutPrefix(p, "/ROOT"); ok {
		p = dir + trim
	}
	return p
}

// hasSlashSuffix reports whether the file reference ends in a /.
func (ref testFileRef) hasSlashSuffix() bool {
	return strings.HasSuffix(ref.template, "/")
}

// testFileDesc is a description of a type of file, combining the kind and reference type.
//
// Some sample testFileDescs:
//   - "name", a plain file.
//   - "DIR/../name", a directory
//   - "name/", where name is a symlink to "DIR/../target/", where target is a plain file.
type testFileDesc struct {
	kind   testFileKind
	ref    testFileRef
	target *testFileDesc // symlink target, nil when kind is not testFileSymlink
}

var rootComprehensive = flag.Bool("root_comprehensive", false,
	"run many more os.Root test variations (slow, uncertain value)")

// allTestFileDescs returns an iterator over all the testFileDescs we use in tests.
func allTestFileDescs() iter.Seq[testFileDesc] {
	// A testFileDesc contains a reference type ("name", "d/../name", "../r/name", etc.) and
	// a file kind (file, directory, symlink, etc.).
	//
	// When the kind is symlink, the desc contains a reference type and file kind for
	// the link target as well. We only exercise one level of symlink (although we
	// could do more), so this means a testFileDesc effectively contains four axes of
	// variation: ref, kind, symlink ref, symlink kind.
	//
	// For example:
	//
	//   - "name" is a file
	//   - "d/../name" is a directory
	//   - "name" is a symlink to "name2" which is a file
	//   - "d/../name" is a symlink to "d/../name2" which is a directory
	//   - etc.
	//
	// It is feasible to test every possible variation of these four axes,
	// but this is quite a few tests and gets quite slow. So by default we exclude
	// some variations. We test:
	//
	//   - every reference to every kind, except symlink
	//   - direct and direct/ references to a symlink to every reference to a file
	//   - a direct reference to a symlink to a direct reference to every kind (except file)
	//
	// The full set of variations may be enabled with the -comprehensive_root_tests flag.

	return func(yield func(testFileDesc) bool) {
		// Every type of reference to every type of file, except symlink.
		for _, ref := range testFileRefs {
			for kind := range testFileMax {
				if kind == testFileUnused || kind == testFileSymlink {
					continue
				}
				desc := testFileDesc{
					kind: kind,
					ref:  ref,
				}
				if !yield(desc) {
					return
				}
			}
		}

		// Unless we're being comprehensive, only direct references to symlinks.
		refs := testFileRefs
		if !*rootComprehensive {
			refs = testFileLimitedRefs
		}
		for _, ref := range refs {
			for linkKind := range testFileMax {
				if linkKind == testFileUnused || linkKind == testFileSymlink {
					continue
				}

				linkRefs := testFileRefs
				if !*rootComprehensive && linkKind != testFileFile && linkKind != testFileDir {
					linkRefs = testFileLimitedRefs
				}
				for _, linkRef := range linkRefs {
					desc := testFileDesc{
						kind: testFileSymlink,
						ref:  ref,
						target: &testFileDesc{
							kind: linkKind,
							ref:  linkRef,
						},
					}
					if !yield(desc) {
						return
					}
				}
			}
		}
	}
}

// String returns the target name.
//
// These are somewhat cryptic to keep test names short.
// For example, "bsSdpbD" is:
//
//	bs  - "BASE/"
//	S   - symlink
//	dpb - "DIR/../BASE"
//	D   - directory
//
// So, open "file1/", where file1 is a symlink to "DIR/../file2", where file2 is a directory.
func (desc testFileDesc) String() string {
	s := desc.ref.name + strings.ToUpper(desc.kind.String()[:1])
	if desc.kind == testFileSymlink {
		s += desc.target.String()
	}
	return s
}

// escapes reports whether accessing this file escapes the root,
// either because the file name escapes or because some element of a symlink chain escapes.
func (desc testFileDesc) escapes() bool {
	if desc.ref.escapes {
		return true
	}
	if desc.kind == testFileSymlink {
		return desc.target.escapes()
	}
	return false
}

func (desc testFileDesc) lescapes() bool {
	if desc.ref.escapes {
		return true
	}
	if runtime.GOOS == "windows" {
		// On POSIX filesystems, a trailing slash at the end of a path causes
		// symlinks in the last path component to be resolved.
		// On Windows, a trailing slash does not cause symlink resolution.
		return false
	}
	if desc.ref.hasSlashSuffix() && desc.kind == testFileSymlink {
		return desc.target.escapes()
	}
	return false
}

// finalKind reports the kind of the file after following all symlinks.
func (desc testFileDesc) finalKind() testFileKind {
	if desc.kind == testFileSymlink {
		return desc.target.finalKind()
	}
	return desc.kind
}

func (desc testFileDesc) lfinalKind() testFileKind {
	switch runtime.GOOS {
	case "windows":
		if desc.ref.hasSlashSuffix() && desc.kind == testFileSymlink && desc.target.kind != testFileDir {
			return testFileError
		}
	default:
		if desc.ref.hasSlashSuffix() && desc.kind == testFileSymlink {
			return desc.target.finalKind()
		}
	}
	return desc.kind
}

func (desc testFileDesc) isError() bool {
	if runtime.GOOS == "js" {
		return false
	}
	var isError func(desc testFileDesc, hasSuffix bool) bool
	isError = func(desc testFileDesc, hasSuffix bool) bool {
		if desc.ref.escapes {
			return false
		}
		if desc.ref.hasSlashSuffix() {
			hasSuffix = true
		}
		switch desc.kind {
		case testFileDir:
			return false
		case testFileSymlink:
			if runtime.GOOS == "windows" && hasSuffix && desc.target.kind != testFileDir {
				return true
			}
			return isError(*desc.target, hasSuffix)
		default:
			return hasSuffix
		}
	}
	return isError(desc, false)
}

func (desc testFileDesc) isSymlinkToDir() bool {
	if desc.kind != testFileSymlink {
		return false
	}
	if desc.ref.escapes {
		return false
	}
	if desc.finalKind() == testFileDir {
		return true
	}
	return false
}

// anySlashSuffix reports whether any of the names in the file
// (either the initial name, or a symlink target)
// include a trailing /.
func (desc testFileDesc) anySlashSuffix() bool {
	name := desc.ref.template
	if len(name) > 0 && os.IsPathSeparator(name[len(name)-1]) {
		return true
	}
	if desc.kind == testFileSymlink {
		return desc.target.anySlashSuffix()
	}
	return false
}

// anySlashSuffix reports whether the name of the file includes a trailing /.
func (desc testFileDesc) slashSuffix() bool {
	name := desc.ref.template
	if len(name) > 0 && os.IsPathSeparator(name[len(name)-1]) {
		return true
	}
	return false
}

// create creates the file(s) for this descriptor.
//
// dir is the test root directory.
// base is the base name of the file we will open within the root.
// (If there are symlinks, base is the start of the symlink chain.)
//
// Tests may create, delete, or move files, which makes it useful to have a way to identify
// and track the files that existed at the start of the test. The token parameter identifies
// which file we're creating. When symlinks are involved, the token is used in creating the
// final, non-symlink file.
func (desc testFileDesc) create(t *testing.T, dir, base, token string) (fi os.FileInfo) {
	path := filepath.Join(dir, base)
	switch desc.kind {
	case testFileAbsent:
		// File does not exist.
	case testFileFile:
		// Regular file. We use the token as the file contents.
		if err := os.WriteFile(path, []byte(token), 0o666); err != nil {
			t.Fatal(err)
		}
	case testFileDir:
		// Directory. We create a subdir within the directory named "c_"+token.
		// (The "c_" prefix is to distinguish this subdir from any files that may
		// have the same name as the token.)
		if err := os.Mkdir(path, 0o777); err != nil {
			t.Fatal(err)
		}
	case testFileSymlink:
		// Symlink. We create a symlink target named "s_"+base.
		if runtime.GOOS == "plan9" {
			t.Skip("symlinks not supported on " + runtime.GOOS)
		}
		linktarget := desc.target.ref.path(dir, "s_"+base)
		if runtime.GOOS == "wasip1" && filepath.IsAbs(linktarget) {
			t.Skip("absolute link targets not supported on " + runtime.GOOS)
		}
		fi = desc.target.create(t, dir, "s_"+base, token)
		if err := os.Symlink(linktarget, path); err != nil {
			t.Fatal(err)
		}
	default:
		t.Fatalf("can't create file of kind: %v", desc.kind)
	}
	if desc.kind == testFileFile || desc.kind == testFileDir {
		var err error
		fi, err = os.Lstat(path)
		if err != nil {
			t.Fatal(err)
		}
	}
	return fi
}

// testRootDescribeFile returns a string identifying a file.
//
// It returns "" if f is nil.
// It returns "source" or "target" if f is the source or target file in the test.
// Otherwise, it returns "unknown file".
func (test *rootMultiTest) describeFile(t *testing.T, f *os.File) string {
	if f == nil {
		return ""
	}
	fi, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	switch {
	case os.SameFile(fi, test.sourceInfo):
		return "source"
	case os.SameFile(fi, test.targetInfo):
		return "target"
	default:
		return "unknown file"
	}
}

// dirTreeContents returns a description of the contents of directory.
// For example:
//
//	drwxrwxrwx dir/
//	-rw-rw-rw- dir/file "file contents"
//	Lrw-rw-rw- symlink => dir/file
func dirTreeContents(t *testing.T, dir string) (contents []string) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	fs.WalkDir(root.FS(), ".", func(path string, d fs.DirEntry, err error) error {
		if path == "." {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			t.Fatal(err)
		}
		ent := info.Mode().String() + " " + path
		switch d.Type() {
		case fs.ModeDir:
			ent += "/"
		case fs.ModeSymlink:
			target, err := root.Readlink(path)
			if err != nil {
				t.Fatal(err)
			}
			if filepath.IsAbs(target) {
				relPath, err := filepath.Rel(dir, target)
				if err == nil && filepath.IsLocal(relPath) {
					target = "/.../" + relPath
				}
			}
			ent += " => " + target
		default:
			f, err := root.Open(path)
			if err != nil {
				ent += " (unreadable)"
			} else {
				content, err := io.ReadAll(f)
				if err != nil {
					t.Fatal(err)
				}
				ent += fmt.Sprintf(" %q", content)
			}
		}
		contents = append(contents, ent)
		return nil
	})
	return contents
}

// TestRootMultiOpen tests os.Root.Open.
//
// This also serves as a prototypical example of using rootMultiTest
// (see also the doc comment on rootMultiTest above).
func TestRootMultiOpen(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		// This function will be run many times, with different inputs:
		//   - in and out of a Root
		//   - opening a file, directory, symlink, or nothing at all
		//   - opening various names: target, DIR/../target, /abs/path/to/target, etc.
		//
		// The test function should perform the requested operation
		// (for example: open "target" in a Root),
		// verify that the result is consistent with expectations,
		// and then return a description of the result.
		//
		// The returned description is used to validate consistent behavior
		// between operations in and out of a Root.
		var open = os.Open
		if test.root != nil {
			open = test.root.Open
		}

		test.setOp("Open(%q)", test.targetPath) // test's operation, for errors
		f, gotErr := open(test.targetPath)
		if gotErr == nil {
			defer f.Close()
		}

		// testRootDescribeFile returns a string identifying a file.
		//
		// This is always "source" or "target" for the source/target files in a test,
		// or "" if f is nil.
		// (Note that most tests use only a target file, no source.)
		got := test.describeFile(t, f)

		switch {
		case test.root != nil && test.target.escapes():
			// The operation escapes the root.
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.finalKind() == testFileAbsent:
			// The file does not exist ("absent").
			test.wantError(t, gotErr, errAny)
		case test.target.anySlashSuffix():
			// The file name or a symlink target contain a trailing slash.
			// Trailing slashes are handled differently on different platforms,
			// so we won't try to assert an outcome when they are present.
			// runRootMultiTest will verify that root.Open and os.Open
			// produce consistent results.
		default:
			// We should have successfully opened the file.
			test.wantError(t, gotErr, nil)
			if want := "target"; got != want {
				t.Fatalf("opened file %q, want %q", got, want)
			}
		}

		// Return the name of the file opened (possibly "" for nothing) and the error.
		// runRootMultiTest will compare the results for in-a-root and out-of-a-root
		// to validate that they are the same.
		return got, gotErr
	})
}

func TestRootMultiChmod(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("Chmod not supported on " + runtime.GOOS)
	}
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var (
			chmod = os.Chmod
			stat  = os.Stat
			lstat = os.Lstat
		)
		if test.root != nil {
			chmod = test.root.Chmod
			stat = test.root.Stat
			lstat = test.root.Lstat
		}

		// Using the wrong mode here can cause problems during test cleanup,
		// if we leave a temp dir with a mode that prevents listing or removing
		// its contents.
		//
		// read+execute permissions let us list directory contents,
		// and we restore writability before deleting the temp dir.
		wantMode := os.FileMode(0o500) // readable, executable
		if runtime.GOOS == "windows" {
			// On Windows, the only modes we support are the default (777/rwx)
			// or read-only (444/r-x). Making a directory read-only doesn't prevent
			// listing its contents, so we can use 444 here.
			wantMode = 0o444 // readable
		}
		t.Cleanup(func() {
			chmod(test.targetPath, 0o700)
		})

		test.setOp("Chmod(%q, %o)", test.targetPath, wantMode)
		gotErr := chmod(test.targetPath, wantMode)

		escapes := test.target.escapes()
		targetKind := test.target.finalKind()
		if runtime.GOOS == "windows" {
			// On Windows, Chmod("symlink") affects the link, not its target.
			// See issue #71492.
			stat = lstat
			escapes = test.target.ref.escapes
			targetKind = test.target.kind
		}

		var gotMode fs.FileMode
		switch {
		case test.root != nil && escapes:
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case targetKind == testFileAbsent:
			test.wantError(t, gotErr, errAny)
		case test.target.anySlashSuffix():
			// Don't expect anything, just be consistent with the OS.
		default:
			test.wantError(t, gotErr, nil)

			fi, err := stat(test.targetPath)
			if err != nil {
				t.Fatalf("could not stat target: %v", err)
			}
			if runtime.GOOS == "windows" && !fi.Mode().IsRegular() {
				// See issue #71492.
				break
			}

			gotMode = fi.Mode() & fs.ModePerm
			if gotMode != wantMode {
				t.Fatalf("file %q:\ngot mode:  %v\nwant mode: %v", test.targetPath, gotMode, wantMode)
			}
		}

		if runtime.GOOS == "windows" && test.root == nil && gotErr != nil {
			// On Windows, os.Chmod calls GetFileAttributes on the target.
			// This seems to fail in a number of situations where the os.Root
			// chmod path works. For now, just skip the consistency check
			// when os.Chmod fails.
			return "", errSkipRootConsistencyCheck
		}

		return gotMode.String(), gotErr
	})
}

func TestRootMultiCreate(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var create = os.Create
		if test.root != nil {
			create = test.root.Create
		}

		test.setOp("Create(%q)", test.targetPath) // test's operation, for errors
		f, gotErr := create(test.targetPath)
		if gotErr == nil {
			defer f.Close()
		}

		switch {
		case test.target.isError():
			test.wantError(t, gotErr, errAny)
		case runtime.GOOS == "windows" && test.target.isSymlinkToDir():
			// The error here is because the link is a Windows directory link,
			// not because the link target is a directory.
			test.wantError(t, gotErr, errAny)
		case test.root != nil && test.target.escapes():
			// The operation escapes the root.
			test.wantError(t, gotErr, os.ErrPathEscapes)
		default:
		}

		return "", gotErr
	})
}

func TestRootMultiLink(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		switch os.Getenv("GOWASIRUNTIME") {
		case "", "wasmtime":
			// This test fails when run with wasmtime, because os.RemoveAll fails
			// to remove the test tempdir.
			t.Skip("test seems to tickle a wasmtime bug")
		}
	}
	testenv.MustHaveLink(t)
	runRootMultiTest2(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var (
			rename = os.Link
		)
		if test.root != nil {
			rename = test.root.Link
		}

		test.setOp("Link(%q, %q)", test.sourcePath, test.targetPath)
		gotErr := rename(test.sourcePath, test.targetPath)

		switch {
		case test.root != nil && test.source.lescapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.source.lfinalKind() == testFileAbsent:
			test.wantError(t, gotErr, errAny)
		case test.source.kind == testFileSymlink:
			// os.Link(old, new) may or may not deference old when it is a symlink.
			// POSIX says that link(2) should deference the source, but implementations
			// are inconsistent.
			return "", errSkipRootConsistencyCheck
		case test.source.slashSuffix() && test.source.lfinalKind() != testFileDir:
			test.wantError(t, gotErr, errAny)
		}
		return "", gotErr
	})
}

func TestRootMultiLstat(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var (
			lstat = os.Lstat
		)
		if test.root != nil {
			lstat = test.root.Lstat
		}

		test.setOp("Lstat(%q)", test.targetPath)
		gotStat, gotErr := lstat(test.targetPath)

		result := ""
		if gotStat != nil {
			result = gotStat.Mode().String()
		}

		escapes := test.target.lescapes()
		finalKind := test.target.lfinalKind()
		if runtime.GOOS == "windows" && test.target.ref.hasSlashSuffix() {
			// When the target of lstat has a trailing slash,
			// Windows follows it.
			escapes = test.target.escapes()
			finalKind = test.target.finalKind()
		}

		switch {
		case test.root != nil && escapes:
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.kind == testFileAbsent:
			// Target does not exist.
			test.wantError(t, gotErr, errAny)
		case finalKind == testFileSymlink:
			test.wantError(t, gotErr, nil)
			if got, want := gotStat.Mode().Type(), fs.ModeSymlink; got != want {
				test.errorf(t, "got mode %v, want %v", got, want)
			}
		case gotErr != nil:
		default:
			if !os.SameFile(gotStat, test.targetInfo) {
				test.errorf(t, "stat result is not for target file; want it to be")
			}
		}

		return result, gotErr
	})
}

func TestRootMultiMkdir(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var (
			mkdir = os.Mkdir
			stat  = os.Stat
		)
		if test.root != nil {
			mkdir = test.root.Mkdir
			stat = test.root.Stat
		}

		test.setOp("Mkdir(%q, 0o777)", test.targetPath)
		gotErr := mkdir(test.targetPath, 0o777)

		switch {
		case test.root != nil && test.target.ref.escapes:
			// "mkdir ../target", or equivalent escaping path.
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.slashSuffix() && test.target.kind == testFileSymlink:
			// "mkdir symlink/", inconsistent behavior across platforms
			// as to whether this follows the symlink or not.
			//
			// If the symlink escapes, this needs to be some kind of error though.
			if test.root != nil && test.target.escapes() {
				test.wantError(t, gotErr, errAny)
			}
			if runtime.GOOS == "openbsd" {
				// Known inconsistency: OpenBSD doesn't resolve the final
				// symlink when creating a directory.
				return "", errSkipRootConsistencyCheck
			}
		case test.target.kind != testFileAbsent:
			// "mkdir target", where target exists.
			test.wantError(t, gotErr, errAny)
		default:
			test.wantError(t, gotErr, nil)
			fi, err := stat(test.targetPath)
			if err != nil {
				t.Fatalf("could not stat target: %v", err)
			}
			if !fi.IsDir() {
				t.Fatalf("%q: not a directory, expected it to be", test.targetPath)
			}
		}
		return "", gotErr
	})
}

func TestRootMultiMkdirAllShallow(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		return testRootMultiMkdirAll(t, test, test.targetPath)
	})
}

func TestRootMultiMkdirAllDeep(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		targetPath := test.targetPath
		if len(targetPath) > 0 && os.IsPathSeparator(targetPath[len(targetPath)-1]) {
			targetPath += "a/b/"
		} else {
			targetPath += "/a/b"
		}
		return testRootMultiMkdirAll(t, test, targetPath)
	})
}

func testRootMultiMkdirAll(t *testing.T, test *rootMultiTest, targetPath string) (string, error) {
	var mkdirAll = os.MkdirAll
	if test.root != nil {
		mkdirAll = test.root.MkdirAll
	}

	test.setOp("MkdirAll(%q, 0o777)", targetPath)
	gotErr := mkdirAll(targetPath, 0o777)

	switch {
	case test.root != nil && test.target.lescapes():
		// "mkdir ../target", or equivalent escaping path.
		test.wantError(t, gotErr, os.ErrPathEscapes)
	case test.root != nil && test.target.escapes():
		// "mkdir ../target", or equivalent escaping path.
		test.wantError(t, gotErr, errAny)
		return "", errSkipRootConsistencyCheck
	case test.root != nil && test.target.kind == testFileSymlink && test.target.target.kind == testFileAbsent && targetPath != test.targetPath:
		// A minor inconsistency between Root.MkdirAll and os.MkdirAll:
		// When an intermediate component of the tree being constructed is a
		// dangling symlink, Root.MkdirAll will follow the symlink and create
		// its target directory, while os.MkdirAll will fail with an error.
		return "", errSkipRootConsistencyCheck
	default:
	}
	return "", gotErr
}

func TestRootMultiRename(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		switch os.Getenv("GOWASIRUNTIME") {
		case "", "wasmtime":
			// This test fails when run with wasmtime, because os.RemoveAll fails
			// to remove the test tempdir.
			t.Skip("test seems to tickle a wasmtime bug")
		}
	}
	runRootMultiTest2(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var (
			rename = os.Rename
		)
		if test.root != nil {
			rename = test.root.Rename
		}

		// TODO: target directory (if any) should be empty

		test.setOp("Rename(%q, %q)", test.sourcePath, test.targetPath)
		gotErr := rename(test.sourcePath, test.targetPath)

		if runtime.GOOS == "windows" &&
			(test.source.finalKind() != test.target.finalKind() || test.source.kind == testFileSymlink || test.target.kind == testFileSymlink) {
			// os.Rename on Windows is implemented using MoveFileEx,
			// while Root.Rename is implemented using NtSetInformationFileEx
			// with an explicit request for POSIX semantics.
			//
			// This means the two do not behave the same when renaming
			// a file onto a directory or vice-versa.
			//
			// We should make this consistent, but for now just skip
			// the consistency checks in this case.
			return "", errSkipRootConsistencyCheck
		}

		switch {
		case test.root != nil && test.source.lescapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.source.lfinalKind() == testFileAbsent:
			test.wantError(t, gotErr, errAny)
		case test.source.slashSuffix() && test.source.lfinalKind() != testFileDir && runtime.GOOS != "js":
			test.wantError(t, gotErr, errAny)
		case test.root != nil && test.target.lescapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case runtime.GOOS == "plan9":
			// Plan9 rename behaves differently.
			// Just rely on consistency checks.
		case test.target.lfinalKind() == testFileDir:
			// POSIX rename() will replace an empty target directory,
			// but os.Rename will not.
			test.wantError(t, gotErr, errAny)
		case test.source.lfinalKind() == testFileDir && test.target.lfinalKind() != testFileAbsent:
			test.wantError(t, gotErr, errAny)
		case test.source.anySlashSuffix() || test.target.anySlashSuffix():
			if runtime.GOOS == "openbsd" {
				// Known inconsistency: OpenBSD doesn't resolve the final
				// symlink when creating a directory.
				return "", errSkipRootConsistencyCheck
			}
		default:
			test.wantError(t, gotErr, nil)
			// TODO: check that the file is in its new location
		}

		if runtime.GOOS == "linux" && (test.source.slashSuffix() || test.target.slashSuffix()) {
			return "", errSkipRootConsistencyCheck
		}

		return "", gotErr
	})
}

func TestRootMultiReadFile(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var readFile = os.ReadFile
		if test.root != nil {
			readFile = test.root.ReadFile
		}

		test.setOp("ReadFile(%q)", test.targetPath)
		data, gotErr := readFile(test.targetPath)
		var got string
		if gotErr == nil {
			got = string(data)
		}

		switch {
		case test.root != nil && test.target.escapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.finalKind() == testFileAbsent:
			test.wantError(t, gotErr, errAny)
		case runtime.GOOS == "plan9":
			// Plan9 lets you read from directories.
			// Just rely on consistency checks.
		case runtime.GOOS == "netbsd":
			// See https://go.dev/issue/80322:
			// NetBSD builder appears to be succeeding on read-from-dir as well.
			return "", gotErr
		case test.target.finalKind() == testFileDir:
			test.wantError(t, gotErr, errAny)
		case test.target.anySlashSuffix():
			// Trailing slashes are handled differently on different platforms,
			// so we won't try to assert an outcome when they are present.
			// runRootMultiTest will verify that root.ReadFile and os.ReadFile
			// produce consistent results.
		default:
			test.wantError(t, gotErr, nil)
			if want := "target"; got != want {
				t.Fatalf("read file content %q, want %q", got, want)
			}
		}

		return got, gotErr
	})
}

func TestRootMultiStat(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var stat = os.Stat
		if test.root != nil {
			stat = test.root.Stat
		}

		test.setOp("Stat(%q)", test.targetPath)
		gotStat, gotErr := stat(test.targetPath)

		switch {
		case test.target.isError():
			test.wantError(t, gotErr, errAny)
		case test.root != nil && test.target.escapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.finalKind() == testFileAbsent:
			test.wantError(t, gotErr, errAny)
		case test.target.anySlashSuffix():
		default:
			test.wantError(t, gotErr, nil)
			if !os.SameFile(gotStat, test.targetInfo) {
				test.errorf(t, "stat result is not for target file; want it to be")
			}
		}
		return "", gotErr
	})
}

func TestRootMultiRemove(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var remove = os.Remove
		if test.root != nil {
			remove = test.root.Remove
		}

		test.setOp("Remove(%q)", test.targetPath)
		gotErr := remove(test.targetPath)

		switch {
		case test.root != nil && test.target.lescapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.kind == testFileAbsent:
			test.wantError(t, gotErr, errAny)
		case test.target.anySlashSuffix():
			if runtime.GOOS == "linux" {
				// Linux treats rmdir("symlink/") as an error when
				// "symlink" is a symlink to a directory.
				// Root.Remove prefers the POSIX interpretation
				// of resolving the symlink.
				return "", errSkipRootConsistencyCheck
			}
		default:
			test.wantError(t, gotErr, nil)
		}
		return "", gotErr
	})
}

func TestRootMultiRemoveAll(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var removeAll = os.RemoveAll
		if test.root != nil {
			removeAll = test.root.RemoveAll
		}

		test.setOp("RemoveAll(%q)", test.targetPath)
		gotErr := removeAll(test.targetPath)

		switch {
		case test.root != nil && test.target.ref.escapes:
			// This is only checking target.ref.escapes,
			// not target.lescapes(), because RemoveAll strips
			// terminal slashes.
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.anySlashSuffix():
			// We are inconsistent on some platforms on whether
			// RemoveAll("symlink/") removes the link or the link target.
			// Something worth addressing, but for now skip the check.
			return "", errSkipRootConsistencyCheck
		default:
			test.wantError(t, gotErr, nil)
		}
		return "", gotErr
	})
}

func TestRootMultiChtimes(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var chtimes = os.Chtimes
		if test.root != nil {
			chtimes = test.root.Chtimes
		}

		now := time.Now()
		test.setOp("Chtimes(%q, %v, %v)", test.targetPath, now, now)
		gotErr := chtimes(test.targetPath, now, now)

		switch {
		case test.target.isError():
			test.wantError(t, gotErr, errAny)
		case test.root != nil && test.target.escapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.finalKind() == testFileAbsent:
			test.wantError(t, gotErr, errAny)
		case test.target.anySlashSuffix():
		default:
			test.wantError(t, gotErr, nil)
		}
		return "", gotErr
	})
}

func TestRootMultiReadlink(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var readlink = os.Readlink
		if test.root != nil {
			readlink = test.root.Readlink
		}

		test.setOp("Readlink(%q)", test.targetPath)
		got, gotErr := readlink(test.targetPath)
		if suffix, ok := strings.CutPrefix(got, test.dir); ok {
			// Replace absolute path prefix with /.../
			got = "/..." + suffix
		}

		switch {
		case test.root != nil && test.target.lescapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.kind != testFileSymlink:
			test.wantError(t, gotErr, errAny)
		case test.target.anySlashSuffix():
		default:
			test.wantError(t, gotErr, nil)
		}
		return got, gotErr
	})
}

func TestRootMultiWriteFile(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var writeFile = os.WriteFile
		if test.root != nil {
			writeFile = test.root.WriteFile
		}

		test.setOp("WriteFile(%q, ...)", test.targetPath)
		gotErr := writeFile(test.targetPath, []byte("data"), 0o666)

		switch {
		case test.target.isError():
			test.wantError(t, gotErr, errAny)
		case runtime.GOOS == "windows" && test.target.isSymlinkToDir():
			test.wantError(t, gotErr, errAny)
		case test.root != nil && test.target.escapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.finalKind() == testFileDir:
			test.wantError(t, gotErr, errAny)
		case test.target.anySlashSuffix():
		default:
			test.wantError(t, gotErr, nil)
		}
		return "", gotErr
	})
}

func TestRootMultiOpenFile(t *testing.T) {
	runRootMultiTest(t, func(t *testing.T, test *rootMultiTest) (string, error) {
		var openFile = os.OpenFile
		if test.root != nil {
			openFile = test.root.OpenFile
		}

		test.setOp("OpenFile(%q, O_RDONLY, 0)", test.targetPath)
		f, gotErr := openFile(test.targetPath, os.O_RDONLY, 0)
		if gotErr == nil {
			defer f.Close()
		}

		got := test.describeFile(t, f)

		switch {
		case test.root != nil && test.target.escapes():
			test.wantError(t, gotErr, os.ErrPathEscapes)
		case test.target.finalKind() == testFileAbsent:
			test.wantError(t, gotErr, errAny)
		case test.target.anySlashSuffix():
		default:
			test.wantError(t, gotErr, nil)
			if want := "target"; got != want {
				t.Fatalf("opened file %q, want %q", got, want)
			}
		}

		return got, gotErr
	})
}
