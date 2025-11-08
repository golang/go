// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package os_test

import (
	"errors"
	"fmt"
	"internal/syscall/windows"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"unsafe"
)

// Verify that Root.Open rejects Windows reserved names.
func TestRootWindowsDeviceNames(t *testing.T) {
	r, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if f, err := r.Open("NUL"); err == nil {
		t.Errorf(`r.Open("NUL") succeeded; want error"`)
		f.Close()
	}
}

// Verify that Root.Open is case-insensitive.
// (The wrong options to NtOpenFile could make operations case-sensitive,
// so this is worth checking.)
func TestRootWindowsCaseInsensitivity(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "file"), nil, 0666); err != nil {
		t.Fatal(err)
	}
	r, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	f, err := r.Open("FILE")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	if err := r.Remove("FILE"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "file")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("os.Stat(file) after deletion: %v, want ErrNotFound", err)
	}
}

// TestRootSymlinkRelativity tests that symlinks created using Root.Symlink have the
// same SYMLINK_FLAG_RELATIVE value as ones creates using os.Symlink.
func TestRootSymlinkRelativity(t *testing.T) {
	dir := t.TempDir()
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	for i, test := range []struct {
		name   string
		target string
	}{{
		name:   "relative",
		target: `foo`,
	}, {
		name:   "absolute",
		target: `C:\foo`,
	}, {
		name:   "current working directory-relative",
		target: `C:foo`,
	}, {
		name:   "root-relative",
		target: `\foo`,
	}, {
		name:   "question prefix",
		target: `\\?\foo`,
	}, {
		name:   "relative with dot dot",
		target: `a\..\b`, // could be cleaned (but isn't)
	}} {
		t.Run(test.name, func(t *testing.T) {
			name := fmt.Sprintf("symlink_%v", i)
			if err := os.Symlink(test.target, filepath.Join(dir, name)); err != nil {
				t.Fatal(err)
			}
			if err := root.Symlink(test.target, name+"_at"); err != nil {
				t.Fatal(err)
			}

			osRDB, err := readSymlinkReparseData(filepath.Join(dir, name))
			if err != nil {
				t.Fatal(err)
			}
			rootRDB, err := readSymlinkReparseData(filepath.Join(dir, name+"_at"))
			if err != nil {
				t.Fatal(err)
			}
			if osRDB.Flags != rootRDB.Flags {
				t.Errorf("symlink target %q: Symlink flags = %x, Root.Symlink flags = %x", test.target, osRDB.Flags, rootRDB.Flags)
			}

			// Compare the link target.
			// os.Symlink converts current working directory-relative links
			// such as c:foo into absolute links.
			osTarget, err := os.Readlink(filepath.Join(dir, name))
			if err != nil {
				t.Fatal(err)
			}
			rootTarget, err := os.Readlink(filepath.Join(dir, name+"_at"))
			if err != nil {
				t.Fatal(err)
			}
			if osTarget != rootTarget {
				t.Errorf("symlink created with target %q: Symlink target = %q, Root.Symlink target = %q", test.target, osTarget, rootTarget)
			}
		})
	}
}

func readSymlinkReparseData(name string) (*windows.SymbolicLinkReparseBuffer, error) {
	nameu16, err := syscall.UTF16FromString(name)
	if err != nil {
		return nil, err
	}
	h, err := syscall.CreateFile(&nameu16[0], syscall.GENERIC_READ, 0, nil, syscall.OPEN_EXISTING,
		syscall.FILE_FLAG_OPEN_REPARSE_POINT|syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return nil, err
	}
	defer syscall.CloseHandle(h)

	var rdbbuf [syscall.MAXIMUM_REPARSE_DATA_BUFFER_SIZE]byte
	var bytesReturned uint32
	err = syscall.DeviceIoControl(h, syscall.FSCTL_GET_REPARSE_POINT, nil, 0, &rdbbuf[0], uint32(len(rdbbuf)), &bytesReturned, nil)
	if err != nil {
		return nil, err
	}

	rdb := (*windows.REPARSE_DATA_BUFFER)(unsafe.Pointer(&rdbbuf[0]))
	if rdb.ReparseTag != syscall.IO_REPARSE_TAG_SYMLINK {
		return nil, fmt.Errorf("%q: not a symlink", name)
	}

	bufoff := unsafe.Offsetof(rdb.DUMMYUNIONNAME)
	symlinkBuf := (*windows.SymbolicLinkReparseBuffer)(unsafe.Pointer(&rdbbuf[bufoff]))

	return symlinkBuf, nil
}

// TestRootSymlinkToDirectory tests that Root.Symlink creates directory links
// when the target is a directory contained within the root.
func TestRootSymlinkToDirectory(t *testing.T) {
	dir := t.TempDir()
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	if err := os.Mkdir(filepath.Join(dir, "dir"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "file"), nil, 0666); err != nil {
		t.Fatal(err)
	}

	dir2 := t.TempDir()

	for i, test := range []struct {
		name    string
		target  string
		wantDir bool
	}{{
		name:    "directory outside root",
		target:  dir2,
		wantDir: false,
	}, {
		name:    "directory inside root",
		target:  "dir",
		wantDir: true,
	}, {
		name:    "file inside root",
		target:  "file",
		wantDir: false,
	}, {
		name:    "nonexistent inside root",
		target:  "nonexistent",
		wantDir: false,
	}} {
		t.Run(test.name, func(t *testing.T) {
			name := fmt.Sprintf("symlink_%v", i)
			if err := root.Symlink(test.target, name); err != nil {
				t.Fatal(err)
			}

			// Lstat strips the directory mode bit from reparse points,
			// so we need to use GetFileInformationByHandle directly to
			// determine if this is a directory link.
			nameu16, err := syscall.UTF16PtrFromString(filepath.Join(dir, name))
			if err != nil {
				t.Fatal(err)
			}
			h, err := syscall.CreateFile(nameu16, 0, 0, nil, syscall.OPEN_EXISTING,
				syscall.FILE_FLAG_OPEN_REPARSE_POINT|syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer syscall.CloseHandle(h)
			var fi syscall.ByHandleFileInformation
			if err := syscall.GetFileInformationByHandle(h, &fi); err != nil {
				t.Fatal(err)
			}
			gotDir := fi.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY != 0

			if got, want := gotDir, test.wantDir; got != want {
				t.Errorf("link target %q: isDir = %v, want %v", test.target, got, want)
			}
		})
	}
}

func TestRootOpenFileTruncateNamedPipe(t *testing.T) {
	t.Parallel()
	name := pipeName()
	pipe := newBytePipe(t, name, false)
	defer pipe.Close()

	root, err := os.OpenRoot(filepath.Dir(name))
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	f, err := root.OpenFile(filepath.Base(name), os.O_TRUNC|os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
}
