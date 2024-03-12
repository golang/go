// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"fmt"
	"internal/syscall/windows"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func TestFixLongPath(t *testing.T) {
	// Test fixLongPath even if long path are supported by the system,
	// else the function might not be tested at all when the test builders
	// support long paths.
	old := windows.CanUseLongPaths
	windows.CanUseLongPaths = false
	t.Cleanup(func() {
		windows.CanUseLongPaths = old
	})

	// 248 is long enough to trigger the longer-than-248 checks in
	// fixLongPath, but short enough not to make a path component
	// longer than 255, which is illegal on Windows. (which
	// doesn't really matter anyway, since this is purely a string
	// function we're testing, and it's not actually being used to
	// do a system call)
	veryLong := "l" + strings.Repeat("o", 248) + "ng"
	for _, test := range []struct{ in, want string }{
		// Short; unchanged:
		{`C:\short.txt`, `C:\short.txt`},
		{`C:\`, `C:\`},
		{`C:`, `C:`},
		// The "long" substring is replaced by a looooooong
		// string which triggers the rewriting. Except in the
		// cases below where it doesn't.
		{`C:\long\foo.txt`, `\\?\C:\long\foo.txt`},
		{`C:/long/foo.txt`, `\\?\C:\long\foo.txt`},
		{`C:\long\foo\\bar\.\baz\\`, `\\?\C:\long\foo\bar\baz\`},
		{`\\server\path\long`, `\\?\UNC\server\path\long`},
		{`long.txt`, `long.txt`},
		{`C:long.txt`, `C:long.txt`},
		{`c:\long\..\bar\baz`, `\\?\c:\bar\baz`},
		{`\\?\c:\long\foo.txt`, `\\?\c:\long\foo.txt`},
		{`\\?\c:\long/foo.txt`, `\\?\c:\long/foo.txt`},
		{`\??\c:\long/foo.txt`, `\??\c:\long/foo.txt`},
	} {
		in := strings.ReplaceAll(test.in, "long", veryLong)
		want := strings.ReplaceAll(test.want, "long", veryLong)
		if got := os.FixLongPath(in); got != want {
			got = strings.ReplaceAll(got, veryLong, "long")
			t.Errorf("fixLongPath(%#q) = %#q; want %#q", test.in, got, test.want)
		}
	}
}

func TestMkdirAllLongPath(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	path := tmpDir
	for i := 0; i < 100; i++ {
		path += `\another-path-component`
	}
	if err := os.MkdirAll(path, 0777); err != nil {
		t.Fatalf("MkdirAll(%q) failed; %v", path, err)
	}
	if err := os.RemoveAll(tmpDir); err != nil {
		t.Fatalf("RemoveAll(%q) failed; %v", tmpDir, err)
	}
}

func TestMkdirAllExtendedLength(t *testing.T) {
	t.Parallel()
	tmpDir := t.TempDir()

	const prefix = `\\?\`
	if len(tmpDir) < 4 || tmpDir[:4] != prefix {
		fullPath, err := syscall.FullPath(tmpDir)
		if err != nil {
			t.Fatalf("FullPath(%q) fails: %v", tmpDir, err)
		}
		tmpDir = prefix + fullPath
	}
	path := tmpDir + `\dir\`
	if err := os.MkdirAll(path, 0777); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", path, err)
	}

	path = path + `.\dir2`
	if err := os.MkdirAll(path, 0777); err == nil {
		t.Fatalf("MkdirAll(%q) should have failed, but did not", path)
	}
}

func TestOpenRootSlash(t *testing.T) {
	t.Parallel()

	tests := []string{
		`/`,
		`\`,
	}

	for _, test := range tests {
		dir, err := os.Open(test)
		if err != nil {
			t.Fatalf("Open(%q) failed: %v", test, err)
		}
		dir.Close()
	}
}

func testMkdirAllAtRoot(t *testing.T, root string) {
	// Create a unique-enough directory name in root.
	base := fmt.Sprintf("%s-%d", t.Name(), os.Getpid())
	path := filepath.Join(root, base)
	if err := os.MkdirAll(path, 0777); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", path, err)
	}
	// Clean up
	if err := os.RemoveAll(path); err != nil {
		t.Fatal(err)
	}
}

func TestMkdirAllExtendedLengthAtRoot(t *testing.T) {
	if testenv.Builder() == "" {
		t.Skipf("skipping non-hermetic test outside of Go builders")
	}

	const prefix = `\\?\`
	vol := filepath.VolumeName(t.TempDir()) + `\`
	if len(vol) < 4 || vol[:4] != prefix {
		vol = prefix + vol
	}
	testMkdirAllAtRoot(t, vol)
}

func TestMkdirAllVolumeNameAtRoot(t *testing.T) {
	if testenv.Builder() == "" {
		t.Skipf("skipping non-hermetic test outside of Go builders")
	}

	vol, err := syscall.UTF16PtrFromString(filepath.VolumeName(t.TempDir()) + `\`)
	if err != nil {
		t.Fatal(err)
	}
	const maxVolNameLen = 50
	var buf [maxVolNameLen]uint16
	err = windows.GetVolumeNameForVolumeMountPoint(vol, &buf[0], maxVolNameLen)
	if err != nil {
		t.Fatal(err)
	}
	volName := syscall.UTF16ToString(buf[:])
	testMkdirAllAtRoot(t, volName)
}

func BenchmarkLongPath(b *testing.B) {
	veryLong := `C:\l` + strings.Repeat("o", 248) + "ng"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		os.FixLongPath(veryLong)
	}
}
