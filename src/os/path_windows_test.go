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

func TestAddExtendedPrefix(t *testing.T) {
	// Test addExtendedPrefix instead of fixLongPath so the path manipulation code
	// is exercised even if long path are supported by the system, else the
	// function might not be tested at all if/when all test builders support long paths.
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal("cannot get cwd")
	}
	drive := strings.ToLower(filepath.VolumeName(cwd))
	cwd = strings.ToLower(cwd[len(drive)+1:])
	// Build a very long pathname. Paths in Go are supposed to be arbitrarily long,
	// so let's make a long path which is comfortably bigger than MAX_PATH on Windows
	// (256) and thus requires fixLongPath to be correctly interpreted in I/O syscalls.
	veryLong := "l" + strings.Repeat("o", 500) + "ng"
	for _, test := range []struct{ in, want string }{
		// Test cases use word substitutions:
		//   * "long" is replaced with a very long pathname
		//   * "c:" or "C:" are replaced with the drive of the current directory (preserving case)
		//   * "cwd" is replaced with the current directory

		// Drive Absolute
		{`C:\long\foo.txt`, `\\?\C:\long\foo.txt`},
		{`C:/long/foo.txt`, `\\?\C:\long\foo.txt`},
		{`C:\\\long///foo.txt`, `\\?\C:\long\foo.txt`},
		{`C:\long\.\foo.txt`, `\\?\C:\long\foo.txt`},
		{`C:\long\..\foo.txt`, `\\?\C:\foo.txt`},
		{`C:\long\..\..\foo.txt`, `\\?\C:\foo.txt`},

		// Drive Relative
		{`C:long\foo.txt`, `\\?\C:\cwd\long\foo.txt`},
		{`C:long/foo.txt`, `\\?\C:\cwd\long\foo.txt`},
		{`C:long///foo.txt`, `\\?\C:\cwd\long\foo.txt`},
		{`C:long\.\foo.txt`, `\\?\C:\cwd\long\foo.txt`},
		{`C:long\..\foo.txt`, `\\?\C:\cwd\foo.txt`},

		// Rooted
		{`\long\foo.txt`, `\\?\C:\long\foo.txt`},
		{`/long/foo.txt`, `\\?\C:\long\foo.txt`},
		{`\long///foo.txt`, `\\?\C:\long\foo.txt`},
		{`\long\.\foo.txt`, `\\?\C:\long\foo.txt`},
		{`\long\..\foo.txt`, `\\?\C:\foo.txt`},

		// Relative
		{`long\foo.txt`, `\\?\C:\cwd\long\foo.txt`},
		{`long/foo.txt`, `\\?\C:\cwd\long\foo.txt`},
		{`long///foo.txt`, `\\?\C:\cwd\long\foo.txt`},
		{`long\.\foo.txt`, `\\?\C:\cwd\long\foo.txt`},
		{`long\..\foo.txt`, `\\?\C:\cwd\foo.txt`},
		{`.\long\foo.txt`, `\\?\C:\cwd\long\foo.txt`},

		// UNC Absolute
		{`\\srv\share\long`, `\\?\UNC\srv\share\long`},
		{`//srv/share/long`, `\\?\UNC\srv\share\long`},
		{`/\srv/share/long`, `\\?\UNC\srv\share\long`},
		{`\\srv\share\long\`, `\\?\UNC\srv\share\long\`},
		{`\\srv\share\bar\.\long`, `\\?\UNC\srv\share\bar\long`},
		{`\\srv\share\bar\..\long`, `\\?\UNC\srv\share\long`},
		{`\\srv\share\bar\..\..\long`, `\\?\UNC\srv\share\long`}, // share name is not removed by ".."

		// Local Device
		{`\\.\C:\long\foo.txt`, `\\.\C:\long\foo.txt`},
		{`//./C:/long/foo.txt`, `\\.\C:\long\foo.txt`},
		{`/\./C:/long/foo.txt`, `\\.\C:\long\foo.txt`},
		{`\\.\C:\long///foo.txt`, `\\.\C:\long\foo.txt`},
		{`\\.\C:\long\.\foo.txt`, `\\.\C:\long\foo.txt`},
		{`\\.\C:\long\..\foo.txt`, `\\.\C:\foo.txt`},

		// Misc tests
		{`C:\short.txt`, `C:\short.txt`},
		{`C:\`, `C:\`},
		{`C:`, `C:`},
		{`\\srv\path`, `\\srv\path`},
		{`long.txt`, `\\?\C:\cwd\long.txt`},
		{`C:long.txt`, `\\?\C:\cwd\long.txt`},
		{`C:\long\.\bar\baz`, `\\?\C:\long\bar\baz`},
		{`C:long\.\bar\baz`, `\\?\C:\cwd\long\bar\baz`},
		{`C:\long\..\bar\baz`, `\\?\C:\bar\baz`},
		{`C:long\..\bar\baz`, `\\?\C:\cwd\bar\baz`},
		{`C:\long\foo\\bar\.\baz\\`, `\\?\C:\long\foo\bar\baz\`},
		{`C:\long\..`, `\\?\C:\`},
		{`C:\.\long\..\.`, `\\?\C:\`},
		{`\\?\C:\long\foo.txt`, `\\?\C:\long\foo.txt`},
		{`\\?\C:\long/foo.txt`, `\\?\C:\long/foo.txt`},
	} {
		in := strings.ReplaceAll(test.in, "long", veryLong)
		in = strings.ToLower(in)
		in = strings.ReplaceAll(in, "c:", drive)

		want := strings.ReplaceAll(test.want, "long", veryLong)
		want = strings.ToLower(want)
		want = strings.ReplaceAll(want, "c:", drive)
		want = strings.ReplaceAll(want, "cwd", cwd)

		got := os.AddExtendedPrefix(in)
		got = strings.ToLower(got)
		if got != want {
			in = strings.ReplaceAll(in, veryLong, "long")
			got = strings.ReplaceAll(got, veryLong, "long")
			want = strings.ReplaceAll(want, veryLong, "long")
			t.Errorf("addExtendedPrefix(%#q) = %#q; want %#q", in, got, want)
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

func TestRemoveAllLongPathRelative(t *testing.T) {
	// Test that RemoveAll doesn't hang with long relative paths.
	// See go.dev/issue/36375.
	tmp := t.TempDir()
	t.Chdir(tmp)
	dir := filepath.Join(tmp, "foo", "bar", strings.Repeat("a", 150), strings.Repeat("b", 150))
	err := os.MkdirAll(dir, 0755)
	if err != nil {
		t.Fatal(err)
	}
	err = os.RemoveAll("foo")
	if err != nil {
		t.Fatal(err)
	}
}

func TestRemoveAllFallback(t *testing.T) {
	windows.TestDeleteatFallback = true
	t.Cleanup(func() { windows.TestDeleteatFallback = false })

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "file1"), []byte{}, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "file2"), []byte{}, 0400); err != nil { // read-only file
		t.Fatal(err)
	}

	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}
}

func testLongPathAbs(t *testing.T, target string) {
	t.Helper()
	testWalkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			t.Error(err)
		}
		return err
	}
	if err := os.MkdirAll(target, 0777); err != nil {
		t.Fatal(err)
	}
	// Test that Walk doesn't fail with long paths.
	// See go.dev/issue/21782.
	filepath.Walk(target, testWalkFn)
	// Test that RemoveAll doesn't hang with long paths.
	// See go.dev/issue/36375.
	if err := os.RemoveAll(target); err != nil {
		t.Error(err)
	}
}

func TestLongPathAbs(t *testing.T) {
	t.Parallel()

	target := t.TempDir() + "\\" + strings.Repeat("a\\", 300)
	testLongPathAbs(t, target)
}

func TestLongPathRel(t *testing.T) {
	t.Chdir(t.TempDir())

	target := strings.Repeat("b\\", 300)
	testLongPathAbs(t, target)
}

func BenchmarkAddExtendedPrefix(b *testing.B) {
	veryLong := `C:\l` + strings.Repeat("o", 248) + "ng"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		os.AddExtendedPrefix(veryLong)
	}
}
