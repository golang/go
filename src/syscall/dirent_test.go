// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package syscall_test

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

func TestDirent(t *testing.T) {
	const (
		direntBufSize   = 2048 // arbitrary? See https://go.dev/issue/37323.
		filenameMinSize = 11
	)

	d := t.TempDir()
	t.Logf("tmpdir: %s", d)

	for i, c := range []byte("0123456789") {
		name := string(bytes.Repeat([]byte{c}, filenameMinSize+i))
		err := os.WriteFile(filepath.Join(d, name), nil, 0644)
		if err != nil {
			t.Fatalf("writefile: %v", err)
		}
	}

	names := make([]string, 0, 10)

	fd, err := syscall.Open(d, syscall.O_RDONLY, 0)
	if err != nil {
		t.Fatalf("syscall.open: %v", err)
	}
	defer syscall.Close(fd)

	buf := bytes.Repeat([]byte{0xCD}, direntBufSize)
	for {
		n, err := syscall.ReadDirent(fd, buf)
		if err == syscall.EINVAL {
			// On linux, 'man getdents64' says that EINVAL indicates “result buffer is too small”.
			// Try a bigger buffer.
			t.Logf("ReadDirent: %v; retrying with larger buffer", err)
			buf = bytes.Repeat([]byte{0xCD}, len(buf)*2)
			continue
		}
		if err != nil {
			t.Fatalf("syscall.readdir: %v", err)
		}
		t.Logf("ReadDirent: read %d bytes", n)
		if n == 0 {
			break
		}

		var consumed, count int
		consumed, count, names = syscall.ParseDirent(buf[:n], -1, names)
		t.Logf("ParseDirent: %d new name(s)", count)
		if consumed != n {
			t.Fatalf("ParseDirent: consumed %d bytes; expected %d", consumed, n)
		}
	}

	sort.Strings(names)
	t.Logf("names: %q", names)

	if len(names) != 10 {
		t.Errorf("got %d names; expected 10", len(names))
	}
	for i, name := range names {
		ord, err := strconv.Atoi(name[:1])
		if err != nil {
			t.Fatalf("names[%d] is non-integer %q: %v", i, names[i], err)
		}
		if expected := string(strings.Repeat(name[:1], filenameMinSize+ord)); name != expected {
			t.Errorf("names[%d] is %q (len %d); expected %q (len %d)", i, name, len(name), expected, len(expected))
		}
	}
}

func TestDirentRepeat(t *testing.T) {
	const N = 100
	// Note: the size of the buffer is small enough that the loop
	// below will need to execute multiple times. See issue #31368.
	size := N * unsafe.Offsetof(syscall.Dirent{}.Name) / 4
	if runtime.GOOS == "freebsd" || runtime.GOOS == "netbsd" {
		if size < 1024 {
			size = 1024 // DIRBLKSIZ, see issue 31403.
		}
		if runtime.GOOS == "freebsd" {
			t.Skip("need to fix issue 31416 first")
		}
	}

	// Make a directory containing N files
	d := t.TempDir()

	var files []string
	for i := 0; i < N; i++ {
		files = append(files, fmt.Sprintf("file%d", i))
	}
	for _, file := range files {
		err := os.WriteFile(filepath.Join(d, file), []byte("contents"), 0644)
		if err != nil {
			t.Fatalf("writefile: %v", err)
		}
	}

	// Read the directory entries using ReadDirent.
	fd, err := syscall.Open(d, syscall.O_RDONLY, 0)
	if err != nil {
		t.Fatalf("syscall.open: %v", err)
	}
	defer syscall.Close(fd)
	var files2 []string
	for {
		buf := make([]byte, size)
		n, err := syscall.ReadDirent(fd, buf)
		if err != nil {
			t.Fatalf("syscall.readdir: %v", err)
		}
		if n == 0 {
			break
		}
		buf = buf[:n]
		for len(buf) > 0 {
			var consumed int
			consumed, _, files2 = syscall.ParseDirent(buf, -1, files2)
			buf = buf[consumed:]
		}
	}

	// Check results
	sort.Strings(files)
	sort.Strings(files2)
	if strings.Join(files, "|") != strings.Join(files2, "|") {
		t.Errorf("bad file list: want\n%q\ngot\n%q", files, files2)
	}
}
