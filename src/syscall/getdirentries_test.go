// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd netbsd openbsd

package syscall_test

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

func TestGetdirentries(t *testing.T) {
	for _, count := range []int{10, 1000} {
		t.Run(fmt.Sprintf("n=%d", count), func(t *testing.T) {
			testGetdirentries(t, count)
		})
	}
}
func testGetdirentries(t *testing.T, count int) {
	if count > 100 && testing.Short() && os.Getenv("GO_BUILDER_NAME") == "" {
		t.Skip("skipping in -short mode")
	}
	d, err := os.MkdirTemp("", "getdirentries-test")
	if err != nil {
		t.Fatalf("Tempdir: %v", err)
	}
	defer os.RemoveAll(d)
	var names []string
	for i := 0; i < count; i++ {
		names = append(names, fmt.Sprintf("file%03d", i))
	}

	// Make files in the temp directory
	for _, name := range names {
		err := os.WriteFile(filepath.Join(d, name), []byte("data"), 0)
		if err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
	}

	// Read files using Getdirentries
	var names2 []string
	fd, err := syscall.Open(d, syscall.O_RDONLY, 0)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer syscall.Close(fd)
	var base uintptr
	var buf [2048]byte
	for {
		n, err := syscall.Getdirentries(fd, buf[:], &base)
		if err != nil {
			t.Fatalf("Getdirentries: %v", err)
		}
		if n == 0 {
			break
		}
		data := buf[:n]
		for len(data) > 0 {
			// If multiple Dirents are written into buf, sometimes when we reach the final one,
			// we have cap(buf) < Sizeof(Dirent). So use an appropriate slice to copy from data.
			var dirent syscall.Dirent
			copy((*[unsafe.Sizeof(dirent)]byte)(unsafe.Pointer(&dirent))[:], data)

			data = data[dirent.Reclen:]
			name := make([]byte, dirent.Namlen)
			for i := 0; i < int(dirent.Namlen); i++ {
				name[i] = byte(dirent.Name[i])
			}
			names2 = append(names2, string(name))
		}
	}

	names = append(names, ".", "..") // Getdirentries returns these also
	sort.Strings(names)
	sort.Strings(names2)
	if strings.Join(names, ":") != strings.Join(names2, ":") {
		t.Errorf("names don't match\n names: %q\nnames2: %q", names, names2)
	}
}
