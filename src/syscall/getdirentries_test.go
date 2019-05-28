// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd netbsd openbsd

package syscall_test

import (
	"fmt"
	"io/ioutil"
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
	d, err := ioutil.TempDir("", "getdirentries-test")
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
		err := ioutil.WriteFile(filepath.Join(d, name), []byte("data"), 0)
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
			dirent := (*syscall.Dirent)(unsafe.Pointer(&data[0]))
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
