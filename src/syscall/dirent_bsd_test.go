// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin,!arm,!arm64 dragonfly freebsd netbsd openbsd

package syscall_test

import (
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

func TestDirent(t *testing.T) {
	const (
		direntBufSize   = 2048
		filenameMinSize = 11
	)

	d, err := ioutil.TempDir("", "dirent-test")
	if err != nil {
		t.Fatalf("tempdir: %v", err)
	}
	defer os.RemoveAll(d)
	t.Logf("tmpdir: %s", d)

	for i, c := range []byte("0123456789") {
		name := string(bytes.Repeat([]byte{c}, filenameMinSize+i))
		err = ioutil.WriteFile(filepath.Join(d, name), nil, 0644)
		if err != nil {
			t.Fatalf("writefile: %v", err)
		}
	}

	buf := bytes.Repeat([]byte("DEADBEAF"), direntBufSize/8)
	fd, err := syscall.Open(d, syscall.O_RDONLY, 0)
	defer syscall.Close(fd)
	if err != nil {
		t.Fatalf("syscall.open: %v", err)
	}
	n, err := syscall.ReadDirent(fd, buf)
	if err != nil {
		t.Fatalf("syscall.readdir: %v", err)
	}
	buf = buf[:n]

	names := make([]string, 0, 10)
	for len(buf) > 0 {
		var bc int
		bc, _, names = syscall.ParseDirent(buf, -1, names)
		buf = buf[bc:]
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
