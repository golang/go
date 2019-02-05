// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix_test

import (
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

// stringsFromByteSlice converts a sequence of attributes to a []string.
// On Darwin, each entry is a NULL-terminated string.
func stringsFromByteSlice(buf []byte) []string {
	var result []string
	off := 0
	for i, b := range buf {
		if b == 0 {
			result = append(result, string(buf[off:i]))
			off = i + 1
		}
	}
	return result
}

func TestUtimesNanoAt(t *testing.T) {
	defer chtmpdir(t)()

	symlink := "symlink1"
	os.Remove(symlink)
	err := os.Symlink("nonexisting", symlink)
	if err != nil {
		t.Fatal(err)
	}

	ts := []unix.Timespec{
		{Sec: 1111, Nsec: 2222},
		{Sec: 3333, Nsec: 4444},
	}
	err = unix.UtimesNanoAt(unix.AT_FDCWD, symlink, ts, unix.AT_SYMLINK_NOFOLLOW)
	if err != nil {
		t.Fatalf("UtimesNanoAt: %v", err)
	}

	var st unix.Stat_t
	err = unix.Lstat(symlink, &st)
	if err != nil {
		t.Fatalf("Lstat: %v", err)
	}

	// Only check Mtimespec, Atimespec might not be supported by the underlying filesystem
	expected := ts[1]
	if st.Mtimespec.Nsec == 0 {
		// Some filesystems only support 1-second time stamp resolution
		// and will always set Nsec to 0.
		expected.Nsec = 0
	}
	if st.Mtimespec != expected {
		t.Errorf("UtimesNanoAt: wrong mtime: got %v, expected %v", st.Mtimespec, expected)
	}
}
