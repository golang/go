// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"path/filepath"
	"runtime"
	"testing"
)

// TestMMap ensures that we can actually mmap on every supported platform.
func TestMMap(t *testing.T) {
	switch runtime.GOOS {
	default:
		t.Skip("unsupported OS")
	case "aix", "darwin", "ios", "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "windows":
	}
	dir := t.TempDir()
	filename := filepath.Join(dir, "foo.out")
	ob := NewOutBuf(nil)
	if err := ob.Open(filename); err != nil {
		t.Fatalf("error opening file: %v", err)
	}
	defer ob.Close()
	if err := ob.Mmap(1 << 20); err != nil {
		t.Errorf("error mmapping file %v", err)
	}
	if !ob.isMmapped() {
		t.Errorf("should be mmapped")
	}
}

// TestWriteLoc ensures that the math surrounding writeLoc is correct.
func TestWriteLoc(t *testing.T) {
	tests := []struct {
		bufLen          int
		off             int64
		heapLen         int
		lenToWrite      int64
		expectedHeapLen int
		writePos        int64
		addressInHeap   bool
	}{
		{100, 0, 0, 100, 0, 0, false},
		{100, 100, 0, 100, 100, 0, true},
		{10, 10, 0, 100, 100, 0, true},
		{10, 20, 10, 100, 110, 10, true},
		{0, 0, 0, 100, 100, 0, true},
	}

	for i, test := range tests {
		ob := &OutBuf{
			buf:  make([]byte, test.bufLen),
			off:  test.off,
			heap: make([]byte, test.heapLen),
		}
		pos, buf := ob.writeLoc(test.lenToWrite)
		if pos != test.writePos {
			t.Errorf("[%d] position = %d, expected %d", i, pos, test.writePos)
		}
		message := "mmapped area"
		expected := ob.buf
		if test.addressInHeap {
			message = "heap"
			expected = ob.heap
		}
		if &buf[0] != &expected[0] {
			t.Errorf("[%d] expected position to be %q", i, message)
		}
		if len(ob.heap) != test.expectedHeapLen {
			t.Errorf("[%d] expected len(ob.heap) == %d, got %d", i, test.expectedHeapLen, len(ob.heap))
		}
	}
}

func TestIsMmapped(t *testing.T) {
	tests := []struct {
		length   int
		expected bool
	}{
		{0, false},
		{1, true},
	}
	for i, test := range tests {
		ob := &OutBuf{buf: make([]byte, test.length)}
		if v := ob.isMmapped(); v != test.expected {

			t.Errorf("[%d] isMmapped == %t, expected %t", i, v, test.expected)
		}
	}
}
