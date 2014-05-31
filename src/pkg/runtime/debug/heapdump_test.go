// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"io/ioutil"
	"os"
	"runtime"
	"testing"
)

func TestWriteHeapDumpNonempty(t *testing.T) {
	if runtime.GOOS == "nacl" {
		t.Skip("WriteHeapDump is not available on NaCl.")
	}
	f, err := ioutil.TempFile("", "heapdumptest")
	if err != nil {
		t.Fatalf("TempFile failed: %v", err)
	}
	defer os.Remove(f.Name())
	defer f.Close()
	WriteHeapDump(f.Fd())
	fi, err := f.Stat()
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	const minSize = 1
	if size := fi.Size(); size < minSize {
		t.Fatalf("Heap dump size %d bytes, expected at least %d bytes", size, minSize)
	}
}
