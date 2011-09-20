// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bufio"
	"bytes"
	"testing"
)

func TestReadVersion(t *testing.T) {
	buf := []byte(serverVersion)
	result, ok := readVersion(bufio.NewReader(bytes.NewBuffer(buf)))
	if !ok {
		t.Error("readVersion didn't read version correctly")
	}
	if !bytes.Equal(buf[:len(buf)-2], result) {
		t.Error("version read did not match expected")
	}
}

func TestReadVersionTooLong(t *testing.T) {
	buf := make([]byte, maxVersionStringBytes+1)
	if _, ok := readVersion(bufio.NewReader(bytes.NewBuffer(buf))); ok {
		t.Errorf("readVersion consumed %d bytes without error", len(buf))
	}
}

func TestReadVersionWithoutCRLF(t *testing.T) {
	buf := []byte(serverVersion)
	buf = buf[:len(buf)-1]
	if _, ok := readVersion(bufio.NewReader(bytes.NewBuffer(buf))); ok {
		t.Error("readVersion did not notice \\n was missing")
	}
}
