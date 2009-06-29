// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Utility functions.

package io

import (
	"bytes";
	"io";
	"os";
)

// ReadAll reads from r until an error or EOF and returns the data it read.
func ReadAll(r Reader) ([]byte, os.Error) {
	var buf bytes.Buffer;
	n, err := Copy(r, &buf);
	return buf.Data(), err;
}

// ReadFile reads the file named by filename and returns the contents.
func ReadFile(filename string) ([]byte, os.Error) {
	f, err := os.Open(filename, os.O_RDONLY, 0);
	if err != nil {
		return nil, err;
	}
	defer f.Close();
	return ReadAll(f);
}
