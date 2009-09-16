// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Utility functions.

package io

import (
	"bytes";
	"os";
)

// ReadAll reads from r until an error or EOF and returns the data it read.
func ReadAll(r Reader) ([]byte, os.Error) {
	var buf bytes.Buffer;
	_, err := Copy(r, &buf);
	return buf.Bytes(), err;
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

// WriteFile writes data to a file named by filename.
// If the file does not exist, WriteFile creates it with permissions perm;
// otherwise WriteFile truncates it before writing.
func WriteFile(filename string, data []byte, perm int) os.Error {
	f, err := os.Open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, perm);
	if err != nil {
		return err;
	}
	n, err := f.Write(data);
	if err == nil && n < len(data) {
		err = ErrShortWrite;
	}
	f.Close();
	return err;
}
