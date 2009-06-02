// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Utility functions.

package io

import (
	"io";
	"os";
)


// ReadFile reads the file named by filename and returns
// its contents if successful.
//
func ReadFile(filename string) ([]byte, os.Error) {
	f, err := os.Open(filename, os.O_RDONLY, 0);
	if err != nil {
		return nil, err;
	}
	var b io.ByteBuffer;
	_, err := io.Copy(f, &b);
	f.Close();
	return b.Data(), err;
}
