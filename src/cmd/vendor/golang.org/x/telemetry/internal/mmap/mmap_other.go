// Copyright 2022 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (js && wasm) || wasip1 || plan9 || (solaris && !go1.20)

package mmap

import (
	"io"
	"os"
)

// mmapFile on other systems doesn't mmap the file. It just reads everything.
func mmapFile(f *os.File, _ *Data) (Data, error) {
	b, err := io.ReadAll(f)
	if err != nil {
		return Data{}, err
	}
	return Data{f, b, nil}, nil
}

func munmapFile(d Data) error {
	return nil
}
