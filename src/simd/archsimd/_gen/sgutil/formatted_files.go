// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import (
	"bufio"
	"bytes"
	"fmt"
	"go/format"
	"os"
	"path/filepath"
)

func CreatePath(newFile string) (*os.File, error) {
	dir := filepath.Dir(newFile)
	err := os.MkdirAll(dir, 0755)
	if err != nil {
		return nil, fmt.Errorf("failed to create directory %s: %w", dir, err)
	}
	f, err := os.Create(newFile)
	if err != nil {
		return nil, fmt.Errorf("failed to create file %s: %w", newFile, err)
	}
	return f, nil
}

// FormatWriteAndClose formats the Go source code in source and writes it
// to newFile.  If there is a problem with the formatting, the
// entire source is numbered and emitted along with the error message,
// to help figure out where the source code generation went wrong.
func FormatWriteAndClose(source *bytes.Buffer, newFile string) {
	b, err := format.Source(source.Bytes())
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		fmt.Fprintf(os.Stderr, "%s\n", NumberLines(source.Bytes()))
		fmt.Fprintf(os.Stderr, "%v\n", err)
		panic(err)
	} else {
		WriteAndClose(b, newFile)
	}
}

// WriteAndClose creates newFile, writes g to it,
// and closes the file.
func WriteAndClose(b []byte, newFile string) {
	ofile, err := CreatePath(newFile)
	if err != nil {
		panic(err)
	}
	ofile.Write(b)
	ofile.Close()
}

// NumberLines takes a slice of bytes, and returns a string where each line
// is numbered, starting from 1.
func NumberLines(data []byte) string {
	var buf bytes.Buffer
	r := bytes.NewReader(data)
	s := bufio.NewScanner(r)
	for i := 1; s.Scan(); i++ {
		fmt.Fprintf(&buf, "%d: %s\n", i, s.Text())
	}
	return buf.String()
}
