// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package obscuretestdata contains functionality used by tests to more easily
// work with testdata that must be obscured primarily due to
// golang.org/issue/34986.
package obscuretestdata

import (
	"encoding/base64"
	"io"
	"os"
)

// Rot13 returns the rot13 encoding or decoding of its input.
func Rot13(data []byte) []byte {
	out := make([]byte, len(data))
	copy(out, data)
	for i, c := range out {
		switch {
		case 'A' <= c && c <= 'M' || 'a' <= c && c <= 'm':
			out[i] = c + 13
		case 'N' <= c && c <= 'Z' || 'n' <= c && c <= 'z':
			out[i] = c - 13
		}
	}
	return out
}

// DecodeToTempFile decodes the named file to a temporary location.
// If successful, it returns the path of the decoded file.
// The caller is responsible for ensuring that the temporary file is removed.
func DecodeToTempFile(name string) (path string, err error) {
	f, err := os.Open(name)
	if err != nil {
		return "", err
	}
	defer f.Close()

	tmp, err := os.CreateTemp("", "obscuretestdata-decoded-")
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(tmp, base64.NewDecoder(base64.StdEncoding, f)); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return "", err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return "", err
	}
	return tmp.Name(), nil
}

// ReadFile reads the named file and returns its decoded contents.
func ReadFile(name string) ([]byte, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(base64.NewDecoder(base64.StdEncoding, f))
}
