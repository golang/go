// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code differs from the slides in that it handles errors.

package main

import (
	"crypto/aes"
	"crypto/cipher"
	"compress/gzip"
	"io"
	"log"
	"os"
)

func EncryptAndGzip(dstfile, srcfile string, key, iv []byte) error {
	r, err := os.Open(srcfile)
	if err != nil {
		return err
	}
	var w io.WriteCloser
	w, err = os.Create(dstfile)
	if err != nil {
		return err
	}
	defer w.Close()
	w, err = gzip.NewWriter(w)
	if err != nil {
		return err
	}
	defer w.Close()
	c, err := aes.NewCipher(key)
	if err != nil {
		return err
	}
	_, err = io.Copy(cipher.StreamWriter{S: cipher.NewOFB(c, iv), W: w}, r)
	return err
}

func main() {
	err := EncryptAndGzip(
		"/tmp/passwd.gz",
		"/etc/passwd",
		make([]byte, 16),
		make([]byte, 16),
	)
	if err != nil {
		log.Fatal(err)
	}
}
