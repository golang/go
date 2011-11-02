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
	var w io.Writer
	w, err = os.Create(dstfile)
	if err != nil {
		return err
	}
	c, err := aes.NewCipher(key)
	if err != nil {
		return err
	}
	w = cipher.StreamWriter{S: cipher.NewOFB(c, iv), W: w}
	w2, err := gzip.NewWriter(w)
	if err != nil {
		return err
	}
	defer w2.Close()
	_, err = io.Copy(w2, r)
	return err
}

func DecryptAndGunzip(dstfile, srcfile string, key, iv []byte) error {
	f, err := os.Open(srcfile)
	if err != nil {
		return err
	}
	defer f.Close()
	c, err := aes.NewCipher(key)
	if err != nil {
		return err
	}
	r := cipher.StreamReader{S: cipher.NewOFB(c, iv), R: f}
	r2, err := gzip.NewReader(r)
	if err != nil {
		return err
	}
	w, err := os.Create(dstfile)
	if err != nil {
		return err
	}
	defer w.Close()
	_, err = io.Copy(w, r2)
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
	err = DecryptAndGunzip(
		"/dev/stdout",
		"/tmp/passwd.gz",
		make([]byte, 16),
		make([]byte, 16),
	)
	if err != nil {
		log.Fatal(err)
	}
}
