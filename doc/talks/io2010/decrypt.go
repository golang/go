// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"crypto/aes"
	"crypto/cipher"
	"compress/gzip"
	"io"
	"os"
)

func EncryptAndGzip(dstfile, srcfile string, key, iv []byte) {
	r, _ := os.Open(srcfile)
	var w io.Writer
	w, _ = os.Create(dstfile)
	c, _ := aes.NewCipher(key)
	w = cipher.StreamWriter{S: cipher.NewOFB(c, iv), W: w}
	w2, _ := gzip.NewWriter(w)
	io.Copy(w2, r)
	w2.Close()
}

func DecryptAndGunzip(dstfile, srcfile string, key, iv []byte) {
	f, _ := os.Open(srcfile)
	defer f.Close()
	c, _ := aes.NewCipher(key)
	r := cipher.StreamReader{S: cipher.NewOFB(c, iv), R: f}
	r2, _ := gzip.NewReader(r)
	w, _ := os.Create(dstfile)
	defer w.Close()
	io.Copy(w, r2)
}

func main() {
	EncryptAndGzip("/tmp/passwd.gz", "/etc/passwd", make([]byte, 16), make([]byte, 16))
	DecryptAndGunzip("/dev/stdout", "/tmp/passwd.gz", make([]byte, 16), make([]byte, 16))
}
