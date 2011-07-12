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
	var w io.WriteCloser
	w, _ = os.Create(dstfile)
	defer w.Close()
	w, _ = gzip.NewWriter(w)
	defer w.Close()
	c, _ := aes.NewCipher(key)
	io.Copy(cipher.StreamWriter{S: cipher.NewOFB(c, iv), W: w}, r)
}

func main() {
	EncryptAndGzip("/tmp/passwd.gz", "/etc/passwd", make([]byte, 16), make([]byte, 16))
}
