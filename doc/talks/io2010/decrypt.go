// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"crypto/aes"
	"crypto/block"
	"compress/gzip"
	"io"
	"os"
)

func EncryptAndGzip(dstfile, srcfile string, key, iv []byte) {
	r, _ := os.Open(srcfile, os.O_RDONLY, 0)
	var w io.Writer
	w, _ = os.Open(dstfile, os.O_WRONLY|os.O_CREATE, 0666)
	c, _ := aes.NewCipher(key)
	w = block.NewOFBWriter(c, iv, w)
	w2, _ := gzip.NewDeflater(w)
	io.Copy(w2, r)
	w2.Close()
}

func DecryptAndGunzip(dstfile, srcfile string, key, iv []byte) {
	f, _ := os.Open(srcfile, os.O_RDONLY, 0)
	defer f.Close()
	c, _ := aes.NewCipher(key)
	r := block.NewOFBReader(c, iv, f)
	r, _ = gzip.NewInflater(r)
	w, _ := os.Open(dstfile, os.O_WRONLY|os.O_CREATE, 0666)
	defer w.Close()
	io.Copy(w, r)
}

func main() {
	EncryptAndGzip("/tmp/passwd.gz", "/etc/passwd", make([]byte, 16), make([]byte, 16))
	DecryptAndGunzip("/dev/stdout", "/tmp/passwd.gz", make([]byte, 16), make([]byte, 16))
}
