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
	var w io.WriteCloser
	w, _ = os.Open(dstfile, os.O_WRONLY|os.O_CREATE, 0666)
	defer w.Close()
	w, _ = gzip.NewDeflater(w)
	defer w.Close()
	c, _ := aes.NewCipher(key)
	io.Copy(block.NewCBCEncrypter(c, iv, w), r)
}

func main() {
	EncryptAndGzip("/tmp/passwd.gz", "/etc/passwd", make([]byte, 16), make([]byte, 16))
}
