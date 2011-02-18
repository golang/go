// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto/openpgp/error"
	"crypto/sha1"
	"encoding/hex"
	"io/ioutil"
	"os"
	"testing"
)

// TestReader wraps a []byte and returns reads of a specific length.
type testReader struct {
	data   []byte
	stride int
}

func (t *testReader) Read(buf []byte) (n int, err os.Error) {
	n = t.stride
	if n > len(t.data) {
		n = len(t.data)
	}
	if n > len(buf) {
		n = len(buf)
	}
	copy(buf, t.data)
	t.data = t.data[n:]
	if len(t.data) == 0 {
		err = os.EOF
	}
	return
}

func testMDCReader(t *testing.T) {
	mdcPlaintext, _ := hex.DecodeString(mdcPlaintextHex)

	for stride := 1; stride < len(mdcPlaintext)/2; stride++ {
		r := &testReader{data: mdcPlaintext, stride: stride}
		mdcReader := &seMDCReader{in: r, h: sha1.New()}
		body, err := ioutil.ReadAll(mdcReader)
		if err != nil {
			t.Errorf("stride: %d, error: %s", stride, err)
			continue
		}
		if !bytes.Equal(body, mdcPlaintext[:len(mdcPlaintext)-22]) {
			t.Errorf("stride: %d: bad contents %x", stride, body)
			continue
		}

		err = mdcReader.Close()
		if err != nil {
			t.Errorf("stride: %d, error on Close: %s", stride, err)
		}
	}

	mdcPlaintext[15] ^= 80

	r := &testReader{data: mdcPlaintext, stride: 2}
	mdcReader := &seMDCReader{in: r, h: sha1.New()}
	_, err := ioutil.ReadAll(mdcReader)
	if err != nil {
		t.Errorf("corruption test, error: %s", err)
		return
	}
	err = mdcReader.Close()
	if err == nil {
		t.Error("corruption: no error")
	} else if _, ok := err.(*error.SignatureError); !ok {
		t.Errorf("corruption: expected SignatureError, got: %s", err)
	}
}

const mdcPlaintextHex = "a302789c3b2d93c4e0eb9aba22283539b3203335af44a134afb800c849cb4c4de10200aff40b45d31432c80cb384299a0655966d6939dfdeed1dddf980"
