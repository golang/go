// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"encoding/hex"
	"io"
	"io/ioutil"
	"testing"
)

func TestCompressed(t *testing.T) {
	packet, err := Read(readerFromHex(compressedHex))
	if err != nil {
		t.Errorf("failed to read Compressed: %s", err)
		return
	}

	c, ok := packet.(*Compressed)
	if !ok {
		t.Error("didn't find Compressed packet")
		return
	}

	contents, err := ioutil.ReadAll(c.Body)
	if err != nil && err != io.EOF {
		t.Error(err)
		return
	}

	expected, _ := hex.DecodeString(compressedExpectedHex)
	if !bytes.Equal(expected, contents) {
		t.Errorf("got:%x want:%x", contents, expected)
	}
}

const compressedHex = "a3013b2d90c4e02b72e25f727e5e496a5e49b11e1700"
const compressedExpectedHex = "cb1062004d14c8fe636f6e74656e74732e0a"
