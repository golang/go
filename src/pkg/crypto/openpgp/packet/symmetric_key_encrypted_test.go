// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"io"
	"io/ioutil"
	"testing"
)

func TestSymmetricKeyEncrypted(t *testing.T) {
	buf := readerFromHex(symmetricallyEncryptedHex)
	packet, err := Read(buf)
	if err != nil {
		t.Errorf("failed to read SymmetricKeyEncrypted: %s", err)
		return
	}
	ske, ok := packet.(*SymmetricKeyEncrypted)
	if !ok {
		t.Error("didn't find SymmetricKeyEncrypted packet")
		return
	}
	err = ske.Decrypt([]byte("password"))
	if err != nil {
		t.Error(err)
		return
	}

	packet, err = Read(buf)
	if err != nil {
		t.Errorf("failed to read SymmetricallyEncrypted: %s", err)
		return
	}
	se, ok := packet.(*SymmetricallyEncrypted)
	if !ok {
		t.Error("didn't find SymmetricallyEncrypted packet")
		return
	}
	r, err := se.Decrypt(ske.CipherFunc, ske.Key)
	if err != nil {
		t.Error(err)
		return
	}

	contents, err := ioutil.ReadAll(r)
	if err != nil && err != io.EOF {
		t.Error(err)
		return
	}

	expectedContents, _ := hex.DecodeString(symmetricallyEncryptedContentsHex)
	if !bytes.Equal(expectedContents, contents) {
		t.Errorf("bad contents got:%x want:%x", contents, expectedContents)
	}
}

const symmetricallyEncryptedHex = "8c0d04030302371a0b38d884f02060c91cf97c9973b8e58e028e9501708ccfe618fb92afef7fa2d80ddadd93cf"
const symmetricallyEncryptedContentsHex = "cb1062004d14c4df636f6e74656e74732e0a"

func TestSerializeSymmetricKeyEncrypted(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	passphrase := []byte("testing")
	cipherFunc := CipherAES128

	key, err := SerializeSymmetricKeyEncrypted(buf, rand.Reader, passphrase, cipherFunc)
	if err != nil {
		t.Errorf("failed to serialize: %s", err)
		return
	}

	p, err := Read(buf)
	if err != nil {
		t.Errorf("failed to reparse: %s", err)
		return
	}
	ske, ok := p.(*SymmetricKeyEncrypted)
	if !ok {
		t.Errorf("parsed a different packet type: %#v", p)
		return
	}

	if !ske.Encrypted {
		t.Errorf("SKE not encrypted but should be")
	}
	if ske.CipherFunc != cipherFunc {
		t.Errorf("SKE cipher function is %d (expected %d)", ske.CipherFunc, cipherFunc)
	}
	err = ske.Decrypt(passphrase)
	if err != nil {
		t.Errorf("failed to decrypt reparsed SKE: %s", err)
		return
	}
	if !bytes.Equal(key, ske.Key) {
		t.Errorf("keys don't match after Decrpyt: %x (original) vs %x (parsed)", key, ske.Key)
	}
}
