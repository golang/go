// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openpgp

import (
	"bytes"
	"crypto/rand"
	"os"
	"io"
	"testing"
	"time"
)

func TestSignDetached(t *testing.T) {
	kring, _ := ReadKeyRing(readerFromHex(testKeys1And2PrivateHex))
	out := bytes.NewBuffer(nil)
	message := bytes.NewBufferString(signedInput)
	err := DetachSign(out, kring[0], message)
	if err != nil {
		t.Error(err)
	}

	testDetachedSignature(t, kring, out, signedInput, "check", testKey1KeyId)
}

func TestSignTextDetached(t *testing.T) {
	kring, _ := ReadKeyRing(readerFromHex(testKeys1And2PrivateHex))
	out := bytes.NewBuffer(nil)
	message := bytes.NewBufferString(signedInput)
	err := DetachSignText(out, kring[0], message)
	if err != nil {
		t.Error(err)
	}

	testDetachedSignature(t, kring, out, signedInput, "check", testKey1KeyId)
}

func TestSignDetachedDSA(t *testing.T) {
	kring, _ := ReadKeyRing(readerFromHex(dsaTestKeyPrivateHex))
	out := bytes.NewBuffer(nil)
	message := bytes.NewBufferString(signedInput)
	err := DetachSign(out, kring[0], message)
	if err != nil {
		t.Error(err)
	}

	testDetachedSignature(t, kring, out, signedInput, "check", testKey3KeyId)
}

func TestNewEntity(t *testing.T) {
	if testing.Short() {
		return
	}

	e, err := NewEntity(rand.Reader, time.Seconds(), "Test User", "test", "test@example.com")
	if err != nil {
		t.Errorf("failed to create entity: %s", err)
		return
	}

	w := bytes.NewBuffer(nil)
	if err := e.SerializePrivate(w); err != nil {
		t.Errorf("failed to serialize entity: %s", err)
		return
	}
	serialized := w.Bytes()

	el, err := ReadKeyRing(w)
	if err != nil {
		t.Errorf("failed to reparse entity: %s", err)
		return
	}

	if len(el) != 1 {
		t.Errorf("wrong number of entities found, got %d, want 1", len(el))
	}

	w = bytes.NewBuffer(nil)
	if err := e.SerializePrivate(w); err != nil {
		t.Errorf("failed to serialize entity second time: %s", err)
		return
	}

	if !bytes.Equal(w.Bytes(), serialized) {
		t.Errorf("results differed")
	}
}

func TestSymmetricEncryption(t *testing.T) {
	buf := new(bytes.Buffer)
	plaintext, err := SymmetricallyEncrypt(buf, []byte("testing"), nil)
	if err != nil {
		t.Errorf("error writing headers: %s", err)
		return
	}
	message := []byte("hello world\n")
	_, err = plaintext.Write(message)
	if err != nil {
		t.Errorf("error writing to plaintext writer: %s", err)
	}
	err = plaintext.Close()
	if err != nil {
		t.Errorf("error closing plaintext writer: %s", err)
	}

	md, err := ReadMessage(buf, nil, func(keys []Key, symmetric bool) ([]byte, os.Error) {
		return []byte("testing"), nil
	})
	if err != nil {
		t.Errorf("error rereading message: %s", err)
	}
	messageBuf := bytes.NewBuffer(nil)
	_, err = io.Copy(messageBuf, md.UnverifiedBody)
	if err != nil {
		t.Errorf("error rereading message: %s", err)
	}
	if !bytes.Equal(message, messageBuf.Bytes()) {
		t.Errorf("recovered message incorrect got '%s', want '%s'", messageBuf.Bytes(), message)
	}
}
