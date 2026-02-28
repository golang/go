// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	. "crypto/cipher"
	"reflect"
	"testing"
)

// Historically, crypto/aes's Block would implement some undocumented
// methods for crypto/cipher to use from NewCTR, NewCBCEncrypter, etc.
// This is no longer the case, but for now test that the mechanism is
// still working until we explicitly decide to remove it.

type block struct {
	Block
}

func (block) BlockSize() int {
	return 16
}

type specialCTR struct {
	Stream
}

func (block) NewCTR(iv []byte) Stream {
	return specialCTR{}
}

func TestCTRAble(t *testing.T) {
	b := block{}
	s := NewCTR(b, make([]byte, 16))
	if _, ok := s.(specialCTR); !ok {
		t.Errorf("NewCTR did not return specialCTR")
	}
}

type specialCBC struct {
	BlockMode
}

func (block) NewCBCEncrypter(iv []byte) BlockMode {
	return specialCBC{}
}

func (block) NewCBCDecrypter(iv []byte) BlockMode {
	return specialCBC{}
}

func TestCBCAble(t *testing.T) {
	b := block{}
	s := NewCBCEncrypter(b, make([]byte, 16))
	if _, ok := s.(specialCBC); !ok {
		t.Errorf("NewCBCEncrypter did not return specialCBC")
	}
	s = NewCBCDecrypter(b, make([]byte, 16))
	if _, ok := s.(specialCBC); !ok {
		t.Errorf("NewCBCDecrypter did not return specialCBC")
	}
}

type specialGCM struct {
	AEAD
}

func (block) NewGCM(nonceSize, tagSize int) (AEAD, error) {
	return specialGCM{}, nil
}

func TestGCM(t *testing.T) {
	b := block{}
	s, err := NewGCM(b)
	if err != nil {
		t.Errorf("NewGCM failed: %v", err)
	}
	if _, ok := s.(specialGCM); !ok {
		t.Errorf("NewGCM did not return specialGCM")
	}
}

// TestNoExtraMethods makes sure we don't accidentally expose methods on the
// underlying implementations of modes.
func TestNoExtraMethods(t *testing.T) {
	testAllImplementations(t, testNoExtraMethods)
}

func testNoExtraMethods(t *testing.T, newBlock func([]byte) Block) {
	b := newBlock(make([]byte, 16))

	ctr := NewCTR(b, make([]byte, 16))
	ctrExpected := []string{"XORKeyStream"}
	if got := exportedMethods(ctr); !reflect.DeepEqual(got, ctrExpected) {
		t.Errorf("CTR: got %v, want %v", got, ctrExpected)
	}

	cbc := NewCBCEncrypter(b, make([]byte, 16))
	cbcExpected := []string{"BlockSize", "CryptBlocks", "SetIV"}
	if got := exportedMethods(cbc); !reflect.DeepEqual(got, cbcExpected) {
		t.Errorf("CBC: got %v, want %v", got, cbcExpected)
	}
	cbc = NewCBCDecrypter(b, make([]byte, 16))
	if got := exportedMethods(cbc); !reflect.DeepEqual(got, cbcExpected) {
		t.Errorf("CBC: got %v, want %v", got, cbcExpected)
	}

	gcm, _ := NewGCM(b)
	gcmExpected := []string{"NonceSize", "Open", "Overhead", "Seal"}
	if got := exportedMethods(gcm); !reflect.DeepEqual(got, gcmExpected) {
		t.Errorf("GCM: got %v, want %v", got, gcmExpected)
	}
}

func exportedMethods(x any) []string {
	var methods []string
	v := reflect.ValueOf(x)
	for i := 0; i < v.NumMethod(); i++ {
		if v.Type().Method(i).IsExported() {
			methods = append(methods, v.Type().Method(i).Name)
		}
	}
	return methods
}
