// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpke

import (
	"crypto/hkdf"
	"crypto/sha256"
	"crypto/sha3"
	"crypto/sha512"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
)

// The KDF is one of the three components of an HPKE ciphersuite, implementing
// key derivation.
type KDF interface {
	ID() uint16
	oneStage() bool
	size() int // Nh
	labeledDerive(suiteID, inputKey []byte, label string, context []byte, length uint16) ([]byte, error)
	labeledExtract(suiteID, salt []byte, label string, inputKey []byte) ([]byte, error)
	labeledExpand(suiteID, randomKey []byte, label string, info []byte, length uint16) ([]byte, error)
}

// NewKDF returns the KDF implementation for the given KDF ID.
//
// Applications are encouraged to use specific implementations like [HKDFSHA256]
// instead, unless runtime agility is required.
func NewKDF(id uint16) (KDF, error) {
	switch id {
	case 0x0001: // HKDF-SHA256
		return HKDFSHA256(), nil
	case 0x0002: // HKDF-SHA384
		return HKDFSHA384(), nil
	case 0x0003: // HKDF-SHA512
		return HKDFSHA512(), nil
	case 0x0010: // SHAKE128
		return SHAKE128(), nil
	case 0x0011: // SHAKE256
		return SHAKE256(), nil
	default:
		return nil, fmt.Errorf("unsupported KDF %04x", id)
	}
}

// HKDFSHA256 returns an HKDF-SHA256 KDF implementation.
func HKDFSHA256() KDF { return hkdfSHA256 }

// HKDFSHA384 returns an HKDF-SHA384 KDF implementation.
func HKDFSHA384() KDF { return hkdfSHA384 }

// HKDFSHA512 returns an HKDF-SHA512 KDF implementation.
func HKDFSHA512() KDF { return hkdfSHA512 }

type hkdfKDF struct {
	hash func() hash.Hash
	id   uint16
	nH   int
}

var hkdfSHA256 = &hkdfKDF{hash: sha256.New, id: 0x0001, nH: sha256.Size}
var hkdfSHA384 = &hkdfKDF{hash: sha512.New384, id: 0x0002, nH: sha512.Size384}
var hkdfSHA512 = &hkdfKDF{hash: sha512.New, id: 0x0003, nH: sha512.Size}

func (kdf *hkdfKDF) ID() uint16 {
	return kdf.id
}

func (kdf *hkdfKDF) size() int {
	return kdf.nH
}

func (kdf *hkdfKDF) oneStage() bool {
	return false
}

func (kdf *hkdfKDF) labeledDerive(_, _ []byte, _ string, _ []byte, _ uint16) ([]byte, error) {
	return nil, errors.New("hpke: internal error: labeledDerive called on two-stage KDF")
}

func (kdf *hkdfKDF) labeledExtract(suiteID []byte, salt []byte, label string, inputKey []byte) ([]byte, error) {
	labeledIKM := make([]byte, 0, 7+len(suiteID)+len(label)+len(inputKey))
	labeledIKM = append(labeledIKM, []byte("HPKE-v1")...)
	labeledIKM = append(labeledIKM, suiteID...)
	labeledIKM = append(labeledIKM, label...)
	labeledIKM = append(labeledIKM, inputKey...)
	return hkdf.Extract(kdf.hash, labeledIKM, salt)
}

func (kdf *hkdfKDF) labeledExpand(suiteID []byte, randomKey []byte, label string, info []byte, length uint16) ([]byte, error) {
	labeledInfo := make([]byte, 0, 2+7+len(suiteID)+len(label)+len(info))
	labeledInfo = binary.BigEndian.AppendUint16(labeledInfo, length)
	labeledInfo = append(labeledInfo, []byte("HPKE-v1")...)
	labeledInfo = append(labeledInfo, suiteID...)
	labeledInfo = append(labeledInfo, label...)
	labeledInfo = append(labeledInfo, info...)
	return hkdf.Expand(kdf.hash, randomKey, string(labeledInfo), int(length))
}

// SHAKE128 returns a SHAKE128 KDF implementation.
func SHAKE128() KDF {
	return shake128KDF
}

// SHAKE256 returns a SHAKE256 KDF implementation.
func SHAKE256() KDF {
	return shake256KDF
}

type shakeKDF struct {
	hash func() *sha3.SHAKE
	id   uint16
	nH   int
}

var shake128KDF = &shakeKDF{hash: sha3.NewSHAKE128, id: 0x0010, nH: 32}
var shake256KDF = &shakeKDF{hash: sha3.NewSHAKE256, id: 0x0011, nH: 64}

func (kdf *shakeKDF) ID() uint16 {
	return kdf.id
}

func (kdf *shakeKDF) size() int {
	return kdf.nH
}

func (kdf *shakeKDF) oneStage() bool {
	return true
}

func (kdf *shakeKDF) labeledDerive(suiteID, inputKey []byte, label string, context []byte, length uint16) ([]byte, error) {
	H := kdf.hash()
	H.Write(inputKey)
	H.Write([]byte("HPKE-v1"))
	H.Write(suiteID)
	H.Write([]byte{byte(len(label) >> 8), byte(len(label))})
	H.Write([]byte(label))
	H.Write([]byte{byte(length >> 8), byte(length)})
	H.Write(context)
	out := make([]byte, length)
	H.Read(out)
	return out, nil
}

func (kdf *shakeKDF) labeledExtract(_, _ []byte, _ string, _ []byte) ([]byte, error) {
	return nil, errors.New("hpke: internal error: labeledExtract called on one-stage KDF")
}

func (kdf *shakeKDF) labeledExpand(_, _ []byte, _ string, _ []byte, _ uint16) ([]byte, error) {
	return nil, errors.New("hpke: internal error: labeledExpand called on one-stage KDF")
}
