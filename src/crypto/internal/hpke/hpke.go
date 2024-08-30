// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpke

import (
	"crypto"
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdh"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"math/bits"

	"golang.org/x/crypto/chacha20poly1305"
	"golang.org/x/crypto/hkdf"
)

// testingOnlyGenerateKey is only used during testing, to provide
// a fixed test key to use when checking the RFC 9180 vectors.
var testingOnlyGenerateKey func() (*ecdh.PrivateKey, error)

type hkdfKDF struct {
	hash crypto.Hash
}

func (kdf *hkdfKDF) LabeledExtract(suiteID []byte, salt []byte, label string, inputKey []byte) []byte {
	labeledIKM := make([]byte, 0, 7+len(suiteID)+len(label)+len(inputKey))
	labeledIKM = append(labeledIKM, []byte("HPKE-v1")...)
	labeledIKM = append(labeledIKM, suiteID...)
	labeledIKM = append(labeledIKM, label...)
	labeledIKM = append(labeledIKM, inputKey...)
	return hkdf.Extract(kdf.hash.New, labeledIKM, salt)
}

func (kdf *hkdfKDF) LabeledExpand(suiteID []byte, randomKey []byte, label string, info []byte, length uint16) []byte {
	labeledInfo := make([]byte, 0, 2+7+len(suiteID)+len(label)+len(info))
	labeledInfo = binary.BigEndian.AppendUint16(labeledInfo, length)
	labeledInfo = append(labeledInfo, []byte("HPKE-v1")...)
	labeledInfo = append(labeledInfo, suiteID...)
	labeledInfo = append(labeledInfo, label...)
	labeledInfo = append(labeledInfo, info...)
	out := make([]byte, length)
	n, err := hkdf.Expand(kdf.hash.New, randomKey, labeledInfo).Read(out)
	if err != nil || n != int(length) {
		panic("hpke: LabeledExpand failed unexpectedly")
	}
	return out
}

// dhKEM implements the KEM specified in RFC 9180, Section 4.1.
type dhKEM struct {
	dh  ecdh.Curve
	kdf hkdfKDF

	suiteID []byte
	nSecret uint16
}

var SupportedKEMs = map[uint16]struct {
	curve   ecdh.Curve
	hash    crypto.Hash
	nSecret uint16
}{
	// RFC 9180 Section 7.1
	0x0020: {ecdh.X25519(), crypto.SHA256, 32},
}

func newDHKem(kemID uint16) (*dhKEM, error) {
	suite, ok := SupportedKEMs[kemID]
	if !ok {
		return nil, errors.New("unsupported suite ID")
	}
	return &dhKEM{
		dh:      suite.curve,
		kdf:     hkdfKDF{suite.hash},
		suiteID: binary.BigEndian.AppendUint16([]byte("KEM"), kemID),
		nSecret: suite.nSecret,
	}, nil
}

func (dh *dhKEM) ExtractAndExpand(dhKey, kemContext []byte) []byte {
	eaePRK := dh.kdf.LabeledExtract(dh.suiteID[:], nil, "eae_prk", dhKey)
	return dh.kdf.LabeledExpand(dh.suiteID[:], eaePRK, "shared_secret", kemContext, dh.nSecret)
}

func (dh *dhKEM) Encap(pubRecipient *ecdh.PublicKey) (sharedSecret []byte, encapPub []byte, err error) {
	var privEph *ecdh.PrivateKey
	if testingOnlyGenerateKey != nil {
		privEph, err = testingOnlyGenerateKey()
	} else {
		privEph, err = dh.dh.GenerateKey(rand.Reader)
	}
	if err != nil {
		return nil, nil, err
	}
	dhVal, err := privEph.ECDH(pubRecipient)
	if err != nil {
		return nil, nil, err
	}
	encPubEph := privEph.PublicKey().Bytes()

	encPubRecip := pubRecipient.Bytes()
	kemContext := append(encPubEph, encPubRecip...)

	return dh.ExtractAndExpand(dhVal, kemContext), encPubEph, nil
}

type Sender struct {
	aead cipher.AEAD
	kem  *dhKEM

	sharedSecret []byte

	suiteID []byte

	key            []byte
	baseNonce      []byte
	exporterSecret []byte

	seqNum uint128
}

var aesGCMNew = func(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}

var SupportedAEADs = map[uint16]struct {
	keySize   int
	nonceSize int
	aead      func([]byte) (cipher.AEAD, error)
}{
	// RFC 9180, Section 7.3
	0x0001: {keySize: 16, nonceSize: 12, aead: aesGCMNew},
	0x0002: {keySize: 32, nonceSize: 12, aead: aesGCMNew},
	0x0003: {keySize: chacha20poly1305.KeySize, nonceSize: chacha20poly1305.NonceSize, aead: chacha20poly1305.New},
}

var SupportedKDFs = map[uint16]func() *hkdfKDF{
	// RFC 9180, Section 7.2
	0x0001: func() *hkdfKDF { return &hkdfKDF{crypto.SHA256} },
}

func SetupSender(kemID, kdfID, aeadID uint16, pub crypto.PublicKey, info []byte) ([]byte, *Sender, error) {
	suiteID := SuiteID(kemID, kdfID, aeadID)

	kem, err := newDHKem(kemID)
	if err != nil {
		return nil, nil, err
	}
	pubRecipient, ok := pub.(*ecdh.PublicKey)
	if !ok {
		return nil, nil, errors.New("incorrect public key type")
	}
	sharedSecret, encapsulatedKey, err := kem.Encap(pubRecipient)
	if err != nil {
		return nil, nil, err
	}

	kdfInit, ok := SupportedKDFs[kdfID]
	if !ok {
		return nil, nil, errors.New("unsupported KDF id")
	}
	kdf := kdfInit()

	aeadInfo, ok := SupportedAEADs[aeadID]
	if !ok {
		return nil, nil, errors.New("unsupported AEAD id")
	}

	pskIDHash := kdf.LabeledExtract(suiteID, nil, "psk_id_hash", nil)
	infoHash := kdf.LabeledExtract(suiteID, nil, "info_hash", info)
	ksContext := append([]byte{0}, pskIDHash...)
	ksContext = append(ksContext, infoHash...)

	secret := kdf.LabeledExtract(suiteID, sharedSecret, "secret", nil)

	key := kdf.LabeledExpand(suiteID, secret, "key", ksContext, uint16(aeadInfo.keySize) /* Nk - key size for AEAD */)
	baseNonce := kdf.LabeledExpand(suiteID, secret, "base_nonce", ksContext, uint16(aeadInfo.nonceSize) /* Nn - nonce size for AEAD */)
	exporterSecret := kdf.LabeledExpand(suiteID, secret, "exp", ksContext, uint16(kdf.hash.Size()) /* Nh - hash output size of the kdf*/)

	aead, err := aeadInfo.aead(key)
	if err != nil {
		return nil, nil, err
	}

	return encapsulatedKey, &Sender{
		kem:            kem,
		aead:           aead,
		sharedSecret:   sharedSecret,
		suiteID:        suiteID,
		key:            key,
		baseNonce:      baseNonce,
		exporterSecret: exporterSecret,
	}, nil
}

func (s *Sender) nextNonce() []byte {
	nonce := s.seqNum.bytes()[16-s.aead.NonceSize():]
	for i := range s.baseNonce {
		nonce[i] ^= s.baseNonce[i]
	}
	// Message limit is, according to the RFC, 2^95+1, which
	// is somewhat confusing, but we do as we're told.
	if s.seqNum.bitLen() >= (s.aead.NonceSize()*8)-1 {
		panic("message limit reached")
	}
	s.seqNum = s.seqNum.addOne()
	return nonce
}

func (s *Sender) Seal(aad, plaintext []byte) ([]byte, error) {

	ciphertext := s.aead.Seal(nil, s.nextNonce(), plaintext, aad)
	return ciphertext, nil
}

func SuiteID(kemID, kdfID, aeadID uint16) []byte {
	suiteID := make([]byte, 0, 4+2+2+2)
	suiteID = append(suiteID, []byte("HPKE")...)
	suiteID = binary.BigEndian.AppendUint16(suiteID, kemID)
	suiteID = binary.BigEndian.AppendUint16(suiteID, kdfID)
	suiteID = binary.BigEndian.AppendUint16(suiteID, aeadID)
	return suiteID
}

func ParseHPKEPublicKey(kemID uint16, bytes []byte) (*ecdh.PublicKey, error) {
	kemInfo, ok := SupportedKEMs[kemID]
	if !ok {
		return nil, errors.New("unsupported KEM id")
	}
	return kemInfo.curve.NewPublicKey(bytes)
}

type uint128 struct {
	hi, lo uint64
}

func (u uint128) addOne() uint128 {
	lo, carry := bits.Add64(u.lo, 1, 0)
	return uint128{u.hi + carry, lo}
}

func (u uint128) bitLen() int {
	return bits.Len64(u.hi) + bits.Len64(u.lo)
}

func (u uint128) bytes() []byte {
	b := make([]byte, 16)
	binary.BigEndian.PutUint64(b[0:], u.hi)
	binary.BigEndian.PutUint64(b[8:], u.lo)
	return b
}
