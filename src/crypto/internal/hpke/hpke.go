// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpke

import (
	"crypto"
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdh"
	"crypto/internal/fips140/hkdf"
	"crypto/rand"
	"errors"
	"internal/byteorder"
	"math/bits"

	"golang.org/x/crypto/chacha20poly1305"
)

// testingOnlyGenerateKey is only used during testing, to provide
// a fixed test key to use when checking the RFC 9180 vectors.
var testingOnlyGenerateKey func() (*ecdh.PrivateKey, error)

type hkdfKDF struct {
	hash crypto.Hash
}

func (kdf *hkdfKDF) LabeledExtract(sid []byte, salt []byte, label string, inputKey []byte) []byte {
	labeledIKM := make([]byte, 0, 7+len(sid)+len(label)+len(inputKey))
	labeledIKM = append(labeledIKM, []byte("HPKE-v1")...)
	labeledIKM = append(labeledIKM, sid...)
	labeledIKM = append(labeledIKM, label...)
	labeledIKM = append(labeledIKM, inputKey...)
	return hkdf.Extract(kdf.hash.New, labeledIKM, salt)
}

func (kdf *hkdfKDF) LabeledExpand(suiteID []byte, randomKey []byte, label string, info []byte, length uint16) []byte {
	labeledInfo := make([]byte, 0, 2+7+len(suiteID)+len(label)+len(info))
	labeledInfo = byteorder.BeAppendUint16(labeledInfo, length)
	labeledInfo = append(labeledInfo, []byte("HPKE-v1")...)
	labeledInfo = append(labeledInfo, suiteID...)
	labeledInfo = append(labeledInfo, label...)
	labeledInfo = append(labeledInfo, info...)
	return hkdf.Expand(kdf.hash.New, randomKey, labeledInfo, int(length))
}

// dhKEM implements the KEM specified in RFC 9180, Section 4.1.
type dhKEM struct {
	dh  ecdh.Curve
	kdf hkdfKDF

	suiteID []byte
	nSecret uint16
}

type KemID uint16

const DHKEM_X25519_HKDF_SHA256 = 0x0020

var SupportedKEMs = map[uint16]struct {
	curve   ecdh.Curve
	hash    crypto.Hash
	nSecret uint16
}{
	// RFC 9180 Section 7.1
	DHKEM_X25519_HKDF_SHA256: {ecdh.X25519(), crypto.SHA256, 32},
}

func newDHKem(kemID uint16) (*dhKEM, error) {
	suite, ok := SupportedKEMs[kemID]
	if !ok {
		return nil, errors.New("unsupported suite ID")
	}
	return &dhKEM{
		dh:      suite.curve,
		kdf:     hkdfKDF{suite.hash},
		suiteID: byteorder.BeAppendUint16([]byte("KEM"), kemID),
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

func (dh *dhKEM) Decap(encPubEph []byte, secRecipient *ecdh.PrivateKey) ([]byte, error) {
	pubEph, err := dh.dh.NewPublicKey(encPubEph)
	if err != nil {
		return nil, err
	}
	dhVal, err := secRecipient.ECDH(pubEph)
	if err != nil {
		return nil, err
	}
	kemContext := append(encPubEph, secRecipient.PublicKey().Bytes()...)

	return dh.ExtractAndExpand(dhVal, kemContext), nil
}

type context struct {
	aead cipher.AEAD

	sharedSecret []byte

	suiteID []byte

	key            []byte
	baseNonce      []byte
	exporterSecret []byte

	seqNum uint128
}

type Sender struct {
	*context
}

type Receipient struct {
	*context
}

var aesGCMNew = func(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}

type AEADID uint16

const (
	AEAD_AES_128_GCM      = 0x0001
	AEAD_AES_256_GCM      = 0x0002
	AEAD_ChaCha20Poly1305 = 0x0003
)

var SupportedAEADs = map[uint16]struct {
	keySize   int
	nonceSize int
	aead      func([]byte) (cipher.AEAD, error)
}{
	// RFC 9180, Section 7.3
	AEAD_AES_128_GCM:      {keySize: 16, nonceSize: 12, aead: aesGCMNew},
	AEAD_AES_256_GCM:      {keySize: 32, nonceSize: 12, aead: aesGCMNew},
	AEAD_ChaCha20Poly1305: {keySize: chacha20poly1305.KeySize, nonceSize: chacha20poly1305.NonceSize, aead: chacha20poly1305.New},
}

type KDFID uint16

const KDF_HKDF_SHA256 = 0x0001

var SupportedKDFs = map[uint16]func() *hkdfKDF{
	// RFC 9180, Section 7.2
	KDF_HKDF_SHA256: func() *hkdfKDF { return &hkdfKDF{crypto.SHA256} },
}

func newContext(sharedSecret []byte, kemID, kdfID, aeadID uint16, info []byte) (*context, error) {
	sid := suiteID(kemID, kdfID, aeadID)

	kdfInit, ok := SupportedKDFs[kdfID]
	if !ok {
		return nil, errors.New("unsupported KDF id")
	}
	kdf := kdfInit()

	aeadInfo, ok := SupportedAEADs[aeadID]
	if !ok {
		return nil, errors.New("unsupported AEAD id")
	}

	pskIDHash := kdf.LabeledExtract(sid, nil, "psk_id_hash", nil)
	infoHash := kdf.LabeledExtract(sid, nil, "info_hash", info)
	ksContext := append([]byte{0}, pskIDHash...)
	ksContext = append(ksContext, infoHash...)

	secret := kdf.LabeledExtract(sid, sharedSecret, "secret", nil)

	key := kdf.LabeledExpand(sid, secret, "key", ksContext, uint16(aeadInfo.keySize) /* Nk - key size for AEAD */)
	baseNonce := kdf.LabeledExpand(sid, secret, "base_nonce", ksContext, uint16(aeadInfo.nonceSize) /* Nn - nonce size for AEAD */)
	exporterSecret := kdf.LabeledExpand(sid, secret, "exp", ksContext, uint16(kdf.hash.Size()) /* Nh - hash output size of the kdf*/)

	aead, err := aeadInfo.aead(key)
	if err != nil {
		return nil, err
	}

	return &context{
		aead:           aead,
		sharedSecret:   sharedSecret,
		suiteID:        sid,
		key:            key,
		baseNonce:      baseNonce,
		exporterSecret: exporterSecret,
	}, nil
}

func SetupSender(kemID, kdfID, aeadID uint16, pub *ecdh.PublicKey, info []byte) ([]byte, *Sender, error) {
	kem, err := newDHKem(kemID)
	if err != nil {
		return nil, nil, err
	}
	sharedSecret, encapsulatedKey, err := kem.Encap(pub)
	if err != nil {
		return nil, nil, err
	}

	context, err := newContext(sharedSecret, kemID, kdfID, aeadID, info)
	if err != nil {
		return nil, nil, err
	}

	return encapsulatedKey, &Sender{context}, nil
}

func SetupReceipient(kemID, kdfID, aeadID uint16, priv *ecdh.PrivateKey, info, encPubEph []byte) (*Receipient, error) {
	kem, err := newDHKem(kemID)
	if err != nil {
		return nil, err
	}
	sharedSecret, err := kem.Decap(encPubEph, priv)
	if err != nil {
		return nil, err
	}

	context, err := newContext(sharedSecret, kemID, kdfID, aeadID, info)
	if err != nil {
		return nil, err
	}

	return &Receipient{context}, nil
}

func (ctx *context) nextNonce() []byte {
	nonce := ctx.seqNum.bytes()[16-ctx.aead.NonceSize():]
	for i := range ctx.baseNonce {
		nonce[i] ^= ctx.baseNonce[i]
	}
	return nonce
}

func (ctx *context) incrementNonce() {
	// Message limit is, according to the RFC, 2^95+1, which
	// is somewhat confusing, but we do as we're told.
	if ctx.seqNum.bitLen() >= (ctx.aead.NonceSize()*8)-1 {
		panic("message limit reached")
	}
	ctx.seqNum = ctx.seqNum.addOne()
}

func (s *Sender) Seal(aad, plaintext []byte) ([]byte, error) {
	ciphertext := s.aead.Seal(nil, s.nextNonce(), plaintext, aad)
	s.incrementNonce()
	return ciphertext, nil
}

func (r *Receipient) Open(aad, ciphertext []byte) ([]byte, error) {
	plaintext, err := r.aead.Open(nil, r.nextNonce(), ciphertext, aad)
	if err != nil {
		return nil, err
	}
	r.incrementNonce()
	return plaintext, nil
}

func suiteID(kemID, kdfID, aeadID uint16) []byte {
	suiteID := make([]byte, 0, 4+2+2+2)
	suiteID = append(suiteID, []byte("HPKE")...)
	suiteID = byteorder.BeAppendUint16(suiteID, kemID)
	suiteID = byteorder.BeAppendUint16(suiteID, kdfID)
	suiteID = byteorder.BeAppendUint16(suiteID, aeadID)
	return suiteID
}

func ParseHPKEPublicKey(kemID uint16, bytes []byte) (*ecdh.PublicKey, error) {
	kemInfo, ok := SupportedKEMs[kemID]
	if !ok {
		return nil, errors.New("unsupported KEM id")
	}
	return kemInfo.curve.NewPublicKey(bytes)
}

func ParseHPKEPrivateKey(kemID uint16, bytes []byte) (*ecdh.PrivateKey, error) {
	kemInfo, ok := SupportedKEMs[kemID]
	if !ok {
		return nil, errors.New("unsupported KEM id")
	}
	return kemInfo.curve.NewPrivateKey(bytes)
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
	byteorder.BePutUint64(b[0:], u.hi)
	byteorder.BePutUint64(b[8:], u.lo)
	return b
}
