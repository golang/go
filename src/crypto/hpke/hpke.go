// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hpke implements Hybrid Public Key Encryption (HPKE) as defined in
// [RFC 9180].
//
// [RFC 9180]: https://www.rfc-editor.org/rfc/rfc9180.html
package hpke

import (
	"crypto/cipher"
	"errors"
	"internal/byteorder"
)

type context struct {
	suiteID []byte

	export func(string, uint16) ([]byte, error)

	aead      cipher.AEAD
	baseNonce []byte
	// seqNum starts at zero and is incremented for each Seal/Open call.
	// 64 bits are enough not to overflow for 500 years at 1ns per operation.
	seqNum uint64
}

// Sender is a sending HPKE context. It is instantiated with a specific KEM
// encapsulation key (i.e. the public key), and it is stateful, incrementing the
// nonce counter for each [Sender.Seal] call.
type Sender struct {
	*context
}

// Recipient is a receiving HPKE context. It is instantiated with a specific KEM
// decapsulation key (i.e. the secret key), and it is stateful, incrementing the
// nonce counter for each successful [Recipient.Open] call.
type Recipient struct {
	*context
}

func newContext(sharedSecret []byte, kemID uint16, kdf KDF, aead AEAD, info []byte) (*context, error) {
	sid := suiteID(kemID, kdf.ID(), aead.ID())

	if kdf.oneStage() {
		secrets := make([]byte, 0, 2+2+len(sharedSecret))
		secrets = byteorder.BEAppendUint16(secrets, 0) // empty psk
		secrets = byteorder.BEAppendUint16(secrets, uint16(len(sharedSecret)))
		secrets = append(secrets, sharedSecret...)

		ksContext := make([]byte, 0, 1+2+2+len(info))
		ksContext = append(ksContext, 0)                   // mode 0
		ksContext = byteorder.BEAppendUint16(ksContext, 0) // empty psk_id
		ksContext = byteorder.BEAppendUint16(ksContext, uint16(len(info)))
		ksContext = append(ksContext, info...)

		secret, err := kdf.labeledDerive(sid, secrets, "secret", ksContext,
			uint16(aead.keySize()+aead.nonceSize()+kdf.size()))
		if err != nil {
			return nil, err
		}
		key := secret[:aead.keySize()]
		baseNonce := secret[aead.keySize() : aead.keySize()+aead.nonceSize()]
		expSecret := secret[aead.keySize()+aead.nonceSize():]

		a, err := aead.aead(key)
		if err != nil {
			return nil, err
		}
		export := func(exporterContext string, length uint16) ([]byte, error) {
			return kdf.labeledDerive(sid, expSecret, "sec", []byte(exporterContext), length)
		}

		return &context{
			aead:      a,
			suiteID:   sid,
			export:    export,
			baseNonce: baseNonce,
		}, nil
	}

	pskIDHash, err := kdf.labeledExtract(sid, nil, "psk_id_hash", nil)
	if err != nil {
		return nil, err
	}
	infoHash, err := kdf.labeledExtract(sid, nil, "info_hash", info)
	if err != nil {
		return nil, err
	}
	ksContext := append([]byte{0}, pskIDHash...)
	ksContext = append(ksContext, infoHash...)

	secret, err := kdf.labeledExtract(sid, sharedSecret, "secret", nil)
	if err != nil {
		return nil, err
	}
	key, err := kdf.labeledExpand(sid, secret, "key", ksContext, uint16(aead.keySize()))
	if err != nil {
		return nil, err
	}
	a, err := aead.aead(key)
	if err != nil {
		return nil, err
	}
	baseNonce, err := kdf.labeledExpand(sid, secret, "base_nonce", ksContext, uint16(aead.nonceSize()))
	if err != nil {
		return nil, err
	}
	expSecret, err := kdf.labeledExpand(sid, secret, "exp", ksContext, uint16(kdf.size()))
	if err != nil {
		return nil, err
	}
	export := func(exporterContext string, length uint16) ([]byte, error) {
		return kdf.labeledExpand(sid, expSecret, "sec", []byte(exporterContext), length)
	}

	return &context{
		aead:      a,
		suiteID:   sid,
		export:    export,
		baseNonce: baseNonce,
	}, nil
}

// NewSender returns a sending HPKE context for the provided KEM encapsulation
// key (i.e. the public key), and using the ciphersuite defined by the
// combination of KEM, KDF, and AEAD.
//
// The info parameter is additional public information that must match between
// sender and recipient.
//
// The returned enc ciphertext can be used to instantiate a matching receiving
// HPKE context with the corresponding KEM decapsulation key.
func NewSender(pk PublicKey, kdf KDF, aead AEAD, info []byte) (enc []byte, s *Sender, err error) {
	sharedSecret, encapsulatedKey, err := pk.encap()
	if err != nil {
		return nil, nil, err
	}
	context, err := newContext(sharedSecret, pk.KEM().ID(), kdf, aead, info)
	if err != nil {
		return nil, nil, err
	}
	return encapsulatedKey, &Sender{context}, nil
}

// NewRecipient returns a receiving HPKE context for the provided KEM
// decapsulation key (i.e. the secret key), and using the ciphersuite defined by
// the combination of KEM, KDF, and AEAD.
//
// The enc parameter must have been produced by a matching sending HPKE context
// with the corresponding KEM encapsulation key. The info parameter is
// additional public information that must match between sender and recipient.
func NewRecipient(enc []byte, k PrivateKey, kdf KDF, aead AEAD, info []byte) (*Recipient, error) {
	sharedSecret, err := k.decap(enc)
	if err != nil {
		return nil, err
	}
	context, err := newContext(sharedSecret, k.KEM().ID(), kdf, aead, info)
	if err != nil {
		return nil, err
	}
	return &Recipient{context}, nil
}

// Seal encrypts the provided plaintext, optionally binding to the additional
// public data aad.
//
// Seal uses incrementing counters for each call, and Open on the receiving side
// must be called in the same order as Seal.
func (s *Sender) Seal(aad, plaintext []byte) ([]byte, error) {
	if s.aead == nil {
		return nil, errors.New("export-only instantiation")
	}
	ciphertext := s.aead.Seal(nil, s.nextNonce(), plaintext, aad)
	s.seqNum++
	return ciphertext, nil
}

// Seal instantiates a single-use HPKE sending HPKE context like [NewSender],
// and then encrypts the provided plaintext like [Sender.Seal] (with no aad).
// Seal returns the concatenation of the encapsulated key and the ciphertext.
func Seal(pk PublicKey, kdf KDF, aead AEAD, info, plaintext []byte) ([]byte, error) {
	enc, s, err := NewSender(pk, kdf, aead, info)
	if err != nil {
		return nil, err
	}
	ct, err := s.Seal(nil, plaintext)
	if err != nil {
		return nil, err
	}
	return append(enc, ct...), nil
}

// Export produces a secret value derived from the shared key between sender and
// recipient. length must be at most 65,535.
func (s *Sender) Export(exporterContext string, length int) ([]byte, error) {
	if length < 0 || length > 0xFFFF {
		return nil, errors.New("invalid length")
	}
	return s.export(exporterContext, uint16(length))
}

// Open decrypts the provided ciphertext, optionally binding to the additional
// public data aad, or returns an error if decryption fails.
//
// Open uses incrementing counters for each successful call, and must be called
// in the same order as Seal on the sending side.
func (r *Recipient) Open(aad, ciphertext []byte) ([]byte, error) {
	if r.aead == nil {
		return nil, errors.New("export-only instantiation")
	}
	plaintext, err := r.aead.Open(nil, r.nextNonce(), ciphertext, aad)
	if err != nil {
		return nil, err
	}
	r.seqNum++
	return plaintext, nil
}

// Open instantiates a single-use HPKE receiving HPKE context like [NewRecipient],
// and then decrypts the provided ciphertext like [Recipient.Open] (with no aad).
// ciphertext must be the concatenation of the encapsulated key and the actual ciphertext.
func Open(k PrivateKey, kdf KDF, aead AEAD, info, ciphertext []byte) ([]byte, error) {
	encSize := k.KEM().encSize()
	if len(ciphertext) < encSize {
		return nil, errors.New("ciphertext too short")
	}
	enc, ciphertext := ciphertext[:encSize], ciphertext[encSize:]
	r, err := NewRecipient(enc, k, kdf, aead, info)
	if err != nil {
		return nil, err
	}
	return r.Open(nil, ciphertext)
}

// Export produces a secret value derived from the shared key between sender and
// recipient. length must be at most 65,535.
func (r *Recipient) Export(exporterContext string, length int) ([]byte, error) {
	if length < 0 || length > 0xFFFF {
		return nil, errors.New("invalid length")
	}
	return r.export(exporterContext, uint16(length))
}

func (ctx *context) nextNonce() []byte {
	nonce := make([]byte, ctx.aead.NonceSize())
	byteorder.BEPutUint64(nonce[len(nonce)-8:], ctx.seqNum)
	for i := range ctx.baseNonce {
		nonce[i] ^= ctx.baseNonce[i]
	}
	return nonce
}

func suiteID(kemID, kdfID, aeadID uint16) []byte {
	suiteID := make([]byte, 0, 4+2+2+2)
	suiteID = append(suiteID, []byte("HPKE")...)
	suiteID = byteorder.BEAppendUint16(suiteID, kemID)
	suiteID = byteorder.BEAppendUint16(suiteID, kdfID)
	suiteID = byteorder.BEAppendUint16(suiteID, aeadID)
	return suiteID
}
