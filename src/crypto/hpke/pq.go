// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpke

import (
	"bytes"
	"crypto"
	"crypto/ecdh"
	"crypto/fips140"
	"crypto/internal/fips140/drbg"
	"crypto/internal/rand"
	"crypto/mlkem"
	"crypto/sha3"
	"errors"
	"internal/byteorder"
)

var mlkem768X25519 = &hybridKEM{
	id: 0x647a,
	label: /**/ `\./` +
		/*   */ `/^\`,
	curve: ecdh.X25519(),

	curveSeedSize:    32,
	curvePointSize:   32,
	pqEncapsKeySize:  mlkem.EncapsulationKeySize768,
	pqCiphertextSize: mlkem.CiphertextSize768,

	pqNewPublicKey: func(data []byte) (crypto.Encapsulator, error) {
		return mlkem.NewEncapsulationKey768(data)
	},
	pqNewPrivateKey: func(data []byte) (crypto.Decapsulator, error) {
		return mlkem.NewDecapsulationKey768(data)
	},
}

// MLKEM768X25519 returns a KEM implementing MLKEM768-X25519 (a.k.a. X-Wing)
// from draft-ietf-hpke-pq.
func MLKEM768X25519() KEM {
	return mlkem768X25519
}

var mlkem768P256 = &hybridKEM{
	id:    0x0050,
	label: "MLKEM768-P256",
	curve: ecdh.P256(),

	curveSeedSize:    32,
	curvePointSize:   65,
	pqEncapsKeySize:  mlkem.EncapsulationKeySize768,
	pqCiphertextSize: mlkem.CiphertextSize768,

	pqNewPublicKey: func(data []byte) (crypto.Encapsulator, error) {
		return mlkem.NewEncapsulationKey768(data)
	},
	pqNewPrivateKey: func(data []byte) (crypto.Decapsulator, error) {
		return mlkem.NewDecapsulationKey768(data)
	},
}

// MLKEM768P256 returns a KEM implementing MLKEM768-P256 from draft-ietf-hpke-pq.
func MLKEM768P256() KEM {
	return mlkem768P256
}

var mlkem1024P384 = &hybridKEM{
	id:    0x0051,
	label: "MLKEM1024-P384",
	curve: ecdh.P384(),

	curveSeedSize:    48,
	curvePointSize:   97,
	pqEncapsKeySize:  mlkem.EncapsulationKeySize1024,
	pqCiphertextSize: mlkem.CiphertextSize1024,

	pqNewPublicKey: func(data []byte) (crypto.Encapsulator, error) {
		return mlkem.NewEncapsulationKey1024(data)
	},
	pqNewPrivateKey: func(data []byte) (crypto.Decapsulator, error) {
		return mlkem.NewDecapsulationKey1024(data)
	},
}

// MLKEM1024P384 returns a KEM implementing MLKEM1024-P384 from draft-ietf-hpke-pq.
func MLKEM1024P384() KEM {
	return mlkem1024P384
}

type hybridKEM struct {
	id    uint16
	label string
	curve ecdh.Curve

	curveSeedSize    int
	curvePointSize   int
	pqEncapsKeySize  int
	pqCiphertextSize int

	pqNewPublicKey  func(data []byte) (crypto.Encapsulator, error)
	pqNewPrivateKey func(data []byte) (crypto.Decapsulator, error)
}

func (kem *hybridKEM) ID() uint16 {
	return kem.id
}

func (kem *hybridKEM) encSize() int {
	return kem.pqCiphertextSize + kem.curvePointSize
}

func (kem *hybridKEM) sharedSecret(ssPQ, ssT, ctT, ekT []byte) []byte {
	h := sha3.New256()
	h.Write(ssPQ)
	h.Write(ssT)
	h.Write(ctT)
	h.Write(ekT)
	h.Write([]byte(kem.label))
	return h.Sum(nil)
}

type hybridPublicKey struct {
	kem *hybridKEM
	t   *ecdh.PublicKey
	pq  crypto.Encapsulator
}

// NewHybridPublicKey returns a PublicKey implementing one of
//
//   - MLKEM768-X25519 (a.k.a. X-Wing)
//   - MLKEM768-P256
//   - MLKEM1024-P384
//
// from draft-ietf-hpke-pq, depending on the underlying curve of t
// ([ecdh.X25519], [ecdh.P256], or [ecdh.P384]) and the type of pq (either
// *[mlkem.EncapsulationKey768] or *[mlkem.EncapsulationKey1024]).
//
// This function is meant for applications that already have instantiated
// crypto/ecdh and crypto/mlkem public keys. Otherwise, applications should use
// the [KEM.NewPublicKey] method of e.g. [MLKEM768X25519].
func NewHybridPublicKey(pq crypto.Encapsulator, t *ecdh.PublicKey) (PublicKey, error) {
	switch t.Curve() {
	case ecdh.X25519():
		if _, ok := pq.(*mlkem.EncapsulationKey768); !ok {
			return nil, errors.New("invalid PQ KEM for X25519 hybrid")
		}
		return &hybridPublicKey{mlkem768X25519, t, pq}, nil
	case ecdh.P256():
		if _, ok := pq.(*mlkem.EncapsulationKey768); !ok {
			return nil, errors.New("invalid PQ KEM for P-256 hybrid")
		}
		return &hybridPublicKey{mlkem768P256, t, pq}, nil
	case ecdh.P384():
		if _, ok := pq.(*mlkem.EncapsulationKey1024); !ok {
			return nil, errors.New("invalid PQ KEM for P-384 hybrid")
		}
		return &hybridPublicKey{mlkem1024P384, t, pq}, nil
	default:
		return nil, errors.New("unsupported curve")
	}
}

func (kem *hybridKEM) NewPublicKey(data []byte) (PublicKey, error) {
	if len(data) != kem.pqEncapsKeySize+kem.curvePointSize {
		return nil, errors.New("invalid public key size")
	}
	pq, err := kem.pqNewPublicKey(data[:kem.pqEncapsKeySize])
	if err != nil {
		return nil, err
	}
	var k *ecdh.PublicKey
	fips140.WithoutEnforcement(func() { // Hybrid of ML-KEM, which is Approved.
		k, err = kem.curve.NewPublicKey(data[kem.pqEncapsKeySize:])
	})
	if err != nil {
		return nil, err
	}
	return NewHybridPublicKey(pq, k)
}

func (pk *hybridPublicKey) KEM() KEM {
	return pk.kem
}

func (pk *hybridPublicKey) Bytes() []byte {
	return append(pk.pq.Bytes(), pk.t.Bytes()...)
}

var testingOnlyEncapsulate func() (ss, ct []byte)

func (pk *hybridPublicKey) encap() (sharedSecret []byte, encapPub []byte, err error) {
	var skE *ecdh.PrivateKey
	fips140.WithoutEnforcement(func() { // Hybrid of ML-KEM, which is Approved.
		skE, err = pk.t.Curve().GenerateKey(rand.Reader)
	})
	if err != nil {
		return nil, nil, err
	}
	if testingOnlyGenerateKey != nil {
		skE = testingOnlyGenerateKey()
	}
	var ssT []byte
	fips140.WithoutEnforcement(func() {
		ssT, err = skE.ECDH(pk.t)
	})
	if err != nil {
		return nil, nil, err
	}
	ctT := skE.PublicKey().Bytes()

	ssPQ, ctPQ := pk.pq.Encapsulate()
	if testingOnlyEncapsulate != nil {
		ssPQ, ctPQ = testingOnlyEncapsulate()
	}

	ss := pk.kem.sharedSecret(ssPQ, ssT, ctT, pk.t.Bytes())
	ct := append(ctPQ, ctT...)
	return ss, ct, nil
}

type hybridPrivateKey struct {
	kem  *hybridKEM
	seed []byte // can be nil
	t    ecdh.KeyExchanger
	pq   crypto.Decapsulator
}

// NewHybridPrivateKey returns a PrivateKey implementing
//
//   - MLKEM768-X25519 (a.k.a. X-Wing)
//   - MLKEM768-P256
//   - MLKEM1024-P384
//
// from draft-ietf-hpke-pq, depending on the underlying curve of t
// ([ecdh.X25519], [ecdh.P256], or [ecdh.P384]) and the type of pq.Encapsulator()
// (either *[mlkem.EncapsulationKey768] or *[mlkem.EncapsulationKey1024]).
//
// This function is meant for applications that already have instantiated
// crypto/ecdh and crypto/mlkem private keys, or another implementation of a
// [ecdh.KeyExchanger] and [crypto.Decapsulator] (e.g. a hardware key).
// Otherwise, applications should use the [KEM.NewPrivateKey] method of e.g.
// [MLKEM768X25519].
func NewHybridPrivateKey(pq crypto.Decapsulator, t ecdh.KeyExchanger) (PrivateKey, error) {
	return newHybridPrivateKey(pq, t, nil)
}

func (kem *hybridKEM) GenerateKey() (PrivateKey, error) {
	seed := make([]byte, 32)
	drbg.Read(seed)
	return kem.NewPrivateKey(seed)
}

func (kem *hybridKEM) NewPrivateKey(priv []byte) (PrivateKey, error) {
	if len(priv) != 32 {
		return nil, errors.New("hpke: invalid hybrid KEM secret length")
	}

	s := sha3.NewSHAKE256()
	s.Write(priv)

	seedPQ := make([]byte, mlkem.SeedSize)
	s.Read(seedPQ)
	pq, err := kem.pqNewPrivateKey(seedPQ)
	if err != nil {
		return nil, err
	}

	seedT := make([]byte, kem.curveSeedSize)
	for {
		s.Read(seedT)
		var k ecdh.KeyExchanger
		fips140.WithoutEnforcement(func() { // Hybrid of ML-KEM, which is Approved.
			k, err = kem.curve.NewPrivateKey(seedT)
		})
		if err != nil {
			continue
		}
		return newHybridPrivateKey(pq, k, priv)
	}
}

func newHybridPrivateKey(pq crypto.Decapsulator, t ecdh.KeyExchanger, seed []byte) (PrivateKey, error) {
	switch t.Curve() {
	case ecdh.X25519():
		if _, ok := pq.Encapsulator().(*mlkem.EncapsulationKey768); !ok {
			return nil, errors.New("invalid PQ KEM for X25519 hybrid")
		}
		return &hybridPrivateKey{mlkem768X25519, bytes.Clone(seed), t, pq}, nil
	case ecdh.P256():
		if _, ok := pq.Encapsulator().(*mlkem.EncapsulationKey768); !ok {
			return nil, errors.New("invalid PQ KEM for P-256 hybrid")
		}
		return &hybridPrivateKey{mlkem768P256, bytes.Clone(seed), t, pq}, nil
	case ecdh.P384():
		if _, ok := pq.Encapsulator().(*mlkem.EncapsulationKey1024); !ok {
			return nil, errors.New("invalid PQ KEM for P-384 hybrid")
		}
		return &hybridPrivateKey{mlkem1024P384, bytes.Clone(seed), t, pq}, nil
	default:
		return nil, errors.New("unsupported curve")
	}
}

func (kem *hybridKEM) DeriveKeyPair(ikm []byte) (PrivateKey, error) {
	suiteID := byteorder.BEAppendUint16([]byte("KEM"), kem.id)
	dk, err := SHAKE256().labeledDerive(suiteID, ikm, "DeriveKeyPair", nil, 32)
	if err != nil {
		return nil, err
	}
	return kem.NewPrivateKey(dk)
}

func (k *hybridPrivateKey) KEM() KEM {
	return k.kem
}

func (k *hybridPrivateKey) Bytes() ([]byte, error) {
	if k.seed == nil {
		return nil, errors.New("private key seed not available")
	}
	return k.seed, nil
}

func (k *hybridPrivateKey) PublicKey() PublicKey {
	return &hybridPublicKey{
		kem: k.kem,
		t:   k.t.PublicKey(),
		pq:  k.pq.Encapsulator(),
	}
}

func (k *hybridPrivateKey) decap(enc []byte) ([]byte, error) {
	if len(enc) != k.kem.pqCiphertextSize+k.kem.curvePointSize {
		return nil, errors.New("invalid encapsulated key size")
	}
	ctPQ, ctT := enc[:k.kem.pqCiphertextSize], enc[k.kem.pqCiphertextSize:]
	ssPQ, err := k.pq.Decapsulate(ctPQ)
	if err != nil {
		return nil, err
	}
	var pub *ecdh.PublicKey
	fips140.WithoutEnforcement(func() { // Hybrid of ML-KEM, which is Approved.
		pub, err = k.t.Curve().NewPublicKey(ctT)
	})
	if err != nil {
		return nil, err
	}
	var ssT []byte
	fips140.WithoutEnforcement(func() {
		ssT, err = k.t.ECDH(pub)
	})
	if err != nil {
		return nil, err
	}
	ss := k.kem.sharedSecret(ssPQ, ssT, ctT, k.t.PublicKey().Bytes())
	return ss, nil
}

var mlkem768 = &mlkemKEM{
	id:             0x0041,
	ciphertextSize: mlkem.CiphertextSize768,
	newPublicKey: func(data []byte) (crypto.Encapsulator, error) {
		return mlkem.NewEncapsulationKey768(data)
	},
	newPrivateKey: func(data []byte) (crypto.Decapsulator, error) {
		return mlkem.NewDecapsulationKey768(data)
	},
	generateKey: func() (crypto.Decapsulator, error) {
		return mlkem.GenerateKey768()
	},
}

// MLKEM768 returns a KEM implementing ML-KEM-768 from draft-ietf-hpke-pq.
func MLKEM768() KEM {
	return mlkem768
}

var mlkem1024 = &mlkemKEM{
	id:             0x0042,
	ciphertextSize: mlkem.CiphertextSize1024,
	newPublicKey: func(data []byte) (crypto.Encapsulator, error) {
		return mlkem.NewEncapsulationKey1024(data)
	},
	newPrivateKey: func(data []byte) (crypto.Decapsulator, error) {
		return mlkem.NewDecapsulationKey1024(data)
	},
	generateKey: func() (crypto.Decapsulator, error) {
		return mlkem.GenerateKey1024()
	},
}

// MLKEM1024 returns a KEM implementing ML-KEM-1024 from draft-ietf-hpke-pq.
func MLKEM1024() KEM {
	return mlkem1024
}

type mlkemKEM struct {
	id             uint16
	ciphertextSize int
	newPublicKey   func(data []byte) (crypto.Encapsulator, error)
	newPrivateKey  func(data []byte) (crypto.Decapsulator, error)
	generateKey    func() (crypto.Decapsulator, error)
}

func (kem *mlkemKEM) ID() uint16 {
	return kem.id
}

func (kem *mlkemKEM) encSize() int {
	return kem.ciphertextSize
}

type mlkemPublicKey struct {
	kem *mlkemKEM
	pq  crypto.Encapsulator
}

// NewMLKEMPublicKey returns a KEMPublicKey implementing
//
//   - ML-KEM-768
//   - ML-KEM-1024
//
// from draft-ietf-hpke-pq, depending on the type of pub
// (*[mlkem.EncapsulationKey768] or *[mlkem.EncapsulationKey1024]).
//
// This function is meant for applications that already have an instantiated
// crypto/mlkem public key. Otherwise, applications should use the
// [KEM.NewPublicKey] method of e.g. [MLKEM768].
func NewMLKEMPublicKey(pub crypto.Encapsulator) (PublicKey, error) {
	switch pub.(type) {
	case *mlkem.EncapsulationKey768:
		return &mlkemPublicKey{mlkem768, pub}, nil
	case *mlkem.EncapsulationKey1024:
		return &mlkemPublicKey{mlkem1024, pub}, nil
	default:
		return nil, errors.New("unsupported public key type")
	}
}

func (kem *mlkemKEM) NewPublicKey(data []byte) (PublicKey, error) {
	pq, err := kem.newPublicKey(data)
	if err != nil {
		return nil, err
	}
	return NewMLKEMPublicKey(pq)
}

func (pk *mlkemPublicKey) KEM() KEM {
	return pk.kem
}

func (pk *mlkemPublicKey) Bytes() []byte {
	return pk.pq.Bytes()
}

func (pk *mlkemPublicKey) encap() (sharedSecret []byte, encapPub []byte, err error) {
	ss, ct := pk.pq.Encapsulate()
	if testingOnlyEncapsulate != nil {
		ss, ct = testingOnlyEncapsulate()
	}
	return ss, ct, nil
}

type mlkemPrivateKey struct {
	kem *mlkemKEM
	pq  crypto.Decapsulator
}

// NewMLKEMPrivateKey returns a KEMPrivateKey implementing
//
//   - ML-KEM-768
//   - ML-KEM-1024
//
// from draft-ietf-hpke-pq, depending on the type of priv.Encapsulator()
// (either *[mlkem.EncapsulationKey768] or *[mlkem.EncapsulationKey1024]).
//
// This function is meant for applications that already have an instantiated
// crypto/mlkem private key. Otherwise, applications should use the
// [KEM.NewPrivateKey] method of e.g. [MLKEM768].
func NewMLKEMPrivateKey(priv crypto.Decapsulator) (PrivateKey, error) {
	switch priv.Encapsulator().(type) {
	case *mlkem.EncapsulationKey768:
		return &mlkemPrivateKey{mlkem768, priv}, nil
	case *mlkem.EncapsulationKey1024:
		return &mlkemPrivateKey{mlkem1024, priv}, nil
	default:
		return nil, errors.New("unsupported public key type")
	}
}

func (kem *mlkemKEM) GenerateKey() (PrivateKey, error) {
	pq, err := kem.generateKey()
	if err != nil {
		return nil, err
	}
	return NewMLKEMPrivateKey(pq)
}

func (kem *mlkemKEM) NewPrivateKey(priv []byte) (PrivateKey, error) {
	pq, err := kem.newPrivateKey(priv)
	if err != nil {
		return nil, err
	}
	return NewMLKEMPrivateKey(pq)
}

func (kem *mlkemKEM) DeriveKeyPair(ikm []byte) (PrivateKey, error) {
	suiteID := byteorder.BEAppendUint16([]byte("KEM"), kem.id)
	dk, err := SHAKE256().labeledDerive(suiteID, ikm, "DeriveKeyPair", nil, 64)
	if err != nil {
		return nil, err
	}
	return kem.NewPrivateKey(dk)
}

func (k *mlkemPrivateKey) KEM() KEM {
	return k.kem
}

func (k *mlkemPrivateKey) Bytes() ([]byte, error) {
	pq, ok := k.pq.(interface {
		Bytes() []byte
	})
	if !ok {
		return nil, errors.New("private key seed not available")
	}
	return pq.Bytes(), nil
}

func (k *mlkemPrivateKey) PublicKey() PublicKey {
	return &mlkemPublicKey{
		kem: k.kem,
		pq:  k.pq.Encapsulator(),
	}
}

func (k *mlkemPrivateKey) decap(enc []byte) ([]byte, error) {
	return k.pq.Decapsulate(enc)
}
