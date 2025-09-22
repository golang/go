// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpke

import (
	"bytes"
	"crypto"
	"crypto/ecdh"
	"crypto/mlkem"
	"crypto/rand"
	"crypto/sha3"
	"errors"
)

const (
	mlkem768       = 0x0041 // ML-KEM-768
	mlkem1024      = 0x0042 // ML-KEM-1024
	mlkem768X25519 = 0x647a // MLKEM768-X25519
	mlkem768P256   = 0x0050 // MLKEM768-P256
	mlkem1024P384  = 0x0051 // MLKEM1024-P384
)

var mlkem768X25519Hybrid = hybrid{
	id: mlkem768X25519,
	label: /**/ `\./` +
		/*   */ `/^\`,
}

var mlkem768P256Hybrid = hybrid{
	id:    mlkem768P256,
	label: "MLKEM768-P256",
}

var mlkem1024P384Hybrid = hybrid{
	id:    mlkem1024P384,
	label: "MLKEM1024-P384",
}

type hybrid struct {
	id    uint16
	label string
}

func (x *hybrid) ID() uint16 {
	return x.id
}

func (x *hybrid) sharedSecret(ssPQ, ssT, ctT, ekT []byte) []byte {
	h := sha3.New256()
	h.Write(ssPQ)
	h.Write(ssT)
	h.Write(ctT)
	h.Write(ekT)
	h.Write([]byte(x.label))
	return h.Sum(nil)
}

type hybridSender struct {
	hybrid
	t  *ecdh.PublicKey
	pq crypto.Encapsulator
}

// NewHybridSender returns a KEMSender implementing one of
//
//   - MLKEM768-X25519 (a.k.a. X-Wing)
//   - MLKEM768-P256
//   - MLKEM1024-P384
//
// from draft-ietf-hpke-pq, depending on the underlying curve of t
// ([ecdh.X25519], [ecdh.P256], or [ecdh.P384]) and the type of pq
// (either *[mlkem.EncapsulationKey768] or *[mlkem.EncapsulationKey1024]).
func NewHybridSender(t *ecdh.PublicKey, pq crypto.Encapsulator) (KEMSender, error) {
	switch t.Curve() {
	case ecdh.X25519():
		if _, ok := pq.(*mlkem.EncapsulationKey768); !ok {
			return nil, errors.New("invalid PQ KEM for X25519 hybrid")
		}
		return &hybridSender{mlkem768X25519Hybrid, t, pq}, nil
	case ecdh.P256():
		if _, ok := pq.(*mlkem.EncapsulationKey768); !ok {
			return nil, errors.New("invalid PQ KEM for P-256 hybrid")
		}
		return &hybridSender{mlkem768P256Hybrid, t, pq}, nil
	case ecdh.P384():
		if _, ok := pq.(*mlkem.EncapsulationKey1024); !ok {
			return nil, errors.New("invalid PQ KEM for P-384 hybrid")
		}
		return &hybridSender{mlkem1024P384Hybrid, t, pq}, nil
	default:
		return nil, errors.New("unsupported curve")
	}
}

func (s *hybridSender) Bytes() []byte {
	return append(s.pq.Bytes(), s.t.Bytes()...)
}

var testingOnlyEncapsulate func() (ss, ct []byte)

func (s *hybridSender) encap() (sharedSecret []byte, encapPub []byte, err error) {
	skE, err := s.t.Curve().GenerateKey(rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	if testingOnlyGenerateKey != nil {
		skE = testingOnlyGenerateKey()
	}
	ssT, err := skE.ECDH(s.t)
	if err != nil {
		return nil, nil, err
	}
	ctT := skE.PublicKey().Bytes()

	ssPQ, ctPQ := s.pq.Encapsulate()
	if testingOnlyEncapsulate != nil {
		ssPQ, ctPQ = testingOnlyEncapsulate()
	}

	ss := s.sharedSecret(ssPQ, ssT, ctT, s.t.Bytes())
	ct := append(ctPQ, ctT...)
	return ss, ct, nil
}

type hybridRecipient struct {
	hybrid
	seed []byte // can be nil
	t    ecdh.KeyExchanger
	pq   crypto.Decapsulator
}

// NewHybridRecipient returns a KEMRecipient implementing
//
//   - MLKEM768-X25519 (a.k.a. X-Wing)
//   - MLKEM768-P256
//   - MLKEM1024-P384
//
// from draft-ietf-hpke-pq, depending on the underlying curve of t
// ([ecdh.X25519], [ecdh.P256], or [ecdh.P384]) and the type of pq.Encapsulator()
// (either *[mlkem.EncapsulationKey768] or *[mlkem.EncapsulationKey1024]).
func NewHybridRecipient(t ecdh.KeyExchanger, pq crypto.Decapsulator) (KEMRecipient, error) {
	return newHybridRecipient(t, pq, nil)
}

func newHybridRecipientFromSeed(id uint16, priv []byte) (KEMRecipient, error) {
	if len(priv) != 32 {
		return nil, errors.New("hpke: invalid hybrid KEM secret length")
	}

	s := sha3.NewSHAKE256()
	s.Write(priv)

	seedPQ := make([]byte, mlkem.SeedSize)
	s.Read(seedPQ)

	var pq crypto.Decapsulator
	switch id {
	case mlkem768X25519, mlkem768P256:
		sk, err := mlkem.NewDecapsulationKey768(seedPQ)
		if err != nil {
			return nil, err
		}
		pq = sk
	case mlkem1024P384:
		sk, err := mlkem.NewDecapsulationKey1024(seedPQ)
		if err != nil {
			return nil, err
		}
		pq = sk
	default:
		return nil, errors.New("hpke: invalid hybrid KEM ID")
	}

	var seedT []byte
	var curve ecdh.Curve
	switch id {
	case mlkem768X25519:
		seedT = make([]byte, 32)
		curve = ecdh.X25519()
	case mlkem768P256:
		seedT = make([]byte, 32)
		curve = ecdh.P256()
	case mlkem1024P384:
		seedT = make([]byte, 48)
		curve = ecdh.P384()
	default:
		return nil, errors.New("hpke: invalid hybrid KEM ID")
	}

	for {
		s.Read(seedT)
		k, err := curve.NewPrivateKey(seedT)
		if err != nil {
			continue
		}
		return newHybridRecipient(k, pq, priv)
	}
}

func newHybridRecipient(t ecdh.KeyExchanger, pq crypto.Decapsulator, seed []byte) (KEMRecipient, error) {
	switch t.Curve() {
	case ecdh.X25519():
		if _, ok := pq.Encapsulator().(*mlkem.EncapsulationKey768); !ok {
			return nil, errors.New("invalid PQ KEM for X25519 hybrid")
		}
		return &hybridRecipient{mlkem768X25519Hybrid, bytes.Clone(seed), t, pq}, nil
	case ecdh.P256():
		if _, ok := pq.Encapsulator().(*mlkem.EncapsulationKey768); !ok {
			return nil, errors.New("invalid PQ KEM for P-256 hybrid")
		}
		return &hybridRecipient{mlkem768P256Hybrid, bytes.Clone(seed), t, pq}, nil
	case ecdh.P384():
		if _, ok := pq.Encapsulator().(*mlkem.EncapsulationKey1024); !ok {
			return nil, errors.New("invalid PQ KEM for P-384 hybrid")
		}
		return &hybridRecipient{mlkem1024P384Hybrid, bytes.Clone(seed), t, pq}, nil
	default:
		return nil, errors.New("unsupported curve")
	}
}

func (r *hybridRecipient) Bytes() ([]byte, error) {
	if r.seed == nil {
		return nil, errors.New("private key seed not available")
	}
	return r.seed, nil
}

func (r *hybridRecipient) KEMSender() KEMSender {
	return &hybridSender{
		hybrid: r.hybrid,
		t:      r.t.PublicKey(),
		pq:     r.pq.Encapsulator(),
	}
}

func (r *hybridRecipient) decap(enc []byte) ([]byte, error) {
	var ctPQ, ctT []byte
	switch r.id {
	case mlkem768X25519:
		if len(enc) != mlkem.CiphertextSize768+32 {
			return nil, errors.New("invalid encapsulated key size")
		}
		ctPQ, ctT = enc[:mlkem.CiphertextSize768], enc[mlkem.CiphertextSize768:]
	case mlkem768P256:
		if len(enc) != mlkem.CiphertextSize768+65 {
			return nil, errors.New("invalid encapsulated key size")
		}
		ctPQ, ctT = enc[:mlkem.CiphertextSize768], enc[mlkem.CiphertextSize768:]
	case mlkem1024P384:
		if len(enc) != mlkem.CiphertextSize1024+97 {
			return nil, errors.New("invalid encapsulated key size")
		}
		ctPQ, ctT = enc[:mlkem.CiphertextSize1024], enc[mlkem.CiphertextSize1024:]
	default:
		return nil, errors.New("internal error: unsupported KEM")
	}
	ssPQ, err := r.pq.Decapsulate(ctPQ)
	if err != nil {
		return nil, err
	}
	pub, err := r.t.Curve().NewPublicKey(ctT)
	if err != nil {
		return nil, err
	}
	ssT, err := r.t.ECDH(pub)
	if err != nil {
		return nil, err
	}
	ss := r.sharedSecret(ssPQ, ssT, ctT, r.t.PublicKey().Bytes())
	return ss, nil
}

type mlkemSender struct {
	id uint16
	pq interface {
		Bytes() []byte
		Encapsulate() (sharedKey []byte, ciphertext []byte)
	}
}

// NewMLKEMSender returns a KEMSender implementing ML-KEM-768 or ML-KEM-1024 from
// draft-ietf-hpke-pq. pub must be either a *[mlkem.EncapsulationKey768] or a
// *[mlkem.EncapsulationKey1024].
func NewMLKEMSender(pub crypto.Encapsulator) (KEMSender, error) {
	switch pub.(type) {
	case *mlkem.EncapsulationKey768:
		return &mlkemSender{
			id: mlkem768,
			pq: pub,
		}, nil
	case *mlkem.EncapsulationKey1024:
		return &mlkemSender{
			id: mlkem1024,
			pq: pub,
		}, nil
	default:
		return nil, errors.New("unsupported public key type")
	}
}

func (s *mlkemSender) ID() uint16 {
	return s.id
}

func (s *mlkemSender) Bytes() []byte {
	return s.pq.Bytes()
}

func (s *mlkemSender) encap() (sharedSecret []byte, encapPub []byte, err error) {
	ss, ct := s.pq.Encapsulate()
	if testingOnlyEncapsulate != nil {
		ss, ct = testingOnlyEncapsulate()
	}
	return ss, ct, nil
}

type mlkemRecipient struct {
	id uint16
	pq crypto.Decapsulator
}

// NewMLKEMRecipient returns a KEMRecipient implementing ML-KEM-768 or ML-KEM-1024
// from draft-ietf-hpke-pq. priv.Encapsulator() must return either a
// *[mlkem.EncapsulationKey768] or a *[mlkem.EncapsulationKey1024].
func NewMLKEMRecipient(priv crypto.Decapsulator) (KEMRecipient, error) {
	switch priv.Encapsulator().(type) {
	case *mlkem.EncapsulationKey768:
		return &mlkemRecipient{
			id: mlkem768,
			pq: priv,
		}, nil
	case *mlkem.EncapsulationKey1024:
		return &mlkemRecipient{
			id: mlkem1024,
			pq: priv,
		}, nil
	default:
		return nil, errors.New("unsupported public key type")
	}
}

func (r *mlkemRecipient) ID() uint16 {
	return r.id
}

func (r *mlkemRecipient) Bytes() ([]byte, error) {
	pq, ok := r.pq.(interface {
		Bytes() []byte
	})
	if !ok {
		return nil, errors.New("private key seed not available")
	}
	return pq.Bytes(), nil
}

func (r *mlkemRecipient) KEMSender() KEMSender {
	s := &mlkemSender{
		id: r.id,
		pq: r.pq.Encapsulator(),
	}
	return s
}

func (r *mlkemRecipient) decap(enc []byte) ([]byte, error) {
	switch r.id {
	case mlkem768:
		if len(enc) != mlkem.CiphertextSize768 {
			return nil, errors.New("invalid encapsulated key size")
		}
	case mlkem1024:
		if len(enc) != mlkem.CiphertextSize1024 {
			return nil, errors.New("invalid encapsulated key size")
		}
	default:
		return nil, errors.New("internal error: unsupported KEM")
	}
	return r.pq.Decapsulate(enc)
}
