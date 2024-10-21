// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/ecdh"
	"crypto/hmac"
	"crypto/internal/fips/tls13"
	"crypto/internal/mlkem768"
	"errors"
	"hash"
	"io"

	"golang.org/x/crypto/sha3"
)

// This file contains the functions necessary to compute the TLS 1.3 key
// schedule. See RFC 8446, Section 7.

// nextTrafficSecret generates the next traffic secret, given the current one,
// according to RFC 8446, Section 7.2.
func (c *cipherSuiteTLS13) nextTrafficSecret(trafficSecret []byte) []byte {
	return tls13.ExpandLabel(c.hash.New, trafficSecret, "traffic upd", nil, c.hash.Size())
}

// trafficKey generates traffic keys according to RFC 8446, Section 7.3.
func (c *cipherSuiteTLS13) trafficKey(trafficSecret []byte) (key, iv []byte) {
	key = tls13.ExpandLabel(c.hash.New, trafficSecret, "key", nil, c.keyLen)
	iv = tls13.ExpandLabel(c.hash.New, trafficSecret, "iv", nil, aeadNonceLength)
	return
}

// finishedHash generates the Finished verify_data or PskBinderEntry according
// to RFC 8446, Section 4.4.4. See sections 4.4 and 4.2.11.2 for the baseKey
// selection.
func (c *cipherSuiteTLS13) finishedHash(baseKey []byte, transcript hash.Hash) []byte {
	finishedKey := tls13.ExpandLabel(c.hash.New, baseKey, "finished", nil, c.hash.Size())
	verifyData := hmac.New(c.hash.New, finishedKey)
	verifyData.Write(transcript.Sum(nil))
	return verifyData.Sum(nil)
}

// exportKeyingMaterial implements RFC5705 exporters for TLS 1.3 according to
// RFC 8446, Section 7.5.
func (c *cipherSuiteTLS13) exportKeyingMaterial(s *tls13.MasterSecret, transcript hash.Hash) func(string, []byte, int) ([]byte, error) {
	expMasterSecret := s.ExporterMasterSecret(transcript)
	return func(label string, context []byte, length int) ([]byte, error) {
		return expMasterSecret.Exporter(label, context, length), nil
	}
}

type keySharePrivateKeys struct {
	curveID CurveID
	ecdhe   *ecdh.PrivateKey
	kyber   *mlkem768.DecapsulationKey
}

// kyberDecapsulate implements decapsulation according to Kyber Round 3.
func kyberDecapsulate(dk *mlkem768.DecapsulationKey, c []byte) ([]byte, error) {
	K, err := dk.Decapsulate(c)
	if err != nil {
		return nil, err
	}
	return kyberSharedSecret(c, K), nil
}

// kyberEncapsulate implements encapsulation according to Kyber Round 3.
func kyberEncapsulate(ek []byte) (c, ss []byte, err error) {
	k, err := mlkem768.NewEncapsulationKey(ek)
	if err != nil {
		return nil, nil, err
	}
	c, ss = k.Encapsulate()
	return c, kyberSharedSecret(c, ss), nil
}

func kyberSharedSecret(c, K []byte) []byte {
	// Package mlkem768 implements ML-KEM, which compared to Kyber removed a
	// final hashing step. Compute SHAKE-256(K || SHA3-256(c), 32) to match Kyber.
	// See https://words.filippo.io/mlkem768/#bonus-track-using-a-ml-kem-implementation-as-kyber-v3.
	h := sha3.NewShake256()
	h.Write(K)
	ch := sha3.Sum256(c)
	h.Write(ch[:])
	out := make([]byte, 32)
	h.Read(out)
	return out
}

const x25519PublicKeySize = 32

// generateECDHEKey returns a PrivateKey that implements Diffie-Hellman
// according to RFC 8446, Section 4.2.8.2.
func generateECDHEKey(rand io.Reader, curveID CurveID) (*ecdh.PrivateKey, error) {
	curve, ok := curveForCurveID(curveID)
	if !ok {
		return nil, errors.New("tls: internal error: unsupported curve")
	}

	return curve.GenerateKey(rand)
}

func curveForCurveID(id CurveID) (ecdh.Curve, bool) {
	switch id {
	case X25519:
		return ecdh.X25519(), true
	case CurveP256:
		return ecdh.P256(), true
	case CurveP384:
		return ecdh.P384(), true
	case CurveP521:
		return ecdh.P521(), true
	default:
		return nil, false
	}
}

func curveIDForCurve(curve ecdh.Curve) (CurveID, bool) {
	switch curve {
	case ecdh.X25519():
		return X25519, true
	case ecdh.P256():
		return CurveP256, true
	case ecdh.P384():
		return CurveP384, true
	case ecdh.P521():
		return CurveP521, true
	default:
		return 0, false
	}
}
