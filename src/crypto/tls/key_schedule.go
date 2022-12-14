// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/elliptic"
	"crypto/hmac"
	"errors"
	"fmt"
	"hash"
	"io"
	"math/big"

	"golang.org/x/crypto/cryptobyte"
	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/hkdf"
)

// This file contains the functions necessary to compute the TLS 1.3 key
// schedule. See RFC 8446, Section 7.

const (
	resumptionBinderLabel         = "res binder"
	clientHandshakeTrafficLabel   = "c hs traffic"
	serverHandshakeTrafficLabel   = "s hs traffic"
	clientApplicationTrafficLabel = "c ap traffic"
	serverApplicationTrafficLabel = "s ap traffic"
	exporterLabel                 = "exp master"
	resumptionLabel               = "res master"
	trafficUpdateLabel            = "traffic upd"
)

// expandLabel implements HKDF-Expand-Label from RFC 8446, Section 7.1.
func (c *cipherSuiteTLS13) expandLabel(secret []byte, label string, context []byte, length int) []byte {
	var hkdfLabel cryptobyte.Builder
	hkdfLabel.AddUint16(uint16(length))
	hkdfLabel.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes([]byte("tls13 "))
		b.AddBytes([]byte(label))
	})
	hkdfLabel.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes(context)
	})
	hkdfLabelBytes, err := hkdfLabel.Bytes()
	if err != nil {
		// Rather than calling BytesOrPanic, we explicitly handle this error, in
		// order to provide a reasonable error message. It should be basically
		// impossible for this to panic, and routing errors back through the
		// tree rooted in this function is quite painful. The labels are fixed
		// size, and the context is either a fixed-length computed hash, or
		// parsed from a field which has the same length limitation. As such, an
		// error here is likely to only be caused during development.
		//
		// NOTE: another reasonable approach here might be to return a
		// randomized slice if we encounter an error, which would break the
		// connection, but avoid panicking. This would perhaps be safer but
		// significantly more confusing to users.
		panic(fmt.Errorf("failed to construct HKDF label: %s", err))
	}
	out := make([]byte, length)
	n, err := hkdf.Expand(c.hash.New, secret, hkdfLabelBytes).Read(out)
	if err != nil || n != length {
		panic("tls: HKDF-Expand-Label invocation failed unexpectedly")
	}
	return out
}

// deriveSecret implements Derive-Secret from RFC 8446, Section 7.1.
func (c *cipherSuiteTLS13) deriveSecret(secret []byte, label string, transcript hash.Hash) []byte {
	if transcript == nil {
		transcript = c.hash.New()
	}
	return c.expandLabel(secret, label, transcript.Sum(nil), c.hash.Size())
}

// extract implements HKDF-Extract with the cipher suite hash.
func (c *cipherSuiteTLS13) extract(newSecret, currentSecret []byte) []byte {
	if newSecret == nil {
		newSecret = make([]byte, c.hash.Size())
	}
	return hkdf.Extract(c.hash.New, newSecret, currentSecret)
}

// nextTrafficSecret generates the next traffic secret, given the current one,
// according to RFC 8446, Section 7.2.
func (c *cipherSuiteTLS13) nextTrafficSecret(trafficSecret []byte) []byte {
	return c.expandLabel(trafficSecret, trafficUpdateLabel, nil, c.hash.Size())
}

// trafficKey generates traffic keys according to RFC 8446, Section 7.3.
func (c *cipherSuiteTLS13) trafficKey(trafficSecret []byte) (key, iv []byte) {
	key = c.expandLabel(trafficSecret, "key", nil, c.keyLen)
	iv = c.expandLabel(trafficSecret, "iv", nil, aeadNonceLength)
	return
}

// finishedHash generates the Finished verify_data or PskBinderEntry according
// to RFC 8446, Section 4.4.4. See sections 4.4 and 4.2.11.2 for the baseKey
// selection.
func (c *cipherSuiteTLS13) finishedHash(baseKey []byte, transcript hash.Hash) []byte {
	finishedKey := c.expandLabel(baseKey, "finished", nil, c.hash.Size())
	verifyData := hmac.New(c.hash.New, finishedKey)
	verifyData.Write(transcript.Sum(nil))
	return verifyData.Sum(nil)
}

// exportKeyingMaterial implements RFC5705 exporters for TLS 1.3 according to
// RFC 8446, Section 7.5.
func (c *cipherSuiteTLS13) exportKeyingMaterial(masterSecret []byte, transcript hash.Hash) func(string, []byte, int) ([]byte, error) {
	expMasterSecret := c.deriveSecret(masterSecret, exporterLabel, transcript)
	return func(label string, context []byte, length int) ([]byte, error) {
		secret := c.deriveSecret(expMasterSecret, label, nil)
		h := c.hash.New()
		h.Write(context)
		return c.expandLabel(secret, "exporter", h.Sum(nil), length), nil
	}
}

// ecdheParameters implements Diffie-Hellman with either NIST curves or X25519,
// according to RFC 8446, Section 4.2.8.2.
type ecdheParameters interface {
	CurveID() CurveID
	PublicKey() []byte
	SharedKey(peerPublicKey []byte) []byte
}

func generateECDHEParameters(rand io.Reader, curveID CurveID) (ecdheParameters, error) {
	if curveID == X25519 {
		privateKey := make([]byte, curve25519.ScalarSize)
		if _, err := io.ReadFull(rand, privateKey); err != nil {
			return nil, err
		}
		publicKey, err := curve25519.X25519(privateKey, curve25519.Basepoint)
		if err != nil {
			return nil, err
		}
		return &x25519Parameters{privateKey: privateKey, publicKey: publicKey}, nil
	}

	curve, ok := curveForCurveID(curveID)
	if !ok {
		return nil, errors.New("tls: internal error: unsupported curve")
	}

	p := &nistParameters{curveID: curveID}
	var err error
	p.privateKey, p.x, p.y, err = elliptic.GenerateKey(curve, rand)
	if err != nil {
		return nil, err
	}
	return p, nil
}

func curveForCurveID(id CurveID) (elliptic.Curve, bool) {
	switch id {
	case CurveP256:
		return elliptic.P256(), true
	case CurveP384:
		return elliptic.P384(), true
	case CurveP521:
		return elliptic.P521(), true
	default:
		return nil, false
	}
}

type nistParameters struct {
	privateKey []byte
	x, y       *big.Int // public key
	curveID    CurveID
}

func (p *nistParameters) CurveID() CurveID {
	return p.curveID
}

func (p *nistParameters) PublicKey() []byte {
	curve, _ := curveForCurveID(p.curveID)
	return elliptic.Marshal(curve, p.x, p.y)
}

func (p *nistParameters) SharedKey(peerPublicKey []byte) []byte {
	curve, _ := curveForCurveID(p.curveID)
	// Unmarshal also checks whether the given point is on the curve.
	x, y := elliptic.Unmarshal(curve, peerPublicKey)
	if x == nil {
		return nil
	}

	xShared, _ := curve.ScalarMult(x, y, p.privateKey)
	sharedKey := make([]byte, (curve.Params().BitSize+7)/8)
	return xShared.FillBytes(sharedKey)
}

type x25519Parameters struct {
	privateKey []byte
	publicKey  []byte
}

func (p *x25519Parameters) CurveID() CurveID {
	return X25519
}

func (p *x25519Parameters) PublicKey() []byte {
	return p.publicKey[:]
}

func (p *x25519Parameters) SharedKey(peerPublicKey []byte) []byte {
	sharedKey, err := curve25519.X25519(p.privateKey, peerPublicKey)
	if err != nil {
		return nil
	}
	return sharedKey
}
