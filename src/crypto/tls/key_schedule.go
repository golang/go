// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"golang_org/x/crypto/cryptobyte"
	"golang_org/x/crypto/hkdf"
	"hash"
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
	out := make([]byte, length)
	n, err := hkdf.Expand(c.hash.New, secret, hkdfLabel.BytesOrPanic()).Read(out)
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
