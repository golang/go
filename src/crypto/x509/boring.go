// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package x509

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/internal/boring/fipstls"
	"crypto/rsa"
)

// boringAllowCert reports whether c is allowed to be used
// in a certificate chain by the current fipstls enforcement setting.
// It is called for each leaf, intermediate, and root certificate.
func boringAllowCert(c *Certificate) bool {
	if !fipstls.Required() {
		return true
	}

	// The key must be RSA 2048, RSA 3072, RSA 4096,
	// or ECDSA P-256, P-384, P-521.
	switch k := c.PublicKey.(type) {
	default:
		return false
	case *rsa.PublicKey:
		if size := k.N.BitLen(); size != 2048 && size != 3072 && size != 4096 {
			return false
		}
	case *ecdsa.PublicKey:
		if k.Curve != elliptic.P256() && k.Curve != elliptic.P384() && k.Curve != elliptic.P521() {
			return false
		}
	}
	return true
}
