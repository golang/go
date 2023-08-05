// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package x509

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/internal/boring/fipstls"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509/pkix"
	"fmt"
	"math/big"
	"strings"
	"testing"
	"time"
)

const (
	boringCertCA = iota
	boringCertLeaf
	boringCertFIPSOK = 0x80
)

func boringRSAKey(t *testing.T, size int) *rsa.PrivateKey {
	k, err := rsa.GenerateKey(rand.Reader, size)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

func boringECDSAKey(t *testing.T, curve elliptic.Curve) *ecdsa.PrivateKey {
	k, err := ecdsa.GenerateKey(curve, rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

type boringCertificate struct {
	name      string
	org       string
	parentOrg string
	der       []byte
	cert      *Certificate
	key       interface{}
	fipsOK    bool
}

func TestBoringAllowCert(t *testing.T) {
	R1 := testBoringCert(t, "R1", boringRSAKey(t, 2048), nil, boringCertCA|boringCertFIPSOK)
	R2 := testBoringCert(t, "R2", boringRSAKey(t, 512), nil, boringCertCA)
	R3 := testBoringCert(t, "R3", boringRSAKey(t, 4096), nil, boringCertCA|boringCertFIPSOK)

	M1_R1 := testBoringCert(t, "M1_R1", boringECDSAKey(t, elliptic.P256()), R1, boringCertCA|boringCertFIPSOK)
	M2_R1 := testBoringCert(t, "M2_R1", boringECDSAKey(t, elliptic.P224()), R1, boringCertCA)

	I_R1 := testBoringCert(t, "I_R1", boringRSAKey(t, 3072), R1, boringCertCA|boringCertFIPSOK)
	testBoringCert(t, "I_R2", I_R1.key, R2, boringCertCA|boringCertFIPSOK)
	testBoringCert(t, "I_M1", I_R1.key, M1_R1, boringCertCA|boringCertFIPSOK)
	testBoringCert(t, "I_M2", I_R1.key, M2_R1, boringCertCA|boringCertFIPSOK)

	I_R3 := testBoringCert(t, "I_R3", boringRSAKey(t, 3072), R3, boringCertCA|boringCertFIPSOK)
	testBoringCert(t, "I_R3", I_R3.key, R3, boringCertCA|boringCertFIPSOK)

	testBoringCert(t, "L1_I", boringECDSAKey(t, elliptic.P384()), I_R1, boringCertLeaf|boringCertFIPSOK)
	testBoringCert(t, "L2_I", boringRSAKey(t, 1024), I_R1, boringCertLeaf)
}

func testBoringCert(t *testing.T, name string, key interface{}, parent *boringCertificate, mode int) *boringCertificate {
	org := name
	parentOrg := ""
	if i := strings.Index(org, "_"); i >= 0 {
		org = org[:i]
		parentOrg = name[i+1:]
	}
	tmpl := &Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization: []string{org},
		},
		NotBefore: time.Unix(0, 0),
		NotAfter:  time.Unix(0, 0),

		KeyUsage:              KeyUsageKeyEncipherment | KeyUsageDigitalSignature,
		ExtKeyUsage:           []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}
	if mode&^boringCertFIPSOK == boringCertLeaf {
		tmpl.DNSNames = []string{"example.com"}
	} else {
		tmpl.IsCA = true
		tmpl.KeyUsage |= KeyUsageCertSign
	}

	var pcert *Certificate
	var pkey interface{}
	if parent != nil {
		pcert = parent.cert
		pkey = parent.key
	} else {
		pcert = tmpl
		pkey = key
	}

	var pub interface{}
	var desc string
	switch k := key.(type) {
	case *rsa.PrivateKey:
		pub = &k.PublicKey
		desc = fmt.Sprintf("RSA-%d", k.N.BitLen())
	case *ecdsa.PrivateKey:
		pub = &k.PublicKey
		desc = "ECDSA-" + k.Curve.Params().Name
	default:
		t.Fatalf("invalid key %T", key)
	}

	der, err := CreateCertificate(rand.Reader, tmpl, pcert, pub, pkey)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}

	// Tell isBoringCertificate to enforce FIPS restrictions for this check.
	fipstls.Force()
	defer fipstls.Abandon()

	fipsOK := mode&boringCertFIPSOK != 0
	if boringAllowCert(cert) != fipsOK {
		t.Errorf("boringAllowCert(cert with %s key) = %v, want %v", desc, !fipsOK, fipsOK)
	}
	return &boringCertificate{name, org, parentOrg, der, cert, key, fipsOK}
}
