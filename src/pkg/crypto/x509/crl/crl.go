// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package crl exposes low-level details of PKIX Certificate Revocation Lists
// as specified in RFC 5280, section 5.
package crl

import (
	"asn1"
	"bytes"
	"encoding/pem"
	"os"
	"time"
)

// CertificateList represents the ASN.1 structure of the same name. See RFC
// 5280, section 5.1. Use crypto/x509/Certificate.CheckCRLSignature to verify
// the signature.
type CertificateList struct {
	TBSCertList        TBSCertificateList
	SignatureAlgorithm AlgorithmIdentifier
	SignatureValue     asn1.BitString
}

// HasExpired returns true iff currentTimeSeconds is past the expiry time of
// certList.
func (certList *CertificateList) HasExpired(currentTimeSeconds int64) bool {
	return certList.TBSCertList.NextUpdate.Seconds() <= currentTimeSeconds
}

// TBSCertificateList represents the ASN.1 structure of the same name. See RFC
// 5280, section 5.1.
type TBSCertificateList struct {
	Raw                 asn1.RawContent
	Version             int "optional,default:2"
	Signature           AlgorithmIdentifier
	Issuer              asn1.RawValue
	ThisUpdate          *time.Time
	NextUpdate          *time.Time
	RevokedCertificates []RevokedCertificate "optional"
	Extensions          []Extension          "tag:0,optional,explicit"
}

// AlgorithmIdentifier represents the ASN.1 structure of the same name. See RFC
// 5280, section 4.1.1.2.
type AlgorithmIdentifier struct {
	Algo   asn1.ObjectIdentifier
	Params asn1.RawValue "optional"
}

// AlgorithmIdentifier represents the ASN.1 structure of the same name. See RFC
// 5280, section 5.1.
type RevokedCertificate struct {
	SerialNumber   asn1.RawValue
	RevocationTime *time.Time
	Extensions     []Extension "optional"
}

// AlgorithmIdentifier represents the ASN.1 structure of the same name. See RFC
// 5280, section 4.2.
type Extension struct {
	Id        asn1.ObjectIdentifier
	IsCritial bool "optional"
	Value     []byte
}

// pemCRLPrefix is the magic string that indicates that we have a PEM encoded
// CRL.
var pemCRLPrefix = []byte("-----BEGIN X509 CRL")
// pemType is the type of a PEM encoded CRL.
var pemType = "X509 CRL"

// Parse parses a CRL from the given bytes. It's often the case that PEM
// encoded CRLs will appear where they should be DER encoded, so this function
// will transparently handle PEM encoding as long as there isn't any leading
// garbage.
func Parse(crlBytes []byte) (certList *CertificateList, err os.Error) {
	if bytes.HasPrefix(crlBytes, pemCRLPrefix) {
		block, _ := pem.Decode(crlBytes)
		if block != nil && block.Type == pemType {
			crlBytes = block.Bytes
		}
	}
	return ParseDER(crlBytes)
}

// ParseDER parses a DER encoded CRL from the given bytes.
func ParseDER(derBytes []byte) (certList *CertificateList, err os.Error) {
	certList = new(CertificateList)
	_, err = asn1.Unmarshal(derBytes, certList)
	if err != nil {
		certList = nil
	}
	return
}
