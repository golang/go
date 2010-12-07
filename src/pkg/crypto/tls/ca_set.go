// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/x509"
	"encoding/pem"
	"strings"
)

// A CASet is a set of certificates.
type CASet struct {
	bySubjectKeyId map[string][]*x509.Certificate
	byName         map[string][]*x509.Certificate
}

// NewCASet returns a new, empty CASet.
func NewCASet() *CASet {
	return &CASet{
		make(map[string][]*x509.Certificate),
		make(map[string][]*x509.Certificate),
	}
}

func nameToKey(name *x509.Name) string {
	return strings.Join(name.Country, ",") + "/" + strings.Join(name.Organization, ",") + "/" + strings.Join(name.OrganizationalUnit, ",") + "/" + name.CommonName
}

// FindVerifiedParent attempts to find the certificate in s which has signed
// the given certificate. If no such certificate can be found or the signature
// doesn't match, it returns nil.
func (s *CASet) FindVerifiedParent(cert *x509.Certificate) (parent *x509.Certificate) {
	var candidates []*x509.Certificate

	if len(cert.AuthorityKeyId) > 0 {
		candidates = s.bySubjectKeyId[string(cert.AuthorityKeyId)]
	}
	if len(candidates) == 0 {
		candidates = s.byName[nameToKey(&cert.Issuer)]
	}

	for _, c := range candidates {
		if cert.CheckSignatureFrom(c) == nil {
			return c
		}
	}

	return nil
}

// AddCert adds a certificate to the set
func (s *CASet) AddCert(cert *x509.Certificate) {
	if len(cert.SubjectKeyId) > 0 {
		keyId := string(cert.SubjectKeyId)
		s.bySubjectKeyId[keyId] = append(s.bySubjectKeyId[keyId], cert)
	}
	name := nameToKey(&cert.Subject)
	s.byName[name] = append(s.byName[name], cert)
}

// SetFromPEM attempts to parse a series of PEM encoded root certificates. It
// appends any certificates found to s and returns true if any certificates
// were successfully parsed. On many Linux systems, /etc/ssl/cert.pem will
// contains the system wide set of root CAs in a format suitable for this
// function.
func (s *CASet) SetFromPEM(pemCerts []byte) (ok bool) {
	for len(pemCerts) > 0 {
		var block *pem.Block
		block, pemCerts = pem.Decode(pemCerts)
		if block == nil {
			break
		}
		if block.Type != "CERTIFICATE" || len(block.Headers) != 0 {
			continue
		}

		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			continue
		}

		s.AddCert(cert)
		ok = true
	}

	return
}
