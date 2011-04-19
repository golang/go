// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"encoding/pem"
	"strings"
)

// Roots is a set of certificates.
type CertPool struct {
	bySubjectKeyId map[string][]*Certificate
	byName         map[string][]*Certificate
}

// NewCertPool returns a new, empty CertPool.
func NewCertPool() *CertPool {
	return &CertPool{
		make(map[string][]*Certificate),
		make(map[string][]*Certificate),
	}
}

func nameToKey(name *Name) string {
	return strings.Join(name.Country, ",") + "/" + strings.Join(name.Organization, ",") + "/" + strings.Join(name.OrganizationalUnit, ",") + "/" + name.CommonName
}

// FindVerifiedParents attempts to find certificates in s which have signed the
// given certificate. If no such certificate can be found or the signature
// doesn't match, it returns nil.
func (s *CertPool) FindVerifiedParents(cert *Certificate) (parents []*Certificate) {
	var candidates []*Certificate

	if len(cert.AuthorityKeyId) > 0 {
		candidates = s.bySubjectKeyId[string(cert.AuthorityKeyId)]
	}
	if len(candidates) == 0 {
		candidates = s.byName[nameToKey(&cert.Issuer)]
	}

	for _, c := range candidates {
		if cert.CheckSignatureFrom(c) == nil {
			parents = append(parents, c)
		}
	}

	return
}

// AddCert adds a certificate to a pool.
func (s *CertPool) AddCert(cert *Certificate) {
	if len(cert.SubjectKeyId) > 0 {
		keyId := string(cert.SubjectKeyId)
		s.bySubjectKeyId[keyId] = append(s.bySubjectKeyId[keyId], cert)
	}
	name := nameToKey(&cert.Subject)
	s.byName[name] = append(s.byName[name], cert)
}

// AppendCertsFromPEM attempts to parse a series of PEM encoded root
// certificates. It appends any certificates found to s and returns true if any
// certificates were successfully parsed.
//
// On many Linux systems, /etc/ssl/cert.pem will contains the system wide set
// of root CAs in a format suitable for this function.
func (s *CertPool) AppendCertsFromPEM(pemCerts []byte) (ok bool) {
	for len(pemCerts) > 0 {
		var block *pem.Block
		block, pemCerts = pem.Decode(pemCerts)
		if block == nil {
			break
		}
		if block.Type != "CERTIFICATE" || len(block.Headers) != 0 {
			continue
		}

		cert, err := ParseCertificate(block.Bytes)
		if err != nil {
			continue
		}

		s.AddCert(cert)
		ok = true
	}

	return
}
