// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/x509";
	"encoding/pem";
)

// A CASet is a set of certificates.
type CASet struct {
	bySubjectKeyId	map[string]*x509.Certificate;
	byName		map[string]*x509.Certificate;
}

func NewCASet() *CASet {
	return &CASet{
		make(map[string]*x509.Certificate),
		make(map[string]*x509.Certificate),
	}
}

func nameToKey(name *x509.Name) string {
	return name.Country + "/" + name.OrganizationalUnit + "/" + name.OrganizationalUnit + "/" + name.CommonName
}

// FindParent attempts to find the certificate in s which signs the given
// certificate. If no such certificate can be found, it returns nil.
func (s *CASet) FindParent(cert *x509.Certificate) (parent *x509.Certificate) {
	var ok bool;

	if len(cert.AuthorityKeyId) > 0 {
		parent, ok = s.bySubjectKeyId[string(cert.AuthorityKeyId)]
	} else {
		parent, ok = s.byName[nameToKey(&cert.Issuer)]
	}

	if !ok {
		return nil
	}
	return parent;
}

// SetFromPEM attempts to parse a series of PEM encoded root certificates. It
// appends any certificates found to s and returns true if any certificates
// were successfully parsed. On many Linux systems, /etc/ssl/cert.pem will
// contains the system wide set of root CAs in a format suitable for this
// function.
func (s *CASet) SetFromPEM(pemCerts []byte) (ok bool) {
	for len(pemCerts) > 0 {
		var block *pem.Block;
		block, pemCerts = pem.Decode(pemCerts);
		if block == nil {
			break
		}
		if block.Type != "CERTIFICATE" || len(block.Headers) != 0 {
			continue
		}

		cert, err := x509.ParseCertificate(block.Bytes);
		if err != nil {
			continue
		}

		if len(cert.SubjectKeyId) > 0 {
			s.bySubjectKeyId[string(cert.SubjectKeyId)] = cert
		}
		s.byName[nameToKey(&cert.Subject)] = cert;
		ok = true;
	}

	return;
}
