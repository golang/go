// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"crypto/sha256"
	"encoding/pem"
	"os"
	"sync"
)

type sum224 [sha256.Size224]byte

// CertPool is a set of certificates.
type CertPool struct {
	byName map[string][]int // cert.RawSubject => index into lazyCerts

	// lazyCerts contains funcs that return a certificate,
	// lazily parsing/decompressing it as needed.
	lazyCerts []lazyCert

	// haveSum maps from sum224(cert.Raw) to true. It's used only
	// for AddCert duplicate detection, to avoid CertPool.contains
	// calls in the AddCert path (because the contains method can
	// call getCert and otherwise negate savings from lazy getCert
	// funcs).
	haveSum map[sum224]bool

	// systemPool indicates whether this is a special pool derived from the
	// system roots. If it includes additional roots, it requires doing two
	// verifications, one using the roots provided by the caller, and one using
	// the system platform verifier.
	systemPool bool
}

// lazyCert is minimal metadata about a Cert and a func to retrieve it
// in its normal expanded *Certificate form.
type lazyCert struct {
	// rawSubject is the Certificate.RawSubject value.
	// It's the same as the CertPool.byName key, but in []byte
	// form to make CertPool.Subjects (as used by crypto/tls) do
	// fewer allocations.
	rawSubject []byte

	// constraint is a function to run against a chain when it is a candidate to
	// be added to the chain. This allows adding arbitrary constraints that are
	// not specified in the certificate itself.
	constraint func([]*Certificate) error

	// getCert returns the certificate.
	//
	// It is not meant to do network operations or anything else
	// where a failure is likely; the func is meant to lazily
	// parse/decompress data that is already known to be good. The
	// error in the signature primarily is meant for use in the
	// case where a cert file existed on local disk when the program
	// started up is deleted later before it's read.
	getCert func() (*Certificate, error)
}

// NewCertPool returns a new, empty CertPool.
func NewCertPool() *CertPool {
	return &CertPool{
		byName:  make(map[string][]int),
		haveSum: make(map[sum224]bool),
	}
}

// len returns the number of certs in the set.
// A nil set is a valid empty set.
func (s *CertPool) len() int {
	if s == nil {
		return 0
	}
	return len(s.lazyCerts)
}

// cert returns cert index n in s.
func (s *CertPool) cert(n int) (*Certificate, func([]*Certificate) error, error) {
	cert, err := s.lazyCerts[n].getCert()
	return cert, s.lazyCerts[n].constraint, err
}

// Clone returns a copy of s.
func (s *CertPool) Clone() *CertPool {
	p := &CertPool{
		byName:     make(map[string][]int, len(s.byName)),
		lazyCerts:  make([]lazyCert, len(s.lazyCerts)),
		haveSum:    make(map[sum224]bool, len(s.haveSum)),
		systemPool: s.systemPool,
	}
	for k, v := range s.byName {
		indexes := make([]int, len(v))
		copy(indexes, v)
		p.byName[k] = indexes
	}
	for k := range s.haveSum {
		p.haveSum[k] = true
	}
	copy(p.lazyCerts, s.lazyCerts)
	return p
}

// SystemCertPool returns a copy of the system cert pool.
//
// On Unix systems other than macOS the environment variables SSL_CERT_FILE and
// SSL_CERT_DIR can be used to override the system default locations for the SSL
// certificate file and SSL certificate files directory, respectively. The
// latter can be a colon-separated list.
//
// Any mutations to the returned pool are not written to disk and do not affect
// any other pool returned by SystemCertPool.
//
// New changes in the system cert pool might not be reflected in subsequent calls.
func SystemCertPool() (*CertPool, error) {
	if sysRoots := systemRootsPool(); sysRoots != nil {
		return sysRoots.Clone(), nil
	}

	return loadSystemRoots()
}

type potentialParent struct {
	cert       *Certificate
	constraint func([]*Certificate) error
}

// findPotentialParents returns the certificates in s which might have signed
// cert.
func (s *CertPool) findPotentialParents(cert *Certificate) []potentialParent {
	if s == nil {
		return nil
	}

	// consider all candidates where cert.Issuer matches cert.Subject.
	// when picking possible candidates the list is built in the order
	// of match plausibility as to save cycles in buildChains:
	//   AKID and SKID match
	//   AKID present, SKID missing / AKID missing, SKID present
	//   AKID and SKID don't match
	var matchingKeyID, oneKeyID, mismatchKeyID []potentialParent
	for _, c := range s.byName[string(cert.RawIssuer)] {
		candidate, constraint, err := s.cert(c)
		if err != nil {
			continue
		}
		kidMatch := bytes.Equal(candidate.SubjectKeyId, cert.AuthorityKeyId)
		switch {
		case kidMatch:
			matchingKeyID = append(matchingKeyID, potentialParent{candidate, constraint})
		case (len(candidate.SubjectKeyId) == 0 && len(cert.AuthorityKeyId) > 0) ||
			(len(candidate.SubjectKeyId) > 0 && len(cert.AuthorityKeyId) == 0):
			oneKeyID = append(oneKeyID, potentialParent{candidate, constraint})
		default:
			mismatchKeyID = append(mismatchKeyID, potentialParent{candidate, constraint})
		}
	}

	found := len(matchingKeyID) + len(oneKeyID) + len(mismatchKeyID)
	if found == 0 {
		return nil
	}
	candidates := make([]potentialParent, 0, found)
	candidates = append(candidates, matchingKeyID...)
	candidates = append(candidates, oneKeyID...)
	candidates = append(candidates, mismatchKeyID...)
	return candidates
}

func (s *CertPool) contains(cert *Certificate) bool {
	if s == nil {
		return false
	}
	return s.haveSum[sha256.Sum224(cert.Raw)]
}

// AddCert adds a certificate to a pool.
func (s *CertPool) AddCert(cert *Certificate) {
	if cert == nil {
		panic("adding nil Certificate to CertPool")
	}
	s.addCertFunc(sha256.Sum224(cert.Raw), string(cert.RawSubject), func() (*Certificate, error) {
		return cert, nil
	}, nil)
}

// addCertFunc adds metadata about a certificate to a pool, along with
// a func to fetch that certificate later when needed.
//
// The rawSubject is Certificate.RawSubject and must be non-empty.
// The getCert func may be called 0 or more times.
func (s *CertPool) addCertFunc(rawSum224 sum224, rawSubject string, getCert func() (*Certificate, error), constraint func([]*Certificate) error) {
	if getCert == nil {
		panic("getCert can't be nil")
	}

	// Check that the certificate isn't being added twice.
	if s.haveSum[rawSum224] {
		return
	}

	s.haveSum[rawSum224] = true
	s.lazyCerts = append(s.lazyCerts, lazyCert{
		rawSubject: []byte(rawSubject),
		getCert:    getCert,
		constraint: constraint,
	})
	s.byName[rawSubject] = append(s.byName[rawSubject], len(s.lazyCerts)-1)
}

// AppendCertsFromFile attempts to parse a series of File certificates.
// It appends any certificates found to s and reports whether any certificates
// were successfully parsed.
//
// On many Linux systems, /etc/ssl/cert.pem will contain the system wide set
// of root CAs in a format suitable for this function.
func (s *CertPool) AppendCertsFromFile(path string) (bool, error) {
	file, err := os.ReadFile(path)
	if err != nil {
		return false, err
	}

	return s.AppendCertsFromPEM(file), nil
}

// AppendCertsFromPEM attempts to parse a series of PEM encoded certificates.
// It appends any certificates found to s and reports whether any certificates
// were successfully parsed.
//
// On many Linux systems, /etc/ssl/cert.pem will contain the system wide set
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

		certBytes := block.Bytes
		cert, err := ParseCertificate(certBytes)
		if err != nil {
			continue
		}
		var lazyCert struct {
			sync.Once
			v *Certificate
		}
		s.addCertFunc(sha256.Sum224(cert.Raw), string(cert.RawSubject), func() (*Certificate, error) {
			lazyCert.Do(func() {
				// This can't fail, as the same bytes already parsed above.
				lazyCert.v, _ = ParseCertificate(certBytes)
				certBytes = nil
			})
			return lazyCert.v, nil
		}, nil)
		ok = true
	}

	return ok
}

// Subjects returns a list of the DER-encoded subjects of
// all of the certificates in the pool.
//
// Deprecated: if s was returned by [SystemCertPool], Subjects
// will not include the system roots.
func (s *CertPool) Subjects() [][]byte {
	res := make([][]byte, s.len())
	for i, lc := range s.lazyCerts {
		res[i] = lc.rawSubject
	}
	return res
}

// Equal reports whether s and other are equal.
func (s *CertPool) Equal(other *CertPool) bool {
	if s == nil || other == nil {
		return s == other
	}
	if s.systemPool != other.systemPool || len(s.haveSum) != len(other.haveSum) {
		return false
	}
	for h := range s.haveSum {
		if !other.haveSum[h] {
			return false
		}
	}
	return true
}

// AddCertWithConstraint adds a certificate to the pool with the additional
// constraint. When Certificate.Verify builds a chain which is rooted by cert,
// it will additionally pass the whole chain to constraint to determine its
// validity. If constraint returns a non-nil error, the chain will be discarded.
// constraint may be called concurrently from multiple goroutines.
func (s *CertPool) AddCertWithConstraint(cert *Certificate, constraint func([]*Certificate) error) {
	if cert == nil {
		panic("adding nil Certificate to CertPool")
	}
	s.addCertFunc(sha256.Sum224(cert.Raw), string(cert.RawSubject), func() (*Certificate, error) {
		return cert, nil
	}, constraint)
}
