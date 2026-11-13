// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import (
	"crypto/sha256"
	"crypto/x509"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"golang.org/x/crypto/cryptobyte"
	"golang.org/x/crypto/cryptobyte/asn1"
)

// Result is the aggregate result of checking a leaf certificate across all
// of its verified chains.
type Result uint8

const (
	// NotCovered means the cache could not determine the certificate's status.
	// Reasons include: the leaf has no embedded SCTs, no filter covers any of
	// the SCT (log ID, timestamp) pairs, the covering filters' own coverage
	// disagrees with the index, or covering filters were found but every one
	// reported the issuer as not enrolled.
	NotCovered Result = iota

	// Revoked means a covering filter listed the certificate as revoked.
	Revoked

	// NotRevoked means at least one covering filter answered conclusively that
	// the certificate is not in its revoked set, and no covering filter listed
	// it as revoked.
	NotRevoked
)

var (
	errMalformedSCTExtension   = errors.New("upki: malformed SCT extension")
	errMalformedTBSCertificate = errors.New("upki: malformed TBSCertificate")
)

// Check checks the end-entity certificate in chains against the C2SP
// revocation cache in dir. The chains must already have been verified. Check
// takes ownership of indexFile and closes it before returning.
func Check(dir string, indexFile *os.File, chains [][]*x509.Certificate) (Result, error) {
	defer func() { _ = indexFile.Close() }()

	if len(chains) == 0 || len(chains[0]) == 0 {
		return NotCovered, errors.New("upki: no verified certificate chains")
	}

	leaf := chains[0][0]
	scts, err := ExtractSCTs(leaf)
	if err != nil {
		return NotCovered, fmt.Errorf("upki: extracting SCTs: %w", err)
	}
	if len(scts) == 0 {
		return NotCovered, nil
	}

	serial, err := rawSerial(leaf)
	if err != nil {
		return NotCovered, fmt.Errorf("upki: extracting serial: %w", err)
	}

	for attempt := 0; ; attempt++ {
		idx, err := newIndex(indexFile)
		if err != nil {
			return NotCovered, err
		}

		c := checker{
			idx:     idx,
			dir:     dir,
			chains:  chains,
			serial:  serial,
			scts:    scts,
			filters: make(map[string]*revocationFilter),
		}
		result, err := c.checkChains()
		if err == nil {
			return result, nil
		}
		if !errors.Is(err, fs.ErrNotExist) || attempt == maxIndexReopens {
			return NotCovered, err
		}

		// A filter named by the open index disappeared. The cache update
		// contract means the index was superseded, so reopen it and retry.
		_ = indexFile.Close()
		nextIndex, openErr := OpenIndex(filepath.Join(dir, IndexFilename))
		if openErr != nil {
			return NotCovered, fmt.Errorf("upki: reopening superseded index: %w", openErr)
		}
		indexFile = nextIndex
	}
}

// maxIndexReopens bounds how many times a single [Check] call reopens a
// superseded index before surfacing the missing-filter error, per the spec's
// allowance to bound retries under pathological cache update patterns.
const maxIndexReopens = 2

// checker holds the state shared by the chain and filter checks in one pass.
type checker struct {
	idx     *index
	dir     string
	chains  [][]*x509.Certificate
	serial  []byte
	scts    []SCT
	filters map[string]*revocationFilter
}

func (c *checker) checkChains() (Result, error) {
	// Every verified chain's second certificate verified the leaf's
	// signature, so the chains share one issuer public key and, unless two
	// intermediates encode that key with different SPKI bytes, one lookup
	// key. Query each distinct key once rather than once per chain.
	notRevoked := false
	seen := make(map[issuerSPKIHash]bool, 1)
	for _, chain := range c.chains {
		if len(chain) < 2 {
			continue
		}

		issuerHash := sha256.Sum256(chain[1].RawSubjectPublicKeyInfo)
		if seen[issuerHash] {
			continue
		}
		seen[issuerHash] = true

		key := newLookupKey(issuerHash, c.serial)
		result, err := c.checkChain(&key)
		if err != nil {
			return NotCovered, err
		}
		switch result {
		case Revoked:
			// Every verified chain describes the same leaf. Finding its
			// issuer-and-serial pair revoked on any valid path is conclusive.
			// Another path cannot make the same leaf acceptable.
			return Revoked, nil
		case NotRevoked:
			notRevoked = true
		}
	}
	if notRevoked {
		return NotRevoked, nil
	}

	return NotCovered, nil
}

func (c *checker) checkChain(key *lookupKey) (Result, error) {
	notRevoked := false
	seen := make(map[string]bool)
	for _, sct := range c.scts {
		filenames, err := c.idx.lookup(sct.logID, sct.timestamp)
		if err != nil {
			return NotCovered, err
		}
		for _, filename := range filenames {
			if seen[filename] {
				continue
			}
			seen[filename] = true

			status, err := c.queryFilter(filename, key)
			if err != nil {
				return NotCovered, err
			}

			switch status {
			case statusRevoked:
				return Revoked, nil
			case statusGood:
				// Conclusive, but a later filter may still revoke. Remember
				// the result but keep looking.
				notRevoked = true
			case statusNotCovered, statusNotEnrolled:
				// The filter couldn't answer: either its own coverage
				// disagrees with the index about this SCT, or it has no
				// block for the issuer. Keep looking. The spec says an
				// implementation SHOULD treat the former as an error, but
				// the reference implementation does not, and failing
				// closed on a producer bug that other consumers tolerate
				// would single out Go clients. If no filter answers the
				// result is NotCovered, which the caller's policy decides.
			}
		}
	}
	if notRevoked {
		return NotRevoked, nil
	}

	return NotCovered, nil
}

// queryFilter loads, parses, and queries filename, reusing a parsed filter
// when another verified chain selected the same file.
func (c *checker) queryFilter(filename string, key *lookupKey) (filterStatus, error) {
	filter, ok := c.filters[filename]
	if !ok {
		data, err := readFilterFile(filepath.Join(c.dir, filename))
		if err != nil {
			return statusNotCovered, fmt.Errorf("upki: reading filter %s: %w", filename, err)
		}
		filter, err = parseFilter(data)
		if err != nil {
			return statusNotCovered, fmt.Errorf("upki: parsing filter %s: %w", filename, err)
		}
		c.filters[filename] = filter
	}

	return filter.contains(key, c.scts), nil
}

// ExtractSCTs decodes the RFC 6962 SignedCertificateTimestampList
// extension from cert and returns one [SCT] per entry.
//
// Returns (nil, nil) when the extension is absent.
//
// Entries using unknown SCT versions are ignored.
//
// It returns errMalformedSCTExtension if the extension is present but malformed.
//
// See RFC 6962 §3.3 for more information.
func ExtractSCTs(cert *x509.Certificate) ([]SCT, error) {
	var raw []byte
	for _, ext := range cert.Extensions {
		if ext.Id.Equal(embeddedSCTOID) {
			raw = ext.Value
			break
		}
	}
	if raw == nil {
		return nil, nil
	}

	// The contents of the ASN.1 OCTET STRING in an X.509 certificate
	// extension are:
	//
	//	opaque SerializedSCT<1..2^16-1>;
	//	struct {
	//	    SerializedSCT sct_list<1..2^16-1>;
	//	} SignedCertificateTimestampList;

	outer := cryptobyte.String(raw)
	var inner cryptobyte.String
	if !outer.ReadASN1(&inner, asn1.OCTET_STRING) || !outer.Empty() {
		return nil, fmt.Errorf("%w: outer wrapper", errMalformedSCTExtension)
	}

	var listBytes cryptobyte.String
	if !inner.ReadUint16LengthPrefixed(&listBytes) || !inner.Empty() {
		return nil, fmt.Errorf("%w: sct list", errMalformedSCTExtension)
	}

	var out []SCT
	for !listBytes.Empty() {
		var serialized cryptobyte.String
		if !listBytes.ReadUint16LengthPrefixed(&serialized) {
			return nil, fmt.Errorf("%w: serialized sct", errMalformedSCTExtension)
		}

		//	struct {
		//	    Version sct_version;          // uint8, v1 == 0
		//	    LogID id;                     // 32 bytes
		//	    uint64 timestamp;
		//	    CtExtensions extensions;      // opaque<0..2^16-1>
		//	    DigitallySigned signature;
		//	} SignedCertificateTimestamp;

		var version uint8
		if !serialized.ReadUint8(&version) {
			return nil, fmt.Errorf("%w: sct version", errMalformedSCTExtension)
		}
		if version != 0 {
			continue
		}

		var s SCT
		if !serialized.CopyBytes(s.logID[:]) {
			return nil, fmt.Errorf("%w: sct log id", errMalformedSCTExtension)
		}

		var ts uint64
		if !serialized.ReadUint64(&ts) {
			return nil, fmt.Errorf("%w: sct timestamp", errMalformedSCTExtension)
		}
		s.timestamp = timestamp(ts)

		// Validate (but discard) the CtExtensions length prefix to
		// catch truncated SCTs. The trailing signature is not parsed:
		// the outer length-prefix already delimits this SerializedSCT.
		var ext cryptobyte.String
		if !serialized.ReadUint16LengthPrefixed(&ext) {
			return nil, fmt.Errorf("%w: sct extensions", errMalformedSCTExtension)
		}

		out = append(out, s)
	}

	return out, nil
}

// SCT is a parsed Signed Certificate Timestamp from a leaf certificate's
// embedded SCT extension.
//
// It carries the two fields CRLite needs to pick covering filters: the
// CT log's identifier and the log's reported issuance timestamp.
//
// Important: The SCT's signature is intentionally not retained.
// In the revocation checking context we assume the CT log inclusion
// proof was already validated.
type SCT struct {
	logID     logID
	timestamp timestamp
}

// logID is a 32-byte CT log identifier (SHA-256 of the log's public key).
type logID [32]byte

// timestamp is a millisecond-since-epoch CT timestamp.
type timestamp uint64

// RFC 6962 §3.3
var embeddedSCTOID = []int{1, 3, 6, 1, 4, 1, 11129, 2, 4, 2}

// rawSerial returns the certificate's serial number as the DER-encoded
// contents of the serialNumber INTEGER, preserving any leading sign
// byte.
//
// In contrast, [x509.Certificate.SerialNumber] is a *big.Int which discards
// the leading sign byte. CRLite hashes the original DER bytes as a lookup
// input, so we parse the raw serial DER ourselves here.
func rawSerial(cert *x509.Certificate) ([]byte, error) {
	//	TBSCertificate ::= SEQUENCE {
	//	    version [0] EXPLICIT INTEGER DEFAULT v1,
	//	    serialNumber CertificateSerialNumber,
	//	    ...
	//	}
	tbs := cryptobyte.String(cert.RawTBSCertificate)
	var inner cryptobyte.String
	if !tbs.ReadASN1(&inner, asn1.SEQUENCE) {
		return nil, fmt.Errorf("%w: outer sequence", errMalformedTBSCertificate)
	}

	// Skip optional [0] EXPLICIT version, if present.
	if inner.PeekASN1Tag(asn1.Tag(0).Constructed().ContextSpecific()) {
		var version cryptobyte.String
		if !inner.ReadASN1(&version, asn1.Tag(0).Constructed().ContextSpecific()) {
			return nil, fmt.Errorf("%w: version", errMalformedTBSCertificate)
		}
	}

	var serial cryptobyte.String
	if !inner.ReadASN1(&serial, asn1.INTEGER) {
		return nil, fmt.Errorf("%w: serial", errMalformedTBSCertificate)
	}

	return serial, nil
}
