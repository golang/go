// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/binary"
	"errors"
	"math/big"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"golang.org/x/crypto/cryptobyte"
)

const (
	checkTestLeafSerial         int64     = 42
	checkTestIssuerSerial       int64     = 100
	checkTestOtherIssuerSerial  int64     = 101
	checkTestSCTTime            timestamp = 1000
	checkTestCoverageLow        timestamp = 0
	checkTestCoverageHigh       timestamp = 2000
	checkTestNarrowCoverageLow  timestamp = 500
	checkTestNarrowCoverageHigh timestamp = 1500
)

// TestCheck exercises the complete lookup path from certificate inputs through
// index selection, filter parsing, and the final aggregate result.
func TestCheck(t *testing.T) {
	t.Parallel()

	coveredLog := filledLogID(0x11)
	issuer := newTestCertificate(t, big.NewInt(checkTestIssuerSerial))
	otherIssuer := newTestCertificate(t, big.NewInt(checkTestOtherIssuerSerial))
	issuerHash := sha256.Sum256(issuer.RawSubjectPublicKeyInfo)
	otherHash := sha256.Sum256(otherIssuer.RawSubjectPublicKeyInfo)
	coverage := []testCoverage{{
		logID: coveredLog, low: checkTestNarrowCoverageLow, high: checkTestNarrowCoverageHigh,
	}}
	index := encodeTestIndex([]testIndexFilter{{
		filename: "test.filter",
		coverage: []testIndexCoverage{{
			logID: coveredLog, low: checkTestNarrowCoverageLow, high: checkTestNarrowCoverageHigh,
		}},
	}})
	sct := testSCTExtension(t, serializedSCT(SCT{logID: coveredLog, timestamp: checkTestSCTTime}))
	uncoveredSCT := testSCTExtension(t, serializedSCT(SCT{logID: filledLogID(0x77), timestamp: checkTestSCTTime}))
	malformedSCT := pkix.Extension{Id: embeddedSCTOID, Value: []byte{0x04, 0x05, 0}}

	// Each case ends Check at a different point: a conclusive answer from a
	// filter (revoked or not revoked), an inconclusive result before any
	// filter is consulted (no covering filter, the issuer not enrolled, or no
	// SCTs at all), or an error from a malformed SCT extension.
	tests := []struct {
		name    string
		leaf    *x509.Certificate
		issuer  *x509.Certificate
		index   []byte
		filter  []byte
		want    Result
		wantErr bool
	}{
		{
			name:   "revoked",
			leaf:   newTestCertificateWithExtension(t, big.NewInt(checkTestLeafSerial), &sct),
			issuer: issuer,
			index:  index,
			filter: encodeTestFilter(coverage, []testBlock{{issuer: issuerHash, inverted: 1}}, 0),
			want:   Revoked,
		},
		{
			name:   "not revoked",
			leaf:   newTestCertificateWithExtension(t, big.NewInt(checkTestLeafSerial), &sct),
			issuer: issuer,
			index:  index,
			filter: encodeTestFilter(coverage, []testBlock{{issuer: issuerHash}}, 0),
			want:   NotRevoked,
		},
		{
			name:   "SCT not covered",
			leaf:   newTestCertificateWithExtension(t, big.NewInt(checkTestLeafSerial), &uncoveredSCT),
			issuer: issuer,
			index:  index,
			filter: encodeTestFilter(coverage, []testBlock{{issuer: issuerHash, inverted: 1}}, 0),
			want:   NotCovered,
		},
		{
			name:   "issuer not enrolled",
			leaf:   newTestCertificateWithExtension(t, big.NewInt(checkTestLeafSerial), &sct),
			issuer: issuer,
			index:  index,
			filter: encodeTestFilter(coverage, []testBlock{{issuer: otherHash, inverted: 1}}, 0),
			want:   NotCovered,
		},
		{
			name:   "no SCTs",
			leaf:   newTestCertificate(t, big.NewInt(checkTestLeafSerial)),
			issuer: issuer,
			index:  []byte("not read"),
			want:   NotCovered,
		},
		{
			name:    "malformed SCTs",
			leaf:    newTestCertificateWithExtension(t, big.NewInt(checkTestLeafSerial), &malformedSCT),
			issuer:  issuer,
			index:   index,
			filter:  encodeTestFilter(coverage, []testBlock{{issuer: issuerHash}}, 0),
			want:    NotCovered,
			wantErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			filters := map[string][]byte{}
			if test.filter != nil {
				filters["test.filter"] = test.filter
			}
			dir, indexFile := writeTestCache(t, test.index, filters)
			got, err := Check(dir, indexFile, [][]*x509.Certificate{{test.leaf, test.issuer}})
			if (err != nil) != test.wantErr {
				t.Fatalf("Check error = %v, wantErr %v", err, test.wantErr)
			}
			if got != test.want {
				t.Errorf("Check = %v, want %v", got, test.want)
			}
		})
	}
}

// TestCheckAggregatesFilters verifies that inconclusive and good answers do
// not hide a later revocation, while a revocation stops further file reads.
func TestCheckAggregatesFilters(t *testing.T) {
	t.Parallel()

	logA, logB := filledLogID(0x11), filledLogID(0x22)
	issuer := newTestCertificate(t, big.NewInt(checkTestIssuerSerial))
	otherIssuer := newTestCertificate(t, big.NewInt(checkTestOtherIssuerSerial))
	issuerHash := sha256.Sum256(issuer.RawSubjectPublicKeyInfo)
	otherHash := sha256.Sum256(otherIssuer.RawSubjectPublicKeyInfo)
	coverageA := testCoverage{logID: logA, low: checkTestCoverageLow, high: checkTestCoverageHigh}
	coverageB := testCoverage{logID: logB, low: checkTestCoverageLow, high: checkTestCoverageHigh}
	indexCoverageA := testIndexCoverage{logID: logA, low: checkTestCoverageLow, high: checkTestCoverageHigh}
	indexCoverageB := testIndexCoverage{logID: logB, low: checkTestCoverageLow, high: checkTestCoverageHigh}
	scts := testSCTExtension(t,
		serializedSCT(SCT{logID: logA, timestamp: checkTestSCTTime}),
		serializedSCT(SCT{logID: logB, timestamp: checkTestSCTTime}))
	leaf := newTestCertificateWithExtension(t, big.NewInt(checkTestLeafSerial), &scts)

	filter := func(coverage testCoverage, issuer issuerSPKIHash, inverted uint8) []byte {

		return encodeTestFilter([]testCoverage{coverage}, []testBlock{{issuer: issuer, inverted: inverted}}, 0)
	}
	tests := []struct {
		name    string
		index   []testIndexFilter
		filters map[string][]byte
		want    Result
	}{
		{
			name: "continues past not enrolled",
			index: []testIndexFilter{
				{filename: "first.filter", coverage: []testIndexCoverage{indexCoverageA}},
				{filename: "second.filter", coverage: []testIndexCoverage{indexCoverageB}},
			},
			filters: map[string][]byte{
				"first.filter":  filter(coverageA, otherHash, 1),
				"second.filter": filter(coverageB, issuerHash, 1),
			},
			want: Revoked,
		},
		{
			name: "revocation wins after good result",
			index: []testIndexFilter{
				{filename: "first.filter", coverage: []testIndexCoverage{indexCoverageA}},
				{filename: "second.filter", coverage: []testIndexCoverage{indexCoverageB}},
			},
			filters: map[string][]byte{
				"first.filter":  filter(coverageA, issuerHash, 0),
				"second.filter": filter(coverageB, issuerHash, 1),
			},
			want: Revoked,
		},
		{
			name: "revocation short circuits",
			index: []testIndexFilter{
				{filename: "first.filter", coverage: []testIndexCoverage{indexCoverageA}},
				{filename: "missing.filter", coverage: []testIndexCoverage{indexCoverageB}},
			},
			filters: map[string][]byte{"first.filter": filter(coverageA, issuerHash, 1)},
			want:    Revoked,
		},
		{
			name: "all filters not enrolled",
			index: []testIndexFilter{
				{filename: "first.filter", coverage: []testIndexCoverage{indexCoverageA}},
				{filename: "second.filter", coverage: []testIndexCoverage{indexCoverageB}},
			},
			filters: map[string][]byte{
				"first.filter":  filter(coverageA, otherHash, 1),
				"second.filter": filter(coverageB, otherHash, 1),
			},
			want: NotCovered,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			dir, indexFile := writeTestCache(t, encodeTestIndex(test.index), test.filters)
			got, err := Check(dir, indexFile, [][]*x509.Certificate{{leaf, issuer}})
			if err != nil {
				t.Fatalf("Check: %v", err)
			}
			if got != test.want {
				t.Errorf("Check = %v, want %v", got, test.want)
			}
		})
	}
}

// TestCheckInconsistentIndex covers the integrity rule for disagreement
// between index coverage and a selected filter's own coverage metadata.
// TestCheckInconsistentIndex covers filters whose own coverage disagrees with
// the index entry that selected them. Such a filter cannot answer, so the
// result is NotCovered rather than an error, and another selected filter can
// still answer.
func TestCheckInconsistentIndex(t *testing.T) {
	t.Parallel()

	log := filledLogID(0x11)
	issuer := newTestCertificate(t, big.NewInt(checkTestIssuerSerial))
	otherIssuer := newTestCertificate(t, big.NewInt(checkTestOtherIssuerSerial))
	issuerHash := sha256.Sum256(issuer.RawSubjectPublicKeyInfo)
	otherHash := sha256.Sum256(otherIssuer.RawSubjectPublicKeyInfo)
	sct := testSCTExtension(t, serializedSCT(SCT{logID: log, timestamp: checkTestSCTTime}))
	leaf := newTestCertificateWithExtension(t, big.NewInt(checkTestLeafSerial), &sct)
	indexCoverage := testIndexCoverage{logID: log, low: checkTestCoverageLow, high: checkTestCoverageHigh}
	disagreeingCoverage := testCoverage{logID: filledLogID(0x22), low: checkTestCoverageLow, high: checkTestCoverageHigh}
	covered := testCoverage{logID: log, low: checkTestCoverageLow, high: checkTestCoverageHigh}

	tests := []struct {
		name    string
		index   []testIndexFilter
		filters map[string][]byte
	}{
		{
			name:  "every selected filter disagrees",
			index: []testIndexFilter{{filename: "bad.filter", coverage: []testIndexCoverage{indexCoverage}}},
			filters: map[string][]byte{"bad.filter": encodeTestFilter(
				[]testCoverage{disagreeingCoverage}, []testBlock{{issuer: issuerHash}}, 0)},
		},
		{
			name: "not enrolled filter confirms coverage",
			index: []testIndexFilter{
				{filename: "bad.filter", coverage: []testIndexCoverage{indexCoverage}},
				{filename: "covered.filter", coverage: []testIndexCoverage{indexCoverage}},
			},
			filters: map[string][]byte{
				"bad.filter": encodeTestFilter(
					[]testCoverage{disagreeingCoverage}, []testBlock{{issuer: issuerHash}}, 0),
				"covered.filter": encodeTestFilter(
					[]testCoverage{covered}, []testBlock{{issuer: otherHash}}, 0),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			dir, indexFile := writeTestCache(t, encodeTestIndex(test.index), test.filters)
			got, err := Check(dir, indexFile, [][]*x509.Certificate{{leaf, issuer}})
			if err != nil || got != NotCovered {
				t.Fatalf("Check = %v, %v, want NotCovered, nil", got, err)
			}
		})
	}
}

// TestCheckAggregatesChains verifies that every distinct issuer key among the
// verified paths is considered: revocation on any path wins, while one
// conclusive good path is sufficient when the remaining paths cannot answer.
//
// Chains returned by x509 verification all have an issuer that verified the
// leaf's signature, so they share one issuer key. Distinct lookup keys arise
// only when intermediates encode that key with different SPKI bytes. Two
// issuer certificates with different keys stand in for that case here.
func TestCheckAggregatesChains(t *testing.T) {
	t.Parallel()

	log := filledLogID(0x11)
	issuerA := newTestCertificate(t, big.NewInt(checkTestIssuerSerial))
	issuerB := newTestCertificate(t, big.NewInt(checkTestOtherIssuerSerial))
	hashA := sha256.Sum256(issuerA.RawSubjectPublicKeyInfo)
	hashB := sha256.Sum256(issuerB.RawSubjectPublicKeyInfo)
	coverage := testCoverage{logID: log, low: checkTestCoverageLow, high: checkTestCoverageHigh}
	indexCoverage := testIndexCoverage{logID: log, low: checkTestCoverageLow, high: checkTestCoverageHigh}
	sct := testSCTExtension(t, serializedSCT(SCT{logID: log, timestamp: checkTestSCTTime}))
	leaf := newTestCertificateWithExtension(t, big.NewInt(checkTestLeafSerial), &sct)

	tests := []struct {
		name    string
		filters map[string][]byte
		chains  [][]*x509.Certificate
		want    Result
	}{
		{
			name: "revocation on alternate chain wins",
			filters: map[string][]byte{
				"good.filter":    encodeTestFilter([]testCoverage{coverage}, []testBlock{{issuer: hashA}}, 0),
				"revoked.filter": encodeTestFilter([]testCoverage{coverage}, []testBlock{{issuer: hashB, inverted: 1}}, 0),
			},
			chains: [][]*x509.Certificate{{leaf, issuerA}, {leaf, issuerB}},
			want:   Revoked,
		},
		{
			name: "good chain and uncovered alternate",
			filters: map[string][]byte{
				"good.filter":    encodeTestFilter([]testCoverage{coverage}, []testBlock{{issuer: hashA}}, 0),
				"revoked.filter": encodeTestFilter([]testCoverage{coverage}, []testBlock{{issuer: issuerSPKIHash{0xff}, inverted: 1}}, 0),
			},
			chains: [][]*x509.Certificate{{leaf, issuerA}, {leaf, issuerB}, {leaf}},
			want:   NotRevoked,
		},
	}
	index := encodeTestIndex([]testIndexFilter{
		{filename: "good.filter", coverage: []testIndexCoverage{indexCoverage}},
		{filename: "revoked.filter", coverage: []testIndexCoverage{indexCoverage}},
	})
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			dir, indexFile := writeTestCache(t, index, test.filters)
			got, err := Check(dir, indexFile, test.chains)
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Errorf("Check = %v, want %v", got, test.want)
			}
		})
	}
}

// TestCheckReopensIndex simulates the cache publication protocol around a
// held index descriptor and verifies successful, bounded, and missing-data
// retry outcomes.
func TestCheckReopensIndex(t *testing.T) {
	t.Parallel()

	log := filledLogID(0x11)
	issuer := newTestCertificate(t, big.NewInt(checkTestIssuerSerial))
	issuerHash := sha256.Sum256(issuer.RawSubjectPublicKeyInfo)
	coverage := testCoverage{logID: log, low: checkTestCoverageLow, high: checkTestCoverageHigh}
	indexCoverage := testIndexCoverage{logID: log, low: checkTestCoverageLow, high: checkTestCoverageHigh}
	sct := testSCTExtension(t, serializedSCT(SCT{logID: log, timestamp: checkTestSCTTime}))
	leaf := newTestCertificateWithExtension(t, big.NewInt(checkTestLeafSerial), &sct)
	chain := [][]*x509.Certificate{{leaf, issuer}}
	indexNaming := func(filename string) []byte {

		return encodeTestIndex([]testIndexFilter{{filename: filename, coverage: []testIndexCoverage{indexCoverage}}})
	}

	t.Run("superseded index", func(t *testing.T) {
		t.Parallel()
		dir, oldIndex := writeTestCache(t, indexNaming("old.filter"), nil)
		if err := os.WriteFile(filepath.Join(dir, "new.filter"), encodeTestFilter(
			[]testCoverage{coverage}, []testBlock{{issuer: issuerHash, inverted: 1}}, 0), 0o644); err != nil {
			t.Fatal(err)
		}
		replaceTestIndex(t, dir, indexNaming("new.filter"))
		got, err := Check(dir, oldIndex, chain)
		if err != nil || got != Revoked {
			t.Fatalf("Check = %v, %v, want Revoked, nil", got, err)
		}
	})

	t.Run("retries bounded", func(t *testing.T) {
		t.Parallel()
		// The index names a filter file that never appears, so every reopen
		// finds the same index and the same missing file. Check must give up
		// after maxIndexReopens and report the missing file instead of
		// retrying forever.
		dir, indexFile := writeTestCache(t, indexNaming("missing.filter"), nil)
		got, err := Check(dir, indexFile, chain)
		if !errors.Is(err, os.ErrNotExist) || got != NotCovered {
			t.Fatalf("Check = %v, %v, want NotCovered and os.ErrNotExist", got, err)
		}
	})

	t.Run("index removed before retry", func(t *testing.T) {
		t.Parallel()
		dir, indexFile := writeTestCache(t, indexNaming("missing.filter"), nil)
		if err := os.Remove(filepath.Join(dir, IndexFilename)); err != nil {
			t.Fatal(err)
		}
		got, err := Check(dir, indexFile, chain)
		if !errors.Is(err, ErrNoData) || got != NotCovered {
			t.Fatalf("Check = %v, %v, want NotCovered and ErrNoData", got, err)
		}
	})
}

func TestCheckErrors(t *testing.T) {
	t.Parallel()

	dir, indexFile := writeTestCache(t, nil, nil)
	if got, err := Check(dir, indexFile, nil); err == nil || got != NotCovered {
		t.Fatalf("Check(nil) = %v, %v, want NotCovered and an error", got, err)
	}

	indexFile, err := os.Open(filepath.Join(dir, IndexFilename))
	if err != nil {
		t.Fatal(err)
	}
	leaf := &x509.Certificate{
		Extensions:        []pkix.Extension{testSCTExtension(t, serializedSCT(SCT{}))},
		RawTBSCertificate: []byte{0xde, 0xad},
	}
	if got, err := Check(dir, indexFile, [][]*x509.Certificate{{leaf, newTestCertificate(t, big.NewInt(1))}}); !errors.Is(err, errMalformedTBSCertificate) || got != NotCovered {
		t.Fatalf("Check(malformed serial) = %v, %v", got, err)
	}
}

func TestCheckClosesIndex(t *testing.T) {
	t.Parallel()

	dir, indexFile := writeTestCache(t, []byte("not read"), nil)
	leaf := newTestCertificate(t, big.NewInt(1))
	if _, err := Check(dir, indexFile, [][]*x509.Certificate{{leaf, leaf}}); err != nil {
		t.Fatal(err)
	}
	if err := indexFile.Close(); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("second index Close error = %v, want os.ErrClosed", err)
	}
}

func TestExtractSCTs(t *testing.T) {
	t.Parallel()

	sctA := SCT{logID: filledLogID(0x11), timestamp: 1700000000000}
	sctB := SCT{logID: filledLogID(0x22), timestamp: 1700000123456}
	tests := []struct {
		name string
		cert *x509.Certificate
		want []SCT
	}{
		{
			name: "absent",
			cert: &x509.Certificate{},
		},
		{
			name: "multiple",
			cert: certificateWithSCTExtension(t, serializedSCT(sctA), serializedSCT(sctB)),
			want: []SCT{sctA, sctB},
		},
		{
			// An SCT whose version byte is not 0 (v1) is skipped without
			// parsing its body, so everything after the version is arbitrary.
			name: "unknown version ignored",
			cert: certificateWithSCTExtension(t,
				serializedSCT(sctA),
				append([]byte{1}, []byte("arbitrary")...),
				serializedSCT(sctB)),
			want: []SCT{sctA, sctB},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			got, err := ExtractSCTs(test.cert)
			if err != nil {
				t.Fatalf("ExtractSCTs: %v", err)
			}
			if !slices.Equal(got, test.want) {
				t.Fatalf("ExtractSCTs = %v, want %v", got, test.want)
			}
		})
	}
}

func TestExtractSCTsMalformed(t *testing.T) {
	t.Parallel()

	valid := sctExtensionValue(t, serializedSCT(SCT{}))
	// The extension value nests four length-delimited layers: an ASN.1
	// OCTET STRING, a uint16-prefixed SCT list, a uint16-prefixed
	// serialized SCT, and inside it the fixed fields (version, 32-byte log
	// ID, 8-byte timestamp) followed by uint16-prefixed CT extensions. Each
	// case truncates or pads one layer.
	tests := []struct {
		name  string
		value []byte
	}{
		{
			// OCTET STRING tag 0x04 claiming 5 content bytes, with one.
			name:  "outer wrapper truncated",
			value: []byte{0x04, 0x05, 0x00},
		},
		{
			// A byte after the end of the OCTET STRING.
			name:  "outer wrapper trailing data",
			value: append(bytes.Clone(valid), 0),
		},
		{
			// SCT list length 5 (0x0005), with one byte following.
			name:  "SCT list truncated",
			value: asn1OctetString(t, []byte{0, 5, 0}),
		},
		{
			// SCT list length 3 holding a serialized SCT length of 5
			// (0x0005) with one byte following.
			name:  "serialized SCT length truncated",
			value: asn1OctetString(t, []byte{0, 3, 0, 5, 0}),
		},
		{
			// A zero-length serialized SCT, so no version byte.
			name:  "version missing",
			value: sctExtensionValue(t, nil),
		},
		{
			// Version 0 (v1) followed by one byte of the 32-byte log ID.
			name:  "log ID truncated",
			value: sctExtensionValue(t, []byte{0, 1}),
		},
		{
			// Version and log ID, with no timestamp.
			name:  "timestamp truncated",
			value: sctExtensionValue(t, append([]byte{0}, make([]byte, 32)...)),
		},
		{
			// Version, log ID and timestamp, with no CT extensions length.
			name:  "extensions truncated",
			value: sctExtensionValue(t, append(append([]byte{0}, make([]byte, 32)...), make([]byte, 8)...)),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			cert := &x509.Certificate{Extensions: []pkix.Extension{{
				Id:    embeddedSCTOID,
				Value: test.value,
			}}}
			if _, err := ExtractSCTs(cert); !errors.Is(err, errMalformedSCTExtension) {
				t.Fatalf("ExtractSCTs error = %v, want errMalformedSCTExtension", err)
			}
		})
	}
}

func TestRawSerial(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		serial *big.Int
		want   []byte
	}{
		{"ordinary", big.NewInt(42), []byte{42}},
		{
			// A positive serial whose top bit is set. DER encodes it with
			// a leading 0x00 so the INTEGER is not read as negative, and
			// rawSerial must keep that byte, unlike big.Int.Bytes.
			name:   "leading sign byte",
			serial: new(big.Int).SetBytes([]byte{0x80, 0xab, 0xcd, 0xef}),
			want:   []byte{0, 0x80, 0xab, 0xcd, 0xef},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			got, err := rawSerial(newTestCertificate(t, test.serial))
			if err != nil {
				t.Fatalf("rawSerial: %v", err)
			}
			if !bytes.Equal(got, test.want) {
				t.Fatalf("rawSerial = %x, want %x", got, test.want)
			}
		})
	}
}

func TestRawSerialMalformed(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		tbs  []byte
	}{
		{
			// Not a SEQUENCE (tag 0x30), so TBSCertificate cannot be entered.
			name: "outer sequence",
			tbs:  []byte{0xde, 0xad},
		},
		{
			// An empty SEQUENCE: no version and no serialNumber INTEGER.
			name: "serial",
			tbs:  []byte{0x30, 0x00},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			cert := &x509.Certificate{RawTBSCertificate: test.tbs}
			if _, err := rawSerial(cert); !errors.Is(err, errMalformedTBSCertificate) {
				t.Fatalf("rawSerial error = %v, want errMalformedTBSCertificate", err)
			}
		})
	}
}

func certificateWithSCTExtension(t *testing.T, scts ...[]byte) *x509.Certificate {
	t.Helper()

	return &x509.Certificate{Extensions: []pkix.Extension{testSCTExtension(t, scts...)}}
}

func testSCTExtension(t *testing.T, scts ...[]byte) pkix.Extension {
	t.Helper()

	return pkix.Extension{
		Id:    embeddedSCTOID,
		Value: sctExtensionValue(t, scts...),
	}
}

func sctExtensionValue(t *testing.T, scts ...[]byte) []byte {
	t.Helper()

	var list []byte
	for _, sct := range scts {
		list = binary.BigEndian.AppendUint16(list, uint16(len(sct)))
		list = append(list, sct...)
	}
	inner := binary.BigEndian.AppendUint16(nil, uint16(len(list)))
	inner = append(inner, list...)

	return asn1OctetString(t, inner)
}

func serializedSCT(sct SCT) []byte {
	serialized := []byte{0} // version v1
	serialized = append(serialized, sct.logID[:]...)
	serialized = binary.BigEndian.AppendUint64(serialized, uint64(sct.timestamp))
	// ExtractSCTs reads through the CT extensions and then discards the
	// remainder of the length-delimited SCT, so no signature is needed.
	serialized = append(serialized, 0, 0) // empty CtExtensions

	return serialized
}

func asn1OctetString(t *testing.T, contents []byte) []byte {
	t.Helper()

	var builder cryptobyte.Builder
	builder.AddASN1OctetString(contents)
	encoded, err := builder.Bytes()
	if err != nil {
		t.Fatal(err)
	}

	return encoded
}

func filledLogID(value byte) logID {
	var id logID
	for i := range id {
		id[i] = value
	}

	return id
}

func newTestCertificate(t *testing.T, serial *big.Int) *x509.Certificate {
	t.Helper()
	return newTestCertificateWithExtension(t, serial, nil)
}

func newTestCertificateWithExtension(t *testing.T, serial *big.Int, extension *pkix.Extension) *x509.Certificate {
	t.Helper()

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: "test"},
		NotBefore:    time.Unix(0, 0),
		NotAfter:     time.Unix(1<<31, 0),
	}
	if extension != nil {
		template.ExtraExtensions = []pkix.Extension{*extension}
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}

	return cert
}

func writeTestCache(t *testing.T, index []byte, filters map[string][]byte) (string, *os.File) {
	t.Helper()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, IndexFilename), index, 0o644); err != nil {
		t.Fatal(err)
	}
	for name, data := range filters {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	indexFile, err := OpenIndex(filepath.Join(dir, IndexFilename))
	if err != nil {
		t.Fatal(err)
	}

	return dir, indexFile
}

func replaceTestIndex(t *testing.T, dir string, data []byte) {
	t.Helper()

	const temporary = "next-index.bin"
	if err := os.WriteFile(filepath.Join(dir, temporary), data, 0o644); err != nil {
		t.Fatal(err)
	}
	// Root.Rename uses delete-sharing-compatible replacement semantics on
	// Windows, matching the cache publication protocol.
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	if err := root.Rename(temporary, IndexFilename); err != nil {
		t.Fatal(err)
	}
}
