// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLineSize = 256

// bitIsSet reports whether the bit at index is set. The bit index
// is in big endian order, so bit index 0 is the leftmost bit.
func bitIsSet(bits []uint64, index uint) bool {
	return bits[index/64]&((1<<63)>>(index%64)) != 0
}

// function is the function code for the named function.
type function uint8

const (
	// KM{,A,C,CTR} function codes
	aes128 function = 18 // AES-128
	aes192          = 19 // AES-192
	aes256          = 20 // AES-256

	// K{I,L}MD function codes
	sha1   = 1 // SHA-1
	sha256 = 2 // SHA-256
	sha512 = 3 // SHA-512

	// KLMD function codes
	ghash = 65 // GHASH
)

// queryResult contains the result of a Query function
// call. Bits are numbered in big endian order so the
// leftmost bit (the MSB) is at index 0.
type queryResult struct {
	bits [2]uint64
}

// Has reports whether the given functions are present.
func (q *queryResult) Has(fns ...function) bool {
	if len(fns) == 0 {
		panic("no function codes provided")
	}
	for _, f := range fns {
		if !bitIsSet(q.bits[:], uint(f)) {
			return false
		}
	}
	return true
}

// facility is a bit index for the named facility.
type facility uint8

const (
	// mandatory facilities
	zarch  facility = 1  // z architecture mode is active
	stflef          = 7  // store-facility-list-extended
	ldisp           = 18 // long-displacement
	eimm            = 21 // extended-immediate

	// miscellaneous facilities
	dfp    = 42 // decimal-floating-point
	etf3eh = 30 // extended-translation 3 enhancement

	// cryptography facilities
	msa  = 17  // message-security-assist
	msa3 = 76  // message-security-assist extension 3
	msa4 = 77  // message-security-assist extension 4
	msa5 = 57  // message-security-assist extension 5
	msa8 = 146 // message-security-assist extension 8

	// Note: vx and highgprs are excluded because they require
	// kernel support and so must be fetched from HWCAP.
)

// facilityList contains the result of an STFLE call.
// Bits are numbered in big endian order so the
// leftmost bit (the MSB) is at index 0.
type facilityList struct {
	bits [4]uint64
}

// Has reports whether the given facilities are present.
func (s *facilityList) Has(fs ...facility) bool {
	if len(fs) == 0 {
		panic("no facility bits provided")
	}
	for _, f := range fs {
		if !bitIsSet(s.bits[:], uint(f)) {
			return false
		}
	}
	return true
}

// The following feature detection functions are defined in cpu_s390x.s.
// They are likely to be expensive to call so the results should be cached.
func stfle() facilityList
func kmQuery() queryResult
func kmcQuery() queryResult
func kmctrQuery() queryResult
func kmaQuery() queryResult
func kimdQuery() queryResult
func klmdQuery() queryResult

func doinit() {
	options = []option{
		{"zarch", &S390X.HasZArch},
		{"stfle", &S390X.HasSTFLE},
		{"ldisp", &S390X.HasLDisp},
		{"msa", &S390X.HasMSA},
		{"eimm", &S390X.HasEImm},
		{"dfp", &S390X.HasDFP},
		{"etf3eh", &S390X.HasETF3Enhanced},
		{"vx", &S390X.HasVX},
	}

	aes := []function{aes128, aes192, aes256}
	facilities := stfle()

	S390X.HasZArch = facilities.Has(zarch)
	S390X.HasSTFLE = facilities.Has(stflef)
	S390X.HasLDisp = facilities.Has(ldisp)
	S390X.HasEImm = facilities.Has(eimm)
	S390X.HasDFP = facilities.Has(dfp)
	S390X.HasETF3Enhanced = facilities.Has(etf3eh)
	S390X.HasMSA = facilities.Has(msa)

	if S390X.HasMSA {
		// cipher message
		km, kmc := kmQuery(), kmcQuery()
		S390X.HasAES = km.Has(aes...)
		S390X.HasAESCBC = kmc.Has(aes...)
		if facilities.Has(msa4) {
			kmctr := kmctrQuery()
			S390X.HasAESCTR = kmctr.Has(aes...)
		}
		if facilities.Has(msa8) {
			kma := kmaQuery()
			S390X.HasAESGCM = kma.Has(aes...)
		}

		// compute message digest
		kimd := kimdQuery() // intermediate (no padding)
		klmd := klmdQuery() // last (padding)
		S390X.HasSHA1 = kimd.Has(sha1) && klmd.Has(sha1)
		S390X.HasSHA256 = kimd.Has(sha256) && klmd.Has(sha256)
		S390X.HasSHA512 = kimd.Has(sha512) && klmd.Has(sha512)
		S390X.HasGHASH = kimd.Has(ghash) // KLMD-GHASH does not exist
	}
}
