// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const cacheLineSize = 256

const (
	// bit mask values from /usr/include/bits/hwcap.h
	hwcap_ZARCH  = 2
	hwcap_STFLE  = 4
	hwcap_MSA    = 8
	hwcap_LDISP  = 16
	hwcap_EIMM   = 32
	hwcap_DFP    = 64
	hwcap_ETF3EH = 256
	hwcap_VX     = 2048
	hwcap_VXE    = 8192
)

// bitIsSet reports whether the bit at index is set. The bit index
// is in big endian order, so bit index 0 is the leftmost bit.
func bitIsSet(bits []uint64, index uint) bool {
	return bits[index/64]&((1<<63)>>(index%64)) != 0
}

// function is the code for the named cryptographic function.
type function uint8

const (
	// KM{,A,C,CTR} function codes
	aes128 function = 18 // AES-128
	aes192 function = 19 // AES-192
	aes256 function = 20 // AES-256

	// K{I,L}MD function codes
	sha1     function = 1  // SHA-1
	sha256   function = 2  // SHA-256
	sha512   function = 3  // SHA-512
	sha3_224 function = 32 // SHA3-224
	sha3_256 function = 33 // SHA3-256
	sha3_384 function = 34 // SHA3-384
	sha3_512 function = 35 // SHA3-512
	shake128 function = 36 // SHAKE-128
	shake256 function = 37 // SHAKE-256

	// KLMD function codes
	ghash function = 65 // GHASH
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
	// cryptography facilities
	msa4 facility = 77  // message-security-assist extension 4
	msa8 facility = 146 // message-security-assist extension 8
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

func doinit() {
	// test HWCAP bit vector
	has := func(featureMask uint) bool {
		return hwCap&featureMask == featureMask
	}

	// mandatory
	S390X.HasZARCH = has(hwcap_ZARCH)

	// optional
	S390X.HasSTFLE = has(hwcap_STFLE)
	S390X.HasLDISP = has(hwcap_LDISP)
	S390X.HasEIMM = has(hwcap_EIMM)
	S390X.HasETF3EH = has(hwcap_ETF3EH)
	S390X.HasDFP = has(hwcap_DFP)
	S390X.HasMSA = has(hwcap_MSA)
	S390X.HasVX = has(hwcap_VX)
	if S390X.HasVX {
		S390X.HasVXE = has(hwcap_VXE)
	}

	// We need implementations of stfle, km and so on
	// to detect cryptographic features.
	if !haveAsmFunctions() {
		return
	}

	// optional cryptographic functions
	if S390X.HasMSA {
		aes := []function{aes128, aes192, aes256}

		// cipher message
		km, kmc := kmQuery(), kmcQuery()
		S390X.HasAES = km.Has(aes...)
		S390X.HasAESCBC = kmc.Has(aes...)
		if S390X.HasSTFLE {
			facilities := stfle()
			if facilities.Has(msa4) {
				kmctr := kmctrQuery()
				S390X.HasAESCTR = kmctr.Has(aes...)
			}
			if facilities.Has(msa8) {
				kma := kmaQuery()
				S390X.HasAESGCM = kma.Has(aes...)
			}
		}

		// compute message digest
		kimd := kimdQuery() // intermediate (no padding)
		klmd := klmdQuery() // last (padding)
		S390X.HasSHA1 = kimd.Has(sha1) && klmd.Has(sha1)
		S390X.HasSHA256 = kimd.Has(sha256) && klmd.Has(sha256)
		S390X.HasSHA512 = kimd.Has(sha512) && klmd.Has(sha512)
		S390X.HasGHASH = kimd.Has(ghash) // KLMD-GHASH does not exist
		sha3 := []function{
			sha3_224, sha3_256, sha3_384, sha3_512,
			shake128, shake256,
		}
		S390X.HasSHA3 = kimd.Has(sha3...) && klmd.Has(sha3...)
	}
}
