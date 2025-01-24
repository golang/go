// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbkdf2

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/hmac"
	"errors"
)

// divRoundUp divides x+y-1 by y, rounding up if the result is not whole.
// This function casts x and y to int64 in order to avoid cases where
// x+y would overflow int on systems where int is an int32. The result
// is an int, which is safe as (x+y-1)/y should always fit, regardless
// of the integer size.
func divRoundUp(x, y int) int {
	return int((int64(x) + int64(y) - 1) / int64(y))
}

func Key[Hash fips140.Hash](h func() Hash, password string, salt []byte, iter, keyLength int) ([]byte, error) {
	setServiceIndicator(salt, keyLength)

	if keyLength <= 0 {
		return nil, errors.New("pkbdf2: keyLength must be larger than 0")
	}

	prf := hmac.New(h, []byte(password))
	hmac.MarkAsUsedInKDF(prf)
	hashLen := prf.Size()
	numBlocks := divRoundUp(keyLength, hashLen)
	const maxBlocks = int64(1<<32 - 1)
	if keyLength+hashLen < keyLength || int64(numBlocks) > maxBlocks {
		return nil, errors.New("pbkdf2: keyLength too long")
	}

	var buf [4]byte
	dk := make([]byte, 0, numBlocks*hashLen)
	U := make([]byte, hashLen)
	for block := 1; block <= numBlocks; block++ {
		// N.B.: || means concatenation, ^ means XOR
		// for each block T_i = U_1 ^ U_2 ^ ... ^ U_iter
		// U_1 = PRF(password, salt || uint(i))
		prf.Reset()
		prf.Write(salt)
		buf[0] = byte(block >> 24)
		buf[1] = byte(block >> 16)
		buf[2] = byte(block >> 8)
		buf[3] = byte(block)
		prf.Write(buf[:4])
		dk = prf.Sum(dk)
		T := dk[len(dk)-hashLen:]
		copy(U, T)

		// U_n = PRF(password, U_(n-1))
		for n := 2; n <= iter; n++ {
			prf.Reset()
			prf.Write(U)
			U = U[:0]
			U = prf.Sum(U)
			for x := range U {
				T[x] ^= U[x]
			}
		}
	}
	return dk[:keyLength], nil
}

func setServiceIndicator(salt []byte, keyLength int) {
	// The HMAC construction will handle the hash function considerations for the service
	// indicator. The remaining PBKDF2 considerations outlined by SP 800-132 pertain to
	// salt and keyLength.

	// The length of the randomly-generated portion of the salt shall be at least 128 bits.
	if len(salt) < 128/8 {
		fips140.RecordNonApproved()
	}

	// Per FIPS 140-3 IG C.M, key lengths below 112 bits are only allowed for
	// legacy use (i.e. verification only) and we don't support that.
	if keyLength < 112/8 {
		fips140.RecordNonApproved()
	}

	fips140.RecordApproved()
}
