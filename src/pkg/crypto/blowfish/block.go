// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package blowfish

// ExpandKey performs a key expansion on the given *Cipher. Specifically, it
// performs the Blowfish algorithm's key schedule which sets up the *Cipher's
// pi and substitution tables for calls to Encrypt. This is used, primarily,
// by the bcrypt package to reuse the Blowfish key schedule during its
// set up. It's unlikely that you need to use this directly.
func ExpandKey(key []byte, c *Cipher) {
	j := 0
	for i := 0; i < 18; i++ {
		var d uint32
		for k := 0; k < 4; k++ {
			d = d<<8 | uint32(key[j])&0x000000FF
			j++
			if j >= len(key) {
				j = 0
			}
		}
		c.p[i] ^= d
	}

	var l, r uint32
	for i := 0; i < 18; i += 2 {
		l, r = encryptBlock(l, r, c)
		c.p[i], c.p[i+1] = l, r
	}

	for i := 0; i < 256; i += 2 {
		l, r = encryptBlock(l, r, c)
		c.s0[i], c.s0[i+1] = l, r
	}
	for i := 0; i < 256; i += 2 {
		l, r = encryptBlock(l, r, c)
		c.s1[i], c.s1[i+1] = l, r
	}
	for i := 0; i < 256; i += 2 {
		l, r = encryptBlock(l, r, c)
		c.s2[i], c.s2[i+1] = l, r
	}
	for i := 0; i < 256; i += 2 {
		l, r = encryptBlock(l, r, c)
		c.s3[i], c.s3[i+1] = l, r
	}
}

// This is similar to ExpandKey, but folds the salt during the key
// schedule. While ExpandKey is essentially expandKeyWithSalt with an all-zero
// salt passed in, reusing ExpandKey turns out to be a place of inefficiency
// and specializing it here is useful.
func expandKeyWithSalt(key []byte, salt []byte, c *Cipher) {
	j := 0
	expandedKey := make([]uint32, 18)
	for i := 0; i < 18; i++ {
		var d uint32
		for k := 0; k < 4; k++ {
			d = d<<8 | uint32(key[j])&0x000000FF
			j++
			if j >= len(key) {
				j = 0
			}
		}
		expandedKey[i] = d
		c.p[i] ^= d
	}

	j = 0
	expandedSalt := make([]uint32, 18)
	for i := 0; i < 18; i++ {
		var d uint32
		for k := 0; k < 4; k++ {
			d = d<<8 | uint32(salt[j])&0x000000FF
			j++
			if j >= len(salt) {
				j = 0
			}
		}
		expandedSalt[i] = d
	}

	var l, r uint32
	for i := 0; i < 18; i += 2 {
		l ^= expandedSalt[i&2]
		r ^= expandedSalt[(i&2)+1]
		l, r = encryptBlock(l, r, c)
		c.p[i], c.p[i+1] = l, r
	}

	for i := 0; i < 256; i += 4 {
		l ^= expandedSalt[2]
		r ^= expandedSalt[3]
		l, r = encryptBlock(l, r, c)
		c.s0[i], c.s0[i+1] = l, r

		l ^= expandedSalt[0]
		r ^= expandedSalt[1]
		l, r = encryptBlock(l, r, c)
		c.s0[i+2], c.s0[i+3] = l, r

	}

	for i := 0; i < 256; i += 4 {
		l ^= expandedSalt[2]
		r ^= expandedSalt[3]
		l, r = encryptBlock(l, r, c)
		c.s1[i], c.s1[i+1] = l, r

		l ^= expandedSalt[0]
		r ^= expandedSalt[1]
		l, r = encryptBlock(l, r, c)
		c.s1[i+2], c.s1[i+3] = l, r
	}

	for i := 0; i < 256; i += 4 {
		l ^= expandedSalt[2]
		r ^= expandedSalt[3]
		l, r = encryptBlock(l, r, c)
		c.s2[i], c.s2[i+1] = l, r

		l ^= expandedSalt[0]
		r ^= expandedSalt[1]
		l, r = encryptBlock(l, r, c)
		c.s2[i+2], c.s2[i+3] = l, r
	}

	for i := 0; i < 256; i += 4 {
		l ^= expandedSalt[2]
		r ^= expandedSalt[3]
		l, r = encryptBlock(l, r, c)
		c.s3[i], c.s3[i+1] = l, r

		l ^= expandedSalt[0]
		r ^= expandedSalt[1]
		l, r = encryptBlock(l, r, c)
		c.s3[i+2], c.s3[i+3] = l, r
	}
}

func encryptBlock(l, r uint32, c *Cipher) (uint32, uint32) {
	xl, xr := l, r
	xl ^= c.p[0]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[1]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[2]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[3]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[4]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[5]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[6]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[7]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[8]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[9]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[10]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[11]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[12]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[13]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[14]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[15]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[16]
	xr ^= c.p[17]
	return xr, xl
}

func decryptBlock(l, r uint32, c *Cipher) (uint32, uint32) {
	xl, xr := l, r
	xl ^= c.p[17]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[16]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[15]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[14]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[13]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[12]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[11]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[10]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[9]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[8]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[7]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[6]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[5]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[4]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[3]
	xr ^= ((c.s0[byte(xl>>24)] + c.s1[byte(xl>>16)]) ^ c.s2[byte(xl>>8)]) + c.s3[byte(xl)] ^ c.p[2]
	xl ^= ((c.s0[byte(xr>>24)] + c.s1[byte(xr>>16)]) ^ c.s2[byte(xr>>8)]) + c.s3[byte(xr)] ^ c.p[1]
	xr ^= c.p[0]
	return xr, xl
}

func zero(x []uint32) {
	for i := range x {
		x[i] = 0
	}
}
