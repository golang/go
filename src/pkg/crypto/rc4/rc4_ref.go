// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64,!arm,!386

package rc4

// XORKeyStream sets dst to the result of XORing src with the key stream.
// Dst and src may be the same slice but otherwise should not overlap.
func (c *Cipher) XORKeyStream(dst, src []byte) {
	i, j := c.i, c.j
	for k, v := range src {
		i += 1
		j += c.s[i]
		c.s[i], c.s[j] = c.s[j], c.s[i]
		dst[k] = v ^ c.s[c.s[i]+c.s[j]]
	}
	c.i, c.j = i, j
}
