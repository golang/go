// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"encoding/asn1"
	"errors"
	"math/big"
	"math/bits"
	"strconv"
	"strings"
)

var (
	errInvalidOID = errors.New("invalid oid")
)

// An OID represents an ASN.1 OBJECT IDENTIFIER.
type OID struct {
	der []byte
}

func newOIDFromDER(der []byte) (OID, bool) {
	if len(der) == 0 || der[len(der)-1]&0x80 != 0 {
		return OID{}, false
	}

	start := 0
	for i, v := range der {
		// ITU-T X.690, section 8.19.2:
		// The subidentifier shall be encoded in the fewest possible octets,
		// that is, the leading octet of the subidentifier shall not have the value 0x80.
		if i == start && v == 0x80 {
			return OID{}, false
		}
		if v&0x80 == 0 {
			start = i + 1
		}
	}

	return OID{der}, true
}

func mustNewOIDFromInts(ints []uint64) OID {
	oid, err := OIDFromInts(ints)
	if err != nil {
		panic("crypto/x509: mustNewOIDFromInts: " + err.Error())
	}
	return oid
}

// OIDFromInts creates a new OID using ints, each integer is a separate component.
func OIDFromInts(oid []uint64) (OID, error) {
	if len(oid) < 2 || oid[0] > 2 || (oid[0] < 2 && oid[1] >= 40) {
		return OID{}, errInvalidOID
	}

	length := base128IntLength(oid[0]*40 + oid[1])
	for _, v := range oid[2:] {
		length += base128IntLength(v)
	}

	der := make([]byte, 0, length)
	der = appendBase128Int(der, oid[0]*40+oid[1])
	for _, v := range oid[2:] {
		der = appendBase128Int(der, v)
	}
	return OID{der}, nil
}

func base128IntLength(n uint64) int {
	if n == 0 {
		return 1
	}
	return (bits.Len64(n) + 6) / 7
}

func appendBase128Int(dst []byte, n uint64) []byte {
	for i := base128IntLength(n) - 1; i >= 0; i-- {
		o := byte(n >> uint(i*7))
		o &= 0x7f
		if i != 0 {
			o |= 0x80
		}
		dst = append(dst, o)
	}
	return dst
}

// Equal returns true when oid and other represents the same Object Identifier.
func (oid OID) Equal(other OID) bool {
	// There is only one possible DER encoding of
	// each unique Object Identifier.
	return bytes.Equal(oid.der, other.der)
}

// EqualASN1OID returns whether an OID equals an asn1.ObjectIdentifier. If
// asn1.ObjectIdentifier cannot represent the OID specified by oid, because
// a component of OID requires more than 31 bits, it returns false.
func (oid OID) EqualASN1OID(other asn1.ObjectIdentifier) bool {
	const (
		valSize         = 31 // amount of usable bits of val for OIDs.
		bitsPerByte     = 7
		maxValSafeShift = (1 << (valSize - bitsPerByte)) - 1
	)
	var (
		val   = 0
		first = true
	)
	for _, v := range oid.der {
		if val > maxValSafeShift {
			return false
		}
		val <<= bitsPerByte
		val |= int(v & 0x7F)
		if v&0x80 == 0 {
			if first {
				if len(other) < 2 {
					return false
				}
				var val1, val2 int
				if val < 80 {
					val1 = val / 40
					val2 = val % 40
				} else {
					val1 = 2
					val2 = val - 80
				}
				if val1 != other[0] || val2 != other[1] {
					return false
				}
				val = 0
				first = false
				other = other[2:]
				continue
			}
			if len(other) == 0 {
				return false
			}
			if val != other[0] {
				return false
			}
			val = 0
			other = other[1:]
		}
	}
	return true
}

// Strings returns the string representation of the Object Identifier.
func (oid OID) String() string {
	var b strings.Builder
	b.Grow(32)
	const (
		valSize         = 64 // size in bits of val.
		bitsPerByte     = 7
		maxValSafeShift = (1 << (valSize - bitsPerByte)) - 1
	)
	var (
		start    = 0
		val      = uint64(0)
		numBuf   = make([]byte, 0, 21)
		bigVal   *big.Int
		overflow bool
	)
	for i, v := range oid.der {
		curVal := v & 0x7F
		valEnd := v&0x80 == 0
		if valEnd {
			if start != 0 {
				b.WriteByte('.')
			}
		}
		if !overflow && val > maxValSafeShift {
			if bigVal == nil {
				bigVal = new(big.Int)
			}
			bigVal = bigVal.SetUint64(val)
			overflow = true
		}
		if overflow {
			bigVal = bigVal.Lsh(bigVal, bitsPerByte).Or(bigVal, big.NewInt(int64(curVal)))
			if valEnd {
				if start == 0 {
					b.WriteString("2.")
					bigVal = bigVal.Sub(bigVal, big.NewInt(80))
				}
				numBuf = bigVal.Append(numBuf, 10)
				b.Write(numBuf)
				numBuf = numBuf[:0]
				val = 0
				start = i + 1
				overflow = false
			}
			continue
		}
		val <<= bitsPerByte
		val |= uint64(curVal)
		if valEnd {
			if start == 0 {
				if val < 80 {
					b.Write(strconv.AppendUint(numBuf, val/40, 10))
					b.WriteByte('.')
					b.Write(strconv.AppendUint(numBuf, val%40, 10))
				} else {
					b.WriteString("2.")
					b.Write(strconv.AppendUint(numBuf, val-80, 10))
				}
			} else {
				b.Write(strconv.AppendUint(numBuf, val, 10))
			}
			val = 0
			start = i + 1
		}
	}
	return b.String()
}

func (oid OID) toASN1OID() (asn1.ObjectIdentifier, bool) {
	out := make([]int, 0, len(oid.der)+1)

	const (
		valSize         = 31 // amount of usable bits of val for OIDs.
		bitsPerByte     = 7
		maxValSafeShift = (1 << (valSize - bitsPerByte)) - 1
	)

	val := 0

	for _, v := range oid.der {
		if val > maxValSafeShift {
			return nil, false
		}

		val <<= bitsPerByte
		val |= int(v & 0x7F)

		if v&0x80 == 0 {
			if len(out) == 0 {
				if val < 80 {
					out = append(out, val/40)
					out = append(out, val%40)
				} else {
					out = append(out, 2)
					out = append(out, val-80)
				}
				val = 0
				continue
			}
			out = append(out, val)
			val = 0
		}
	}

	return out, true
}
