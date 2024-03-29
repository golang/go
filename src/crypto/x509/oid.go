// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"encoding/asn1"
	"errors"
	"math"
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

// ParseOID parses a Object Identifier string, represented by ASCII numbers separated by dots.
func ParseOID(oid string) (OID, error) {
	var o OID
	return o, o.UnmarshalText([]byte(oid))
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

func base128BigIntLength(n *big.Int) int {
	if n.Cmp(big.NewInt(0)) == 0 {
		return 1
	}
	return (n.BitLen() + 6) / 7
}

func appendBase128BigInt(dst []byte, n *big.Int) []byte {
	if n.Cmp(big.NewInt(0)) == 0 {
		return append(dst, 0)
	}

	for i := base128BigIntLength(n) - 1; i >= 0; i-- {
		o := byte(big.NewInt(0).Rsh(n, uint(i)*7).Bits()[0])
		o &= 0x7f
		if i != 0 {
			o |= 0x80
		}
		dst = append(dst, o)
	}
	return dst
}

// MarshalText implements [encoding.TextMarshaler]
func (o OID) MarshalText() ([]byte, error) {
	return []byte(o.String()), nil
}

// MarshalText implements [encoding.TextUnmarshaler]
func (o *OID) UnmarshalText(text []byte) error {
	oid := string(text)

	for _, c := range oid {
		isDigit := c >= '0' && c <= '9'
		if !isDigit && c != ' ' && c != '.' {
			return errInvalidOID
		}
	}

	end := strings.IndexByte(oid, '.')
	if end == -1 {
		return errInvalidOID
	}

	firstNum := oid[:end]
	oid = oid[end+1:]
	end = strings.IndexByte(oid, '.')
	if end == -1 {
		end = len(oid)
	}
	secondNum := oid[:end]

	var (
		first  = big.NewInt(0)
		second = big.NewInt(0)
	)

	if _, ok := first.SetString(firstNum, 10); !ok {
		return errInvalidOID
	}
	if _, ok := second.SetString(secondNum, 10); !ok {
		return errInvalidOID
	}

	if first.Cmp(big.NewInt(2)) > 0 || (first.Cmp(big.NewInt(2)) < 0 && second.Cmp(big.NewInt(40)) >= 0) {
		return errInvalidOID
	}

	firstComponent := first.Mul(first, big.NewInt(40))
	firstComponent.Add(firstComponent, second)

	der := appendBase128BigInt(make([]byte, 0, 32), firstComponent)

	if end != len(oid) {
		oid = oid[end+1:]
		for {
			end := strings.IndexByte(oid, '.')
			if end == -1 {
				end = len(oid)
			}

			b, ok := big.NewInt(0).SetString(oid[:end], 10)
			if !ok {
				return errInvalidOID
			}
			der = appendBase128BigInt(der, b)

			if end == len(oid) {
				break
			}
			oid = oid[end+1:]
		}
	}

	o.der = der
	return nil
}

// MarshalText implements [encoding.BinaryMarshaler]
func (o OID) MarshalBinary() ([]byte, error) {
	return bytes.Clone(o.der), nil
}

// MarshalText implements [encoding.BinaryUnmarshaler]
func (o *OID) UnmarshalBinary(b []byte) error {
	oid, ok := newOIDFromDER(b)
	if !ok {
		return errInvalidOID
	}
	*o = oid
	return nil
}

// Equal returns true when oid and other represents the same Object Identifier.
func (oid OID) Equal(other OID) bool {
	// There is only one possible DER encoding of
	// each unique Object Identifier.
	return bytes.Equal(oid.der, other.der)
}

func parseBase128Int(bytes []byte, initOffset int) (ret, offset int, failed bool) {
	offset = initOffset
	var ret64 int64
	for shifted := 0; offset < len(bytes); shifted++ {
		// 5 * 7 bits per byte == 35 bits of data
		// Thus the representation is either non-minimal or too large for an int32
		if shifted == 5 {
			failed = true
			return
		}
		ret64 <<= 7
		b := bytes[offset]
		// integers should be minimally encoded, so the leading octet should
		// never be 0x80
		if shifted == 0 && b == 0x80 {
			failed = true
			return
		}
		ret64 |= int64(b & 0x7f)
		offset++
		if b&0x80 == 0 {
			ret = int(ret64)
			// Ensure that the returned value fits in an int on all platforms
			if ret64 > math.MaxInt32 {
				failed = true
			}
			return
		}
	}
	failed = true
	return
}

// EqualASN1OID returns whether an OID equals an asn1.ObjectIdentifier. If
// asn1.ObjectIdentifier cannot represent the OID specified by oid, because
// a component of OID requires more than 31 bits, it returns false.
func (oid OID) EqualASN1OID(other asn1.ObjectIdentifier) bool {
	if len(other) < 2 {
		return false
	}
	v, offset, failed := parseBase128Int(oid.der, 0)
	if failed {
		// This should never happen, since we've already parsed the OID,
		// but just in case.
		return false
	}
	if v < 80 {
		a, b := v/40, v%40
		if other[0] != a || other[1] != b {
			return false
		}
	} else {
		a, b := 2, v-80
		if other[0] != a || other[1] != b {
			return false
		}
	}

	i := 2
	for ; offset < len(oid.der); i++ {
		v, offset, failed = parseBase128Int(oid.der, offset)
		if failed {
			// Again, shouldn't happen, since we've already parsed
			// the OID, but better safe than sorry.
			return false
		}
		if v != other[i] {
			return false
		}
	}

	return i == len(other)
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
