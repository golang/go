// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package asn1 implements parsing of DER-encoded ASN.1 data structures,
// as defined in ITU-T Rec X.690.
//
// See also “A Layman's Guide to a Subset of ASN.1, BER, and DER,”
// http://luca.ntop.org/Teaching/Appunti/asn1.html.
package asn1

// ASN.1 is a syntax for specifying abstract objects and BER, DER, PER, XER etc
// are different encoding formats for those objects. Here, we'll be dealing
// with DER, the Distinguished Encoding Rules. DER is used in X.509 because
// it's fast to parse and, unlike BER, has a unique encoding for every object.
// When calculating hashes over objects, it's important that the resulting
// bytes be the same at both ends and DER removes this margin of error.
//
// ASN.1 is very complex and this package doesn't attempt to implement
// everything by any means.

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"strconv"
	"time"
	"unicode/utf16"
	"unicode/utf8"
)

// A StructuralError suggests that the ASN.1 data is valid, but the Go type
// which is receiving it doesn't match.
type StructuralError struct {
	Msg string
}

func (e StructuralError) Error() string { return "asn1: structure error: " + e.Msg }

// A SyntaxError suggests that the ASN.1 data is invalid.
type SyntaxError struct {
	Msg string
}

func (e SyntaxError) Error() string { return "asn1: syntax error: " + e.Msg }

// We start by dealing with each of the primitive types in turn.

// BOOLEAN

func parseBool(bytes []byte) (ret bool, err error) {
	if len(bytes) != 1 {
		err = SyntaxError{"invalid boolean"}
		return
	}

	// DER demands that "If the encoding represents the boolean value TRUE,
	// its single contents octet shall have all eight bits set to one."
	// Thus only 0 and 255 are valid encoded values.
	switch bytes[0] {
	case 0:
		ret = false
	case 0xff:
		ret = true
	default:
		err = SyntaxError{"invalid boolean"}
	}

	return
}

// INTEGER

// checkInteger returns nil if the given bytes are a valid DER-encoded
// INTEGER and an error otherwise.
func checkInteger(bytes []byte) error {
	if len(bytes) == 0 {
		return StructuralError{"empty integer"}
	}
	if len(bytes) == 1 {
		return nil
	}
	if (bytes[0] == 0 && bytes[1]&0x80 == 0) || (bytes[0] == 0xff && bytes[1]&0x80 == 0x80) {
		return StructuralError{"integer not minimally-encoded"}
	}
	return nil
}

// parseInt64 treats the given bytes as a big-endian, signed integer and
// returns the result.
func parseInt64(bytes []byte) (ret int64, err error) {
	err = checkInteger(bytes)
	if err != nil {
		return
	}
	if len(bytes) > 8 {
		// We'll overflow an int64 in this case.
		err = StructuralError{"integer too large"}
		return
	}
	for bytesRead := 0; bytesRead < len(bytes); bytesRead++ {
		ret <<= 8
		ret |= int64(bytes[bytesRead])
	}

	// Shift up and down in order to sign extend the result.
	ret <<= 64 - uint8(len(bytes))*8
	ret >>= 64 - uint8(len(bytes))*8
	return
}

// parseInt treats the given bytes as a big-endian, signed integer and returns
// the result.
func parseInt32(bytes []byte) (int32, error) {
	if err := checkInteger(bytes); err != nil {
		return 0, err
	}
	ret64, err := parseInt64(bytes)
	if err != nil {
		return 0, err
	}
	if ret64 != int64(int32(ret64)) {
		return 0, StructuralError{"integer too large"}
	}
	return int32(ret64), nil
}

var bigOne = big.NewInt(1)

// parseBigInt treats the given bytes as a big-endian, signed integer and returns
// the result.
func parseBigInt(bytes []byte) (*big.Int, error) {
	if err := checkInteger(bytes); err != nil {
		return nil, err
	}
	ret := new(big.Int)
	if len(bytes) > 0 && bytes[0]&0x80 == 0x80 {
		// This is a negative number.
		notBytes := make([]byte, len(bytes))
		for i := range notBytes {
			notBytes[i] = ^bytes[i]
		}
		ret.SetBytes(notBytes)
		ret.Add(ret, bigOne)
		ret.Neg(ret)
		return ret, nil
	}
	ret.SetBytes(bytes)
	return ret, nil
}

// BIT STRING

// BitString is the structure to use when you want an ASN.1 BIT STRING type. A
// bit string is padded up to the nearest byte in memory and the number of
// valid bits is recorded. Padding bits will be zero.
type BitString struct {
	Bytes     []byte // bits packed into bytes.
	BitLength int    // length in bits.
}

// At returns the bit at the given index. If the index is out of range it
// returns false.
func (b BitString) At(i int) int {
	if i < 0 || i >= b.BitLength {
		return 0
	}
	x := i / 8
	y := 7 - uint(i%8)
	return int(b.Bytes[x]>>y) & 1
}

// RightAlign returns a slice where the padding bits are at the beginning. The
// slice may share memory with the BitString.
func (b BitString) RightAlign() []byte {
	shift := uint(8 - (b.BitLength % 8))
	if shift == 8 || len(b.Bytes) == 0 {
		return b.Bytes
	}

	a := make([]byte, len(b.Bytes))
	a[0] = b.Bytes[0] >> shift
	for i := 1; i < len(b.Bytes); i++ {
		a[i] = b.Bytes[i-1] << (8 - shift)
		a[i] |= b.Bytes[i] >> shift
	}

	return a
}

// parseBitString parses an ASN.1 bit string from the given byte slice and returns it.
func parseBitString(bytes []byte) (ret BitString, err error) {
	if len(bytes) == 0 {
		err = SyntaxError{"zero length BIT STRING"}
		return
	}
	paddingBits := int(bytes[0])
	if paddingBits > 7 ||
		len(bytes) == 1 && paddingBits > 0 ||
		bytes[len(bytes)-1]&((1<<bytes[0])-1) != 0 {
		err = SyntaxError{"invalid padding bits in BIT STRING"}
		return
	}
	ret.BitLength = (len(bytes)-1)*8 - paddingBits
	ret.Bytes = bytes[1:]
	return
}

// NULL

// NullRawValue is a RawValue with its Tag set to the ASN.1 NULL type tag (5).
var NullRawValue = RawValue{Tag: TagNull}

// NullBytes contains bytes representing the DER-encoded ASN.1 NULL type.
var NullBytes = []byte{TagNull, 0}

// OBJECT IDENTIFIER

// An ObjectIdentifier represents an ASN.1 OBJECT IDENTIFIER.
type ObjectIdentifier []int

// Equal reports whether oi and other represent the same identifier.
func (oi ObjectIdentifier) Equal(other ObjectIdentifier) bool {
	if len(oi) != len(other) {
		return false
	}
	for i := 0; i < len(oi); i++ {
		if oi[i] != other[i] {
			return false
		}
	}

	return true
}

func (oi ObjectIdentifier) String() string {
	var s string

	for i, v := range oi {
		if i > 0 {
			s += "."
		}
		s += strconv.Itoa(v)
	}

	return s
}

// parseObjectIdentifier parses an OBJECT IDENTIFIER from the given bytes and
// returns it. An object identifier is a sequence of variable length integers
// that are assigned in a hierarchy.
func parseObjectIdentifier(bytes []byte) (s ObjectIdentifier, err error) {
	if len(bytes) == 0 {
		err = SyntaxError{"zero length OBJECT IDENTIFIER"}
		return
	}

	// In the worst case, we get two elements from the first byte (which is
	// encoded differently) and then every varint is a single byte long.
	s = make([]int, len(bytes)+1)

	// The first varint is 40*value1 + value2:
	// According to this packing, value1 can take the values 0, 1 and 2 only.
	// When value1 = 0 or value1 = 1, then value2 is <= 39. When value1 = 2,
	// then there are no restrictions on value2.
	v, offset, err := parseBase128Int(bytes, 0)
	if err != nil {
		return
	}
	if v < 80 {
		s[0] = v / 40
		s[1] = v % 40
	} else {
		s[0] = 2
		s[1] = v - 80
	}

	i := 2
	for ; offset < len(bytes); i++ {
		v, offset, err = parseBase128Int(bytes, offset)
		if err != nil {
			return
		}
		s[i] = v
	}
	s = s[0:i]
	return
}

// ENUMERATED

// An Enumerated is represented as a plain int.
type Enumerated int

// FLAG

// A Flag accepts any data and is set to true if present.
type Flag bool

// parseBase128Int parses a base-128 encoded int from the given offset in the
// given byte slice. It returns the value and the new offset.
func parseBase128Int(bytes []byte, initOffset int) (ret, offset int, err error) {
	offset = initOffset
	var ret64 int64
	for shifted := 0; offset < len(bytes); shifted++ {
		// 5 * 7 bits per byte == 35 bits of data
		// Thus the representation is either non-minimal or too large for an int32
		if shifted == 5 {
			err = StructuralError{"base 128 integer too large"}
			return
		}
		ret64 <<= 7
		b := bytes[offset]
		// integers should be minimally encoded, so the leading octet should
		// never be 0x80
		if shifted == 0 && b == 0x80 {
			err = SyntaxError{"integer is not minimally encoded"}
			return
		}
		ret64 |= int64(b & 0x7f)
		offset++
		if b&0x80 == 0 {
			ret = int(ret64)
			// Ensure that the returned value fits in an int on all platforms
			if ret64 > math.MaxInt32 {
				err = StructuralError{"base 128 integer too large"}
			}
			return
		}
	}
	err = SyntaxError{"truncated base 128 integer"}
	return
}

// UTCTime

func parseUTCTime(bytes []byte) (ret time.Time, err error) {
	s := string(bytes)

	formatStr := "0601021504Z0700"
	ret, err = time.Parse(formatStr, s)
	if err != nil {
		formatStr = "060102150405Z0700"
		ret, err = time.Parse(formatStr, s)
	}
	if err != nil {
		return
	}

	if serialized := ret.Format(formatStr); serialized != s {
		err = fmt.Errorf("asn1: time did not serialize back to the original value and may be invalid: given %q, but serialized as %q", s, serialized)
		return
	}

	if ret.Year() >= 2050 {
		// UTCTime only encodes times prior to 2050. See https://tools.ietf.org/html/rfc5280#section-4.1.2.5.1
		ret = ret.AddDate(-100, 0, 0)
	}

	return
}

// parseGeneralizedTime parses the GeneralizedTime from the given byte slice
// and returns the resulting time.
func parseGeneralizedTime(bytes []byte) (ret time.Time, err error) {
	const formatStr = "20060102150405Z0700"
	s := string(bytes)

	if ret, err = time.Parse(formatStr, s); err != nil {
		return
	}

	if serialized := ret.Format(formatStr); serialized != s {
		err = fmt.Errorf("asn1: time did not serialize back to the original value and may be invalid: given %q, but serialized as %q", s, serialized)
	}

	return
}

// NumericString

// parseNumericString parses an ASN.1 NumericString from the given byte array
// and returns it.
func parseNumericString(bytes []byte) (ret string, err error) {
	for _, b := range bytes {
		if !isNumeric(b) {
			return "", SyntaxError{"NumericString contains invalid character"}
		}
	}
	return string(bytes), nil
}

// isNumeric reports whether the given b is in the ASN.1 NumericString set.
func isNumeric(b byte) bool {
	return '0' <= b && b <= '9' ||
		b == ' '
}

// PrintableString

// parsePrintableString parses an ASN.1 PrintableString from the given byte
// array and returns it.
func parsePrintableString(bytes []byte) (ret string, err error) {
	for _, b := range bytes {
		if !isPrintable(b, allowAsterisk, allowAmpersand) {
			err = SyntaxError{"PrintableString contains invalid character"}
			return
		}
	}
	ret = string(bytes)
	return
}

type asteriskFlag bool
type ampersandFlag bool

const (
	allowAsterisk  asteriskFlag = true
	rejectAsterisk asteriskFlag = false

	allowAmpersand  ampersandFlag = true
	rejectAmpersand ampersandFlag = false
)

// isPrintable reports whether the given b is in the ASN.1 PrintableString set.
// If asterisk is allowAsterisk then '*' is also allowed, reflecting existing
// practice. If ampersand is allowAmpersand then '&' is allowed as well.
func isPrintable(b byte, asterisk asteriskFlag, ampersand ampersandFlag) bool {
	return 'a' <= b && b <= 'z' ||
		'A' <= b && b <= 'Z' ||
		'0' <= b && b <= '9' ||
		'\'' <= b && b <= ')' ||
		'+' <= b && b <= '/' ||
		b == ' ' ||
		b == ':' ||
		b == '=' ||
		b == '?' ||
		// This is technically not allowed in a PrintableString.
		// However, x509 certificates with wildcard strings don't
		// always use the correct string type so we permit it.
		(bool(asterisk) && b == '*') ||
		// This is not technically allowed either. However, not
		// only is it relatively common, but there are also a
		// handful of CA certificates that contain it. At least
		// one of which will not expire until 2027.
		(bool(ampersand) && b == '&')
}

// IA5String

// parseIA5String parses an ASN.1 IA5String (ASCII string) from the given
// byte slice and returns it.
func parseIA5String(bytes []byte) (ret string, err error) {
	for _, b := range bytes {
		if b >= utf8.RuneSelf {
			err = SyntaxError{"IA5String contains invalid character"}
			return
		}
	}
	ret = string(bytes)
	return
}

// T61String

// parseT61String parses an ASN.1 T61String (8-bit clean string) from the given
// byte slice and returns it.
func parseT61String(bytes []byte) (ret string, err error) {
	return string(bytes), nil
}

// UTF8String

// parseUTF8String parses an ASN.1 UTF8String (raw UTF-8) from the given byte
// array and returns it.
func parseUTF8String(bytes []byte) (ret string, err error) {
	if !utf8.Valid(bytes) {
		return "", errors.New("asn1: invalid UTF-8 string")
	}
	return string(bytes), nil
}

// BMPString

// parseBMPString parses an ASN.1 BMPString (Basic Multilingual Plane of
// ISO/IEC/ITU 10646-1) from the given byte slice and returns it.
func parseBMPString(bmpString []byte) (string, error) {
	if len(bmpString)%2 != 0 {
		return "", errors.New("pkcs12: odd-length BMP string")
	}

	// Strip terminator if present.
	if l := len(bmpString); l >= 2 && bmpString[l-1] == 0 && bmpString[l-2] == 0 {
		bmpString = bmpString[:l-2]
	}

	s := make([]uint16, 0, len(bmpString)/2)
	for len(bmpString) > 0 {
		s = append(s, uint16(bmpString[0])<<8+uint16(bmpString[1]))
		bmpString = bmpString[2:]
	}

	return string(utf16.Decode(s)), nil
}

// A RawValue represents an undecoded ASN.1 object.
type RawValue struct {
	Class, Tag int
	IsCompound bool
	Bytes      []byte
	FullBytes  []byte // includes the tag and length
}

// RawContent is used to signal that the undecoded, DER data needs to be
// preserved for a struct. To use it, the first field of the struct must have
// this type. It's an error for any of the other fields to have this type.
type RawContent []byte

// Tagging

// parseTagAndLength parses an ASN.1 tag and length pair from the given offset
// into a byte slice. It returns the parsed data and the new offset. SET and
// SET OF (tag 17) are mapped to SEQUENCE and SEQUENCE OF (tag 16) since we
// don't distinguish between ordered and unordered objects in this code.
func parseTagAndLength(bytes []byte, initOffset int) (ret tagAndLength, offset int, err error) {
	offset = initOffset
	// parseTagAndLength should not be called without at least a single
	// byte to read. Thus this check is for robustness:
	if offset >= len(bytes) {
		err = errors.New("asn1: internal error in parseTagAndLength")
		return
	}
	b := bytes[offset]
	offset++
	ret.class = int(b >> 6)
	ret.isCompound = b&0x20 == 0x20
	ret.tag = int(b & 0x1f)

	// If the bottom five bits are set, then the tag number is actually base 128
	// encoded afterwards
	if ret.tag == 0x1f {
		ret.tag, offset, err = parseBase128Int(bytes, offset)
		if err != nil {
			return
		}
		// Tags should be encoded in minimal form.
		if ret.tag < 0x1f {
			err = SyntaxError{"non-minimal tag"}
			return
		}
	}
	if offset >= len(bytes) {
		err = SyntaxError{"truncated tag or length"}
		return
	}
	b = bytes[offset]
	offset++
	if b&0x80 == 0 {
		// The length is encoded in the bottom 7 bits.
		ret.length = int(b & 0x7f)
	} else {
		// Bottom 7 bits give the number of length bytes to follow.
		numBytes := int(b & 0x7f)
		if numBytes == 0 {
			err = SyntaxError{"indefinite length found (not DER)"}
			return
		}
		ret.length = 0
		for i := 0; i < numBytes; i++ {
			if offset >= len(bytes) {
				err = SyntaxError{"truncated tag or length"}
				return
			}
			b = bytes[offset]
			offset++
			if ret.length >= 1<<23 {
				// We can't shift ret.length up without
				// overflowing.
				err = StructuralError{"length too large"}
				return
			}
			ret.length <<= 8
			ret.length |= int(b)
			if ret.length == 0 {
				// DER requires that lengths be minimal.
				err = StructuralError{"superfluous leading zeros in length"}
				return
			}
		}
		// Short lengths must be encoded in short form.
		if ret.length < 0x80 {
			err = StructuralError{"non-minimal length"}
			return
		}
	}

	return
}

// parseSequenceOf is used for SEQUENCE OF and SET OF values. It tries to parse
// a number of ASN.1 values from the given byte slice and returns them as a
// slice of Go values of the given type.
func parseSequenceOf(bytes []byte, sliceType reflect.Type, elemType reflect.Type) (ret reflect.Value, err error) {
	matchAny, expectedTag, compoundType, ok := getUniversalType(elemType)
	if !ok {
		err = StructuralError{"unknown Go type for slice"}
		return
	}

	// First we iterate over the input and count the number of elements,
	// checking that the types are correct in each case.
	numElements := 0
	for offset := 0; offset < len(bytes); {
		var t tagAndLength
		t, offset, err = parseTagAndLength(bytes, offset)
		if err != nil {
			return
		}
		switch t.tag {
		case TagIA5String, TagGeneralString, TagT61String, TagUTF8String, TagNumericString, TagBMPString:
			// We pretend that various other string types are
			// PRINTABLE STRINGs so that a sequence of them can be
			// parsed into a []string.
			t.tag = TagPrintableString
		case TagGeneralizedTime, TagUTCTime:
			// Likewise, both time types are treated the same.
			t.tag = TagUTCTime
		}

		if !matchAny && (t.class != ClassUniversal || t.isCompound != compoundType || t.tag != expectedTag) {
			err = StructuralError{"sequence tag mismatch"}
			return
		}
		if invalidLength(offset, t.length, len(bytes)) {
			err = SyntaxError{"truncated sequence"}
			return
		}
		offset += t.length
		numElements++
	}
	ret = reflect.MakeSlice(sliceType, numElements, numElements)
	params := fieldParameters{}
	offset := 0
	for i := 0; i < numElements; i++ {
		offset, err = parseField(ret.Index(i), bytes, offset, params)
		if err != nil {
			return
		}
	}
	return
}

var (
	bitStringType        = reflect.TypeOf(BitString{})
	objectIdentifierType = reflect.TypeOf(ObjectIdentifier{})
	enumeratedType       = reflect.TypeOf(Enumerated(0))
	flagType             = reflect.TypeOf(Flag(false))
	timeType             = reflect.TypeOf(time.Time{})
	rawValueType         = reflect.TypeOf(RawValue{})
	rawContentsType      = reflect.TypeOf(RawContent(nil))
	bigIntType           = reflect.TypeOf(new(big.Int))
)

// invalidLength reports whether offset + length > sliceLength, or if the
// addition would overflow.
func invalidLength(offset, length, sliceLength int) bool {
	return offset+length < offset || offset+length > sliceLength
}

// parseField is the main parsing function. Given a byte slice and an offset
// into the array, it will try to parse a suitable ASN.1 value out and store it
// in the given Value.
func parseField(v reflect.Value, bytes []byte, initOffset int, params fieldParameters) (offset int, err error) {
	offset = initOffset
	fieldType := v.Type()

	// If we have run out of data, it may be that there are optional elements at the end.
	if offset == len(bytes) {
		if !setDefaultValue(v, params) {
			err = SyntaxError{"sequence truncated"}
		}
		return
	}

	// Deal with the ANY type.
	if ifaceType := fieldType; ifaceType.Kind() == reflect.Interface && ifaceType.NumMethod() == 0 {
		var t tagAndLength
		t, offset, err = parseTagAndLength(bytes, offset)
		if err != nil {
			return
		}
		if invalidLength(offset, t.length, len(bytes)) {
			err = SyntaxError{"data truncated"}
			return
		}
		var result any
		if !t.isCompound && t.class == ClassUniversal {
			innerBytes := bytes[offset : offset+t.length]
			switch t.tag {
			case TagPrintableString:
				result, err = parsePrintableString(innerBytes)
			case TagNumericString:
				result, err = parseNumericString(innerBytes)
			case TagIA5String:
				result, err = parseIA5String(innerBytes)
			case TagT61String:
				result, err = parseT61String(innerBytes)
			case TagUTF8String:
				result, err = parseUTF8String(innerBytes)
			case TagInteger:
				result, err = parseInt64(innerBytes)
			case TagBitString:
				result, err = parseBitString(innerBytes)
			case TagOID:
				result, err = parseObjectIdentifier(innerBytes)
			case TagUTCTime:
				result, err = parseUTCTime(innerBytes)
			case TagGeneralizedTime:
				result, err = parseGeneralizedTime(innerBytes)
			case TagOctetString:
				result = innerBytes
			case TagBMPString:
				result, err = parseBMPString(innerBytes)
			default:
				// If we don't know how to handle the type, we just leave Value as nil.
			}
		}
		offset += t.length
		if err != nil {
			return
		}
		if result != nil {
			v.Set(reflect.ValueOf(result))
		}
		return
	}

	t, offset, err := parseTagAndLength(bytes, offset)
	if err != nil {
		return
	}
	if params.explicit {
		expectedClass := ClassContextSpecific
		if params.application {
			expectedClass = ClassApplication
		}
		if offset == len(bytes) {
			err = StructuralError{"explicit tag has no child"}
			return
		}
		if t.class == expectedClass && t.tag == *params.tag && (t.length == 0 || t.isCompound) {
			if fieldType == rawValueType {
				// The inner element should not be parsed for RawValues.
			} else if t.length > 0 {
				t, offset, err = parseTagAndLength(bytes, offset)
				if err != nil {
					return
				}
			} else {
				if fieldType != flagType {
					err = StructuralError{"zero length explicit tag was not an asn1.Flag"}
					return
				}
				v.SetBool(true)
				return
			}
		} else {
			// The tags didn't match, it might be an optional element.
			ok := setDefaultValue(v, params)
			if ok {
				offset = initOffset
			} else {
				err = StructuralError{"explicitly tagged member didn't match"}
			}
			return
		}
	}

	matchAny, universalTag, compoundType, ok1 := getUniversalType(fieldType)
	if !ok1 {
		err = StructuralError{fmt.Sprintf("unknown Go type: %v", fieldType)}
		return
	}

	// Special case for strings: all the ASN.1 string types map to the Go
	// type string. getUniversalType returns the tag for PrintableString
	// when it sees a string, so if we see a different string type on the
	// wire, we change the universal type to match.
	if universalTag == TagPrintableString {
		if t.class == ClassUniversal {
			switch t.tag {
			case TagIA5String, TagGeneralString, TagT61String, TagUTF8String, TagNumericString, TagBMPString:
				universalTag = t.tag
			}
		} else if params.stringType != 0 {
			universalTag = params.stringType
		}
	}

	// Special case for time: UTCTime and GeneralizedTime both map to the
	// Go type time.Time.
	if universalTag == TagUTCTime && t.tag == TagGeneralizedTime && t.class == ClassUniversal {
		universalTag = TagGeneralizedTime
	}

	if params.set {
		universalTag = TagSet
	}

	matchAnyClassAndTag := matchAny
	expectedClass := ClassUniversal
	expectedTag := universalTag

	if !params.explicit && params.tag != nil {
		expectedClass = ClassContextSpecific
		expectedTag = *params.tag
		matchAnyClassAndTag = false
	}

	if !params.explicit && params.application && params.tag != nil {
		expectedClass = ClassApplication
		expectedTag = *params.tag
		matchAnyClassAndTag = false
	}

	if !params.explicit && params.private && params.tag != nil {
		expectedClass = ClassPrivate
		expectedTag = *params.tag
		matchAnyClassAndTag = false
	}

	// We have unwrapped any explicit tagging at this point.
	if !matchAnyClassAndTag && (t.class != expectedClass || t.tag != expectedTag) ||
		(!matchAny && t.isCompound != compoundType) {
		// Tags don't match. Again, it could be an optional element.
		ok := setDefaultValue(v, params)
		if ok {
			offset = initOffset
		} else {
			err = StructuralError{fmt.Sprintf("tags don't match (%d vs %+v) %+v %s @%d", expectedTag, t, params, fieldType.Name(), offset)}
		}
		return
	}
	if invalidLength(offset, t.length, len(bytes)) {
		err = SyntaxError{"data truncated"}
		return
	}
	innerBytes := bytes[offset : offset+t.length]
	offset += t.length

	// We deal with the structures defined in this package first.
	switch v := v.Addr().Interface().(type) {
	case *RawValue:
		*v = RawValue{t.class, t.tag, t.isCompound, innerBytes, bytes[initOffset:offset]}
		return
	case *ObjectIdentifier:
		*v, err = parseObjectIdentifier(innerBytes)
		return
	case *BitString:
		*v, err = parseBitString(innerBytes)
		return
	case *time.Time:
		if universalTag == TagUTCTime {
			*v, err = parseUTCTime(innerBytes)
			return
		}
		*v, err = parseGeneralizedTime(innerBytes)
		return
	case *Enumerated:
		parsedInt, err1 := parseInt32(innerBytes)
		if err1 == nil {
			*v = Enumerated(parsedInt)
		}
		err = err1
		return
	case *Flag:
		*v = true
		return
	case **big.Int:
		parsedInt, err1 := parseBigInt(innerBytes)
		if err1 == nil {
			*v = parsedInt
		}
		err = err1
		return
	}
	switch val := v; val.Kind() {
	case reflect.Bool:
		parsedBool, err1 := parseBool(innerBytes)
		if err1 == nil {
			val.SetBool(parsedBool)
		}
		err = err1
		return
	case reflect.Int, reflect.Int32, reflect.Int64:
		if val.Type().Size() == 4 {
			parsedInt, err1 := parseInt32(innerBytes)
			if err1 == nil {
				val.SetInt(int64(parsedInt))
			}
			err = err1
		} else {
			parsedInt, err1 := parseInt64(innerBytes)
			if err1 == nil {
				val.SetInt(parsedInt)
			}
			err = err1
		}
		return
	// TODO(dfc) Add support for the remaining integer types
	case reflect.Struct:
		structType := fieldType

		for i := 0; i < structType.NumField(); i++ {
			if !structType.Field(i).IsExported() {
				err = StructuralError{"struct contains unexported fields"}
				return
			}
		}

		if structType.NumField() > 0 &&
			structType.Field(0).Type == rawContentsType {
			bytes := bytes[initOffset:offset]
			val.Field(0).Set(reflect.ValueOf(RawContent(bytes)))
		}

		innerOffset := 0
		for i := 0; i < structType.NumField(); i++ {
			field := structType.Field(i)
			if i == 0 && field.Type == rawContentsType {
				continue
			}
			innerOffset, err = parseField(val.Field(i), innerBytes, innerOffset, parseFieldParameters(field.Tag.Get("asn1")))
			if err != nil {
				return
			}
		}
		// We allow extra bytes at the end of the SEQUENCE because
		// adding elements to the end has been used in X.509 as the
		// version numbers have increased.
		return
	case reflect.Slice:
		sliceType := fieldType
		if sliceType.Elem().Kind() == reflect.Uint8 {
			val.Set(reflect.MakeSlice(sliceType, len(innerBytes), len(innerBytes)))
			reflect.Copy(val, reflect.ValueOf(innerBytes))
			return
		}
		newSlice, err1 := parseSequenceOf(innerBytes, sliceType, sliceType.Elem())
		if err1 == nil {
			val.Set(newSlice)
		}
		err = err1
		return
	case reflect.String:
		var v string
		switch universalTag {
		case TagPrintableString:
			v, err = parsePrintableString(innerBytes)
		case TagNumericString:
			v, err = parseNumericString(innerBytes)
		case TagIA5String:
			v, err = parseIA5String(innerBytes)
		case TagT61String:
			v, err = parseT61String(innerBytes)
		case TagUTF8String:
			v, err = parseUTF8String(innerBytes)
		case TagGeneralString:
			// GeneralString is specified in ISO-2022/ECMA-35,
			// A brief review suggests that it includes structures
			// that allow the encoding to change midstring and
			// such. We give up and pass it as an 8-bit string.
			v, err = parseT61String(innerBytes)
		case TagBMPString:
			v, err = parseBMPString(innerBytes)

		default:
			err = SyntaxError{fmt.Sprintf("internal error: unknown string type %d", universalTag)}
		}
		if err == nil {
			val.SetString(v)
		}
		return
	}
	err = StructuralError{"unsupported: " + v.Type().String()}
	return
}

// canHaveDefaultValue reports whether k is a Kind that we will set a default
// value for. (A signed integer, essentially.)
func canHaveDefaultValue(k reflect.Kind) bool {
	switch k {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return true
	}

	return false
}

// setDefaultValue is used to install a default value, from a tag string, into
// a Value. It is successful if the field was optional, even if a default value
// wasn't provided or it failed to install it into the Value.
func setDefaultValue(v reflect.Value, params fieldParameters) (ok bool) {
	if !params.optional {
		return
	}
	ok = true
	if params.defaultValue == nil {
		return
	}
	if canHaveDefaultValue(v.Kind()) {
		v.SetInt(*params.defaultValue)
	}
	return
}

// Unmarshal parses the DER-encoded ASN.1 data structure b
// and uses the reflect package to fill in an arbitrary value pointed at by val.
// Because Unmarshal uses the reflect package, the structs
// being written to must use upper case field names. If val
// is nil or not a pointer, Unmarshal returns an error.
//
// After parsing b, any bytes that were leftover and not used to fill
// val will be returned in rest. When parsing a SEQUENCE into a struct,
// any trailing elements of the SEQUENCE that do not have matching
// fields in val will not be included in rest, as these are considered
// valid elements of the SEQUENCE and not trailing data.
//
// An ASN.1 INTEGER can be written to an int, int32, int64,
// or *big.Int (from the math/big package).
// If the encoded value does not fit in the Go type,
// Unmarshal returns a parse error.
//
// An ASN.1 BIT STRING can be written to a BitString.
//
// An ASN.1 OCTET STRING can be written to a []byte.
//
// An ASN.1 OBJECT IDENTIFIER can be written to an
// ObjectIdentifier.
//
// An ASN.1 ENUMERATED can be written to an Enumerated.
//
// An ASN.1 UTCTIME or GENERALIZEDTIME can be written to a time.Time.
//
// An ASN.1 PrintableString, IA5String, or NumericString can be written to a string.
//
// Any of the above ASN.1 values can be written to an interface{}.
// The value stored in the interface has the corresponding Go type.
// For integers, that type is int64.
//
// An ASN.1 SEQUENCE OF x or SET OF x can be written
// to a slice if an x can be written to the slice's element type.
//
// An ASN.1 SEQUENCE or SET can be written to a struct
// if each of the elements in the sequence can be
// written to the corresponding element in the struct.
//
// The following tags on struct fields have special meaning to Unmarshal:
//
//	application specifies that an APPLICATION tag is used
//	private     specifies that a PRIVATE tag is used
//	default:x   sets the default value for optional integer fields (only used if optional is also present)
//	explicit    specifies that an additional, explicit tag wraps the implicit one
//	optional    marks the field as ASN.1 OPTIONAL
//	set         causes a SET, rather than a SEQUENCE type to be expected
//	tag:x       specifies the ASN.1 tag number; implies ASN.1 CONTEXT SPECIFIC
//
// When decoding an ASN.1 value with an IMPLICIT tag into a string field,
// Unmarshal will default to a PrintableString, which doesn't support
// characters such as '@' and '&'. To force other encodings, use the following
// tags:
//
//	ia5     causes strings to be unmarshaled as ASN.1 IA5String values
//	numeric causes strings to be unmarshaled as ASN.1 NumericString values
//	utf8    causes strings to be unmarshaled as ASN.1 UTF8String values
//
// If the type of the first field of a structure is RawContent then the raw
// ASN1 contents of the struct will be stored in it.
//
// If the name of a slice type ends with "SET" then it's treated as if
// the "set" tag was set on it. This results in interpreting the type as a
// SET OF x rather than a SEQUENCE OF x. This can be used with nested slices
// where a struct tag cannot be given.
//
// Other ASN.1 types are not supported; if it encounters them,
// Unmarshal returns a parse error.
func Unmarshal(b []byte, val any) (rest []byte, err error) {
	return UnmarshalWithParams(b, val, "")
}

// An invalidUnmarshalError describes an invalid argument passed to Unmarshal.
// (The argument to Unmarshal must be a non-nil pointer.)
type invalidUnmarshalError struct {
	Type reflect.Type
}

func (e *invalidUnmarshalError) Error() string {
	if e.Type == nil {
		return "asn1: Unmarshal recipient value is nil"
	}

	if e.Type.Kind() != reflect.Pointer {
		return "asn1: Unmarshal recipient value is non-pointer " + e.Type.String()
	}
	return "asn1: Unmarshal recipient value is nil " + e.Type.String()
}

// UnmarshalWithParams allows field parameters to be specified for the
// top-level element. The form of the params is the same as the field tags.
func UnmarshalWithParams(b []byte, val any, params string) (rest []byte, err error) {
	v := reflect.ValueOf(val)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return nil, &invalidUnmarshalError{reflect.TypeOf(val)}
	}
	offset, err := parseField(v.Elem(), b, 0, parseFieldParameters(params))
	if err != nil {
		return nil, err
	}
	return b[offset:], nil
}
