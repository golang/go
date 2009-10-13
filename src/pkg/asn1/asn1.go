// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements parsing of DER-encoded ASN.1 data structures,
// as defined in ITU-T Rec. X.690.
//
// See also ``A Layman's Guide to a Subset of ASN.1, BER, and DER,''
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
	"fmt";
	"os";
	"reflect";
	"strconv";
	"strings";
	"time";
)

// A StructuralError suggests that the ASN.1 data is valid, but the Go type
// which is receiving it doesn't match.
type StructuralError struct {
	Msg	string;
}

func (e StructuralError) String() string {
	return "ASN.1 structure error: " + e.Msg;
}

// A SyntaxError suggests that the ASN.1 data is invalid.
type SyntaxError struct {
	Msg	string;
}

func (e SyntaxError) String() string {
	return "ASN.1 syntax error: " + e.Msg;
}

// We start by dealing with each of the primitive types in turn.

// BOOLEAN

func parseBool(bytes []byte) (ret bool, err os.Error) {
	if len(bytes) != 1 {
		err = SyntaxError{"invalid boolean"};
		return;
	}

	return bytes[0] != 0, nil;
}

// INTEGER

// parseInt64 treats the given bytes as a big-endian, signed integer and
// returns the result.
func parseInt64(bytes []byte) (ret int64, err os.Error) {
	if len(bytes) > 8 {
		// We'll overflow an int64 in this case.
		err = StructuralError{"integer too large"};
		return;
	}
	for bytesRead := 0; bytesRead < len(bytes); bytesRead++ {
		ret <<= 8;
		ret |= int64(bytes[bytesRead]);
	}

	// Shift up and down in order to sign extend the result.
	ret <<= 64 - uint8(len(bytes))*8;
	ret >>= 64 - uint8(len(bytes))*8;
	return;
}

// parseInt treats the given bytes as a big-endian, signed integer and returns
// the result.
func parseInt(bytes []byte) (int, os.Error) {
	ret64, err := parseInt64(bytes);
	if err != nil {
		return 0, err;
	}
	if ret64 != int64(int(ret64)) {
		return 0, StructuralError{"integer too large"};
	}
	return int(ret64), nil;
}

// BIT STRING

// BitString is the structure to use when you want an ASN.1 BIT STRING type. A
// bit string is padded up to the nearest byte in memory and the number of
// valid bits is recorded. Padding bits will be zero.
type BitString struct {
	Bytes		[]byte;	// bits packed into bytes.
	BitLength	int;	// length in bits.
}

// At returns the bit at the given index. If the index is out of range it
// returns false.
func (b BitString) At(i int) int {
	if i < 0 || i >= b.BitLength {
		return 0;
	}
	x := i / 8;
	y := 7 - uint(i % 8);
	return int(b.Bytes[x] >> y) & 1;
}

// parseBitString parses an ASN.1 bit string from the given byte array and returns it.
func parseBitString(bytes []byte) (ret BitString, err os.Error) {
	if len(bytes) == 0 {
		err = SyntaxError{"zero length BIT STRING"};
		return;
	}
	paddingBits := int(bytes[0]);
	if paddingBits > 7 ||
	   len(bytes) == 1 && paddingBits > 0 ||
	   bytes[len(bytes)-1] & ((1 << bytes[0])-1) != 0 {
		err = SyntaxError{"invalid padding bits in BIT STRING"};
		return;
	}
	ret.BitLength = (len(bytes)-1)*8 - paddingBits;
	ret.Bytes = bytes[1:len(bytes)];
	return;
}

// OBJECT IDENTIFIER

// An ObjectIdentifier represents an ASN.1 OBJECT IDENTIFIER.
type ObjectIdentifier []int

// parseObjectIdentifier parses an OBJECT IDENTIFER from the given bytes and
// returns it. An object identifer is a sequence of variable length integers
// that are assigned in a hierarachy.
func parseObjectIdentifier(bytes []byte) (s []int, err os.Error) {
	if len(bytes) == 0 {
		err = SyntaxError{"zero length OBJECT IDENTIFIER"};
		return;
	}

	// In the worst case, we get two elements from the first byte (which is
	// encoded differently) and then every varint is a single byte long.
	s = make([]int, len(bytes)+1);

	// The first byte is 40*value1 + value2:
	s[0] = int(bytes[0]) / 40;
	s[1] = int(bytes[0]) % 40;
	i := 2;
	for offset := 1; offset < len(bytes); i++ {
		var v int;
		v, offset, err = parseBase128Int(bytes, offset);
		if err != nil {
			return;
		}
		s[i] = v;
	}
	s = s[0:i];
	return;
}

// parseBase128Int parses a base-128 encoded int from the given offset in the
// given byte array. It returns the value and the new offset.
func parseBase128Int(bytes []byte, initOffset int) (ret, offset int, err os.Error) {
	offset = initOffset;
	for shifted := 0; offset < len(bytes); shifted++ {
		if shifted > 4 {
			err = StructuralError{"base 128 integer too large"};
			return;
		}
		ret <<= 7;
		b := bytes[offset];
		ret |= int(b&0x7f);
		offset++;
		if b&0x80 == 0 {
			return;
		}
	}
	err = SyntaxError{"truncated base 128 integer"};
	return;
}

// UTCTime

func isDigit(b byte) bool {
	return '0' <= b && b <= '9';
}

// twoDigits returns the value of two, base 10 digits.
func twoDigits(bytes []byte, max int) (int, bool) {
	for i := 0; i < 2; i++ {
		if !isDigit(bytes[i]) {
			return 0, false;
		}
	}
	value := (int(bytes[0]) - '0')*10 + int(bytes[1] - '0');
	if value > max {
		return 0, false;
	}
	return value, true;
}

// parseUTCTime parses the UTCTime from the given byte array and returns the
// resulting time.
func parseUTCTime(bytes []byte) (ret time.Time, err os.Error) {
	// A UTCTime can take the following formats:
	//
	//             1111111
	//   01234567890123456
	//
	//   YYMMDDhhmmZ
	//   YYMMDDhhmm+hhmm
	//   YYMMDDhhmm-hhmm
	//   YYMMDDhhmmssZ
	//   YYMMDDhhmmss+hhmm
	//   YYMMDDhhmmss-hhmm
	if len(bytes) < 11 {
		err = SyntaxError{"UTCTime too short"};
		return;
	}
	var ok1, ok2, ok3, ok4, ok5 bool;
	year, ok1 := twoDigits(bytes[0:2], 99);
	// RFC 5280, section 5.1.2.4 says that years 2050 or later use another date
	// scheme.
	if year > 50 {
		ret.Year = 1900+int64(year);
	} else {
		ret.Year = 2000+int64(year);
	}
	ret.Month, ok2 = twoDigits(bytes[2:4], 12);
	ret.Day, ok3 = twoDigits(bytes[4:6], 31);
	ret.Hour, ok4 = twoDigits(bytes[6:8], 23);
	ret.Minute, ok5 = twoDigits(bytes[8:10], 59);
	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 {
		goto Error;
	}
	bytes = bytes[10:len(bytes)];
	switch bytes[0] {
	case '0', '1', '2', '3', '4', '5', '6':
		if len(bytes) < 3 {
			goto Error;
		}
		ret.Second, ok1 = twoDigits(bytes[0:2], 60);	// 60, not 59, because of leap seconds.
		if !ok1 {
			goto Error;
		}
		bytes = bytes[2:len(bytes)];
	}
	if len(bytes) == 0 {
		goto Error;
	}
	switch bytes[0] {
	case 'Z':
		if len(bytes) != 1 {
			goto Error;
		}
		return;
	case '-', '+':
		if len(bytes) != 5 {
			goto Error;
		}
		hours, ok1 := twoDigits(bytes[1:3], 12);
		minutes, ok2 := twoDigits(bytes[3:5], 59);
		if !ok1 || !ok2 {
			goto Error;
		}
		sign := 1;
		if bytes[0] == '-' {
			sign = -1;
		}
		ret.ZoneOffset = sign*(60*(hours*60 + minutes));
	default:
		goto Error;
	}
	return;

Error:
	err = SyntaxError{"invalid UTCTime"};
	return;
}

// PrintableString

// parsePrintableString parses a ASN.1 PrintableString from the given byte
// array and returns it.
func parsePrintableString(bytes []byte) (ret string, err os.Error) {
	for _, b := range bytes {
		if !isPrintable(b) {
			err = SyntaxError{"PrintableString contains invalid character"};
			return;
		}
	}
	ret = string(bytes);
	return;
}

// isPrintable returns true iff the given b is in the ASN.1 PrintableString set.
func isPrintable(b byte) bool {
	return 'a' <= b && b <= 'z' ||
	       'A' <= b && b <= 'Z' ||
	       '0' <= b && b <= '9' ||
	       '\'' <= b && b <= ')' ||
	       '+' <= b && b <= '/' ||
	       b == ' ' ||
	       b == ':' ||
	       b == '=' ||
	       b == '?';
}

// IA5String

// parseIA5String parses a ASN.1 IA5String (ASCII string) from the given
// byte array and returns it.
func parseIA5String(bytes []byte) (ret string, err os.Error) {
	for _, b := range bytes {
		if b >= 0x80 {
			err = SyntaxError{"IA5String contains invalid character"};
			return;
		}
	}
	ret = string(bytes);
	return;
}

// A RawValue represents an undecoded ASN.1 object.
type RawValue struct {
	Class, Tag	int;
	IsCompound	bool;
	Bytes		[]byte;
}

// Tagging

// ASN.1 objects have metadata preceeding them:
//   the tag: the type of the object
//   a flag denoting if this object is compound or not
//   the class type: the namespace of the tag
//   the length of the object, in bytes

// Here are some standard tags and classes

const (
	tagBoolean		= 1;
	tagInteger		= 2;
	tagBitString		= 3;
	tagOctetString		= 4;
	tagOID			= 6;
	tagSequence		= 16;
	tagSet			= 17;
	tagPrintableString	= 19;
	tagIA5String		= 22;
	tagUTCTime		= 23;
)

const (
	classUniversal		= 0;
	classApplication	= 1;
	classContextSpecific	= 2;
	classPrivate		= 3;
)

type tagAndLength struct {
	class, tag, length	int;
	isCompound		bool;
}

// parseTagAndLength parses an ASN.1 tag and length pair from the given offset
// into a byte array. It returns the parsed data and the new offset. SET and
// SET OF (tag 17) are mapped to SEQUENCE and SEQUENCE OF (tag 16) since we
// don't distinguish between ordered and unordered objects in this code.
func parseTagAndLength(bytes []byte, initOffset int) (ret tagAndLength, offset int, err os.Error) {
	offset = initOffset;
	b := bytes[offset];
	offset++;
	ret.class = int(b>>6);
	ret.isCompound = b&0x20 == 0x20;
	ret.tag = int(b&0x1f);

	// If the bottom five bits are set, then the tag number is actually base 128
	// encoded afterwards
	if ret.tag == 0x1f {
		ret.tag, offset, err = parseBase128Int(bytes, offset);
		if err != nil {
			return;
		}
	}
	if offset >= len(bytes) {
		err = SyntaxError{"truncated tag or length"};
		return;
	}
	b = bytes[offset];
	offset++;
	if b&0x80 == 0 {
		// The length is encoded in the bottom 7 bits.
		ret.length = int(b&0x7f);
	} else {
		// Bottom 7 bits give the number of length bytes to follow.
		numBytes := int(b&0x7f);
		// We risk overflowing a signed 32-bit number if we accept more than 3 bytes.
		if numBytes > 3 {
			err = StructuralError{"length too large"};
			return;
		}
		if numBytes == 0 {
			err = SyntaxError{"indefinite length found (not DER)"};
			return;
		}
		ret.length = 0;
		for i := 0; i < numBytes; i++ {
			if offset >= len(bytes) {
				err = SyntaxError{"truncated tag or length"};
				return;
			}
			b = bytes[offset];
			offset++;
			ret.length <<= 8;
			ret.length |= int(b);
		}
	}

	// We magically map SET and SET OF to SEQUENCE and SEQUENCE OF
	// because we treat everything as ordered.
	if ret.tag == tagSet {
		ret.tag = tagSequence;
	}
	return;
}

// ASN.1 has IMPLICIT and EXPLICIT tags, which can be translated as "instead
// of" and "in addition to". When not specified, every primitive type has a
// default tag in the UNIVERSAL class.
//
// For example: a BIT STRING is tagged [UNIVERSAL 3] by default (although ASN.1
// doesn't actually have a UNIVERSAL keyword). However, by saying [IMPLICIT
// CONTEXT-SPECIFIC 42], that means that the tag is replaced by another.
//
// On the other hand, if it said [EXPLICIT CONTEXT-SPECIFIC 10], then an
// /additional/ tag would wrap the default tag. This explicit tag will have the
// compound flag set.
//
// (This is used in order to remove ambiguity with optional elements.)
//
// You can layer EXPLICIT and IMPLICIT tags to an arbitary depth, however we
// don't support that here. We support a single layer of EXPLICIT or IMPLICIT
// tagging with tag strings on the fields of a structure.

// fieldParameters is the parsed representation of tag string from a structure field.
type fieldParameters struct {
	optional	bool;	// true iff the field is OPTIONAL
	explicit	bool;	// true iff and EXPLICIT tag is in use.
	defaultValue	*int64;	// a default value for INTEGER typed fields (maybe nil).
	tag		*int;	// the EXPLICIT or IMPLICIT tag (maybe nil).

// Invariants:
//   if explicit is set, tag is non-nil.
}

// Given a tag string with the format specified in the package comment,
// parseFieldParameters will parse it into a fieldParameters structure,
// ignoring unknown parts of the string.
func parseFieldParameters(str string) (ret fieldParameters) {
	for _, part := range strings.Split(str, ",", 0) {
		switch {
		case part == "optional":
			ret.optional = true;
		case part == "explicit":
			ret.explicit = true;
			if ret.tag == nil {
				ret.tag = new(int);
				*ret.tag = 0;
			}
		case strings.HasPrefix(part, "default:"):
			i, err := strconv.Atoi64(part[8:len(part)]);
			if err == nil {
				ret.defaultValue = new(int64);
				*ret.defaultValue = i;
			}
		case strings.HasPrefix(part, "tag:"):
			i, err := strconv.Atoi(part[4:len(part)]);
			if err == nil {
				ret.tag = new(int);
				*ret.tag = i;
			}
		}
	}
	return;
}

// Given a reflected Go type, getUniversalType returns the default tag number
// and expected compound flag.
func getUniversalType(t reflect.Type) (tagNumber int, isCompound, ok bool) {
	switch t {
	case objectIdentifierType:
		return tagOID, false, true;
	case bitStringType:
		return tagBitString, false, true;
	case timeType:
		return tagUTCTime, false, true;
	}
	switch i := t.(type) {
	case *reflect.BoolType:
		return tagBoolean, false, true;
	case *reflect.IntType:
		return tagInteger, false, true;
	case *reflect.Int64Type:
		return tagInteger, false, true;
	case *reflect.StructType:
		return tagSequence, true, true;
	case *reflect.SliceType:
		if _, ok := t.(*reflect.SliceType).Elem().(*reflect.Uint8Type); ok {
			return tagOctetString, false, true;
		}
		return tagSequence, true, true;
	case *reflect.StringType:
		return tagPrintableString, false, true;
	}
	return 0, false, false;
}

// parseSequenceOf is used for SEQUENCE OF and SET OF values. It tries to parse
// a number of ASN.1 values from the given byte array and returns them as a
// slice of Go values of the given type.
func parseSequenceOf(bytes []byte, sliceType *reflect.SliceType, elemType reflect.Type) (ret *reflect.SliceValue, err os.Error) {
	expectedTag, compoundType, ok := getUniversalType(elemType);
	if !ok {
		err = StructuralError{"unknown Go type for slice"};
		return;
	}

	// First we iterate over the input and count the number of elements,
	// checking that the types are correct in each case.
	numElements := 0;
	for offset := 0; offset < len(bytes); {
		var t tagAndLength;
		t, offset, err = parseTagAndLength(bytes, offset);
		if err != nil {
			return;
		}
		if t.class != classUniversal || t.isCompound != compoundType || t.tag != expectedTag {
			err = StructuralError{"sequence tag mismatch"};
			return;
		}
		if invalidLength(offset, t.length, len(bytes)) {
			err = SyntaxError{"truncated sequence"};
			return;
		}
		offset += t.length;
		numElements++;
	}
	ret = reflect.MakeSlice(sliceType, numElements, numElements);
	params := fieldParameters{};
	offset := 0;
	for i := 0; i < numElements; i++ {
		offset, err = parseField(ret.Elem(i), bytes, offset, params);
		if err != nil {
			return;
		}
	}
	return;
}

var (
	bitStringType		= reflect.Typeof(BitString{});
	objectIdentifierType	= reflect.Typeof(ObjectIdentifier{});
	timeType		= reflect.Typeof(time.Time{});
	rawValueType		= reflect.Typeof(RawValue{});
)

// invalidLength returns true iff offset + length > sliceLength, or if the
// addition would overflow.
func invalidLength(offset, length, sliceLength int) bool {
	return offset+length < offset || offset+length > sliceLength;
}

// parseField is the main parsing function. Given a byte array and an offset
// into the array, it will try to parse a suitable ASN.1 value out and store it
// in the given Value.
func parseField(v reflect.Value, bytes []byte, initOffset int, params fieldParameters) (offset int, err os.Error) {
	offset = initOffset;
	fieldType := v.Type();

	// If we have run out of data, it may be that there are optional elements at the end.
	if offset == len(bytes) {
		if !setDefaultValue(v, params) {
			err = SyntaxError{"sequence truncated"};
		}
		return;
	}

	// Deal with raw values.
	if fieldType == rawValueType {
		var t tagAndLength;
		t, offset, err = parseTagAndLength(bytes, offset);
		if err != nil {
			return;
		}
		if invalidLength(offset, t.length, len(bytes)) {
			err = SyntaxError{"data truncated"};
			return;
		}
		result := RawValue{t.class, t.tag, t.isCompound, bytes[offset : offset + t.length]};
		offset += t.length;
		v.(*reflect.StructValue).Set(reflect.NewValue(result).(*reflect.StructValue));
		return;
	}

	// Deal with the ANY type.
	if ifaceType, ok := fieldType.(*reflect.InterfaceType); ok && ifaceType.NumMethod() == 0 {
		ifaceValue := v.(*reflect.InterfaceValue);
		var t tagAndLength;
		t, offset, err = parseTagAndLength(bytes, offset);
		if err != nil {
			return;
		}
		if invalidLength(offset, t.length, len(bytes)) {
			err = SyntaxError{"data truncated"};
			return;
		}
		var result interface{}
		if !t.isCompound && t.class == classUniversal {
			innerBytes := bytes[offset : offset + t.length];
			switch t.tag {
			case tagPrintableString:
				result, err = parsePrintableString(innerBytes);
			case tagIA5String:
				result, err = parseIA5String(innerBytes);
			case tagInteger:
				result, err = parseInt64(innerBytes);
			case tagBitString:
				result, err = parseBitString(innerBytes);
			case tagOID:
				result, err = parseObjectIdentifier(innerBytes);
			case tagUTCTime:
				result, err = parseUTCTime(innerBytes);
			case tagOctetString:
				result = innerBytes;
			default:
			// If we don't know how to handle the type, we just leave Value as nil.
			}
		}
		offset += t.length;
		if err != nil {
			return;
		}
		if result != nil {
			ifaceValue.Set(reflect.NewValue(result));
		}
		return;
	}
	universalTag, compoundType, ok1 := getUniversalType(fieldType);
	if !ok1 {
		err = StructuralError{fmt.Sprintf("unknown Go type: %v", fieldType)};
		return;
	}

	t, offset, err := parseTagAndLength(bytes, offset);
	if err != nil {
		return;
	}
	if params.explicit {
		if t.class == classContextSpecific && t.tag == *params.tag && t.isCompound {
			t, offset, err = parseTagAndLength(bytes, offset);
			if err != nil {
				return;
			}
		} else {
			// The tags didn't match, it might be an optional element.
			ok := setDefaultValue(v, params);
			if ok {
				offset = initOffset;
			} else {
				err = StructuralError{"explicitly tagged member didn't match"};
			}
			return;
		}
	}

	// Special case for strings: PrintableString and IA5String both map to
	// the Go type string. getUniversalType returns the tag for
	// PrintableString when it sees a string so, if we see an IA5String on
	// the wire, we change the universal type to match.
	if universalTag == tagPrintableString && t.tag == tagIA5String {
		universalTag = tagIA5String;
	}

	expectedClass := classUniversal;
	expectedTag := universalTag;

	if !params.explicit && params.tag != nil {
		expectedClass = classContextSpecific;
		expectedTag = *params.tag;
	}

	// We have unwrapped any explicit tagging at this point.
	if t.class != expectedClass || t.tag != expectedTag || t.isCompound != compoundType {
		// Tags don't match. Again, it could be an optional element.
		ok := setDefaultValue(v, params);
		if ok {
			offset = initOffset;
		} else {
			err = StructuralError{fmt.Sprintf("tags don't match (%d vs %+v) %+v %s %#v", expectedTag, t, params, fieldType.Name(), bytes[offset:len(bytes)])};
		}
		return;
	}
	if invalidLength(offset, t.length, len(bytes)) {
		err = SyntaxError{"data truncated"};
		return;
	}
	innerBytes := bytes[offset : offset + t.length];

	// We deal with the structures defined in this package first.
	switch fieldType {
	case objectIdentifierType:
		newSlice, err1 := parseObjectIdentifier(innerBytes);
		sliceValue := v.(*reflect.SliceValue);
		sliceValue.Set(reflect.MakeSlice(sliceValue.Type().(*reflect.SliceType), len(newSlice), len(newSlice)));
		if err1 == nil {
			reflect.ArrayCopy(sliceValue, reflect.NewValue(newSlice).(reflect.ArrayOrSliceValue));
		}
		offset += t.length;
		err = err1;
		return;
	case bitStringType:
		structValue := v.(*reflect.StructValue);
		bs, err1 := parseBitString(innerBytes);
		offset += t.length;
		if err1 == nil {
			structValue.Set(reflect.NewValue(bs).(*reflect.StructValue));
		}
		err = err1;
		return;
	case timeType:
		structValue := v.(*reflect.StructValue);
		time, err1 := parseUTCTime(innerBytes);
		offset += t.length;
		if err1 == nil {
			structValue.Set(reflect.NewValue(time).(*reflect.StructValue));
		}
		err = err1;
		return;
	}
	switch val := v.(type) {
	case *reflect.BoolValue:
		parsedBool, err1 := parseBool(innerBytes);
		offset += t.length;
		if err1 == nil {
			val.Set(parsedBool);
		}
		err = err1;
		return;
	case *reflect.IntValue:
		parsedInt, err1 := parseInt(innerBytes);
		offset += t.length;
		if err1 == nil {
			val.Set(parsedInt);
		}
		err = err1;
		return;
	case *reflect.Int64Value:
		parsedInt, err1 := parseInt64(innerBytes);
		offset += t.length;
		if err1 == nil {
			val.Set(parsedInt);
		}
		err = err1;
		return;
	case *reflect.StructValue:
		structType := fieldType.(*reflect.StructType);
		innerOffset := 0;
		for i := 0; i < structType.NumField(); i++ {
			field := structType.Field(i);
			innerOffset, err = parseField(val.Field(i), innerBytes, innerOffset, parseFieldParameters(field.Tag));
			if err != nil {
				return;
			}
		}
		offset += t.length;
		// We allow extra bytes at the end of the SEQUENCE because
		// adding elements to the end has been used in X.509 as the
		// version numbers have increased.
		return;
	case *reflect.SliceValue:
		sliceType := fieldType.(*reflect.SliceType);
		if _, ok := sliceType.Elem().(*reflect.Uint8Type); ok {
			val.Set(reflect.MakeSlice(sliceType, len(innerBytes), len(innerBytes)));
			reflect.ArrayCopy(val, reflect.NewValue(innerBytes).(reflect.ArrayOrSliceValue));
			return;
		}
		newSlice, err1 := parseSequenceOf(innerBytes, sliceType, sliceType.Elem());
		offset += t.length;
		if err1 == nil {
			val.Set(newSlice);
		}
		err = err1;
		return;
	case *reflect.StringValue:
		var v string;
		switch universalTag {
		case tagPrintableString:
			v, err = parsePrintableString(innerBytes);
		case tagIA5String:
			v, err = parseIA5String(innerBytes);
		default:
			err = SyntaxError{fmt.Sprintf("internal error: unknown string type %d", universalTag)};
		}
		if err == nil {
			val.Set(v);
		}
		return;
	}
	err = StructuralError{"unknown Go type"};
	return;
}

// setDefaultValue is used to install a default value, from a tag string, into
// a Value. It is successful is the field was optional, even if a default value
// wasn't provided or it failed to install it into the Value.
func setDefaultValue(v reflect.Value, params fieldParameters) (ok bool) {
	if !params.optional {
		return;
	}
	ok = true;
	if params.defaultValue == nil {
		return;
	}
	switch val := v.(type) {
	case *reflect.IntValue:
		val.Set(int(*params.defaultValue));
	case *reflect.Int64Value:
		val.Set(int64(*params.defaultValue));
	}
	return;
}

// Unmarshal parses the DER-encoded ASN.1 data structure b
// and uses the reflect package to fill in an arbitrary value pointed at by val.
// Because Unmarshal uses the reflect package, the structs
// being written to must use upper case field names.
//
// An ASN.1 INTEGER can be written to an int or int64.
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
// An ASN.1 PrintableString or IA5String can be written to a string.
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
//	optional		marks the field as ASN.1 OPTIONAL
//	[explicit] tag:x	specifies the ASN.1 tag number; implies ASN.1 CONTEXT SPECIFIC
//	default:x		sets the default value for optional integer fields
//
// Other ASN.1 types are not supported; if it encounters them,
// Unmarshal returns a parse error.
func Unmarshal(val interface{}, b []byte) os.Error {
	v := reflect.NewValue(val).(*reflect.PtrValue).Elem();
	_, err := parseField(v, b, 0, fieldParameters{});
	return err;
}
