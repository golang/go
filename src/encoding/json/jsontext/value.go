// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"bytes"
	"errors"
	"io"
	"slices"
	"sync"

	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonwire"
)

// NOTE: Value is analogous to v1 json.RawMessage.

// AppendFormat formats the JSON value in src and appends it to dst
// according to the specified options.
// See [Value.Format] for more details about the formatting behavior.
//
// The dst and src may overlap.
// If an error is reported, then the entirety of src is appended to dst.
func AppendFormat(dst, src []byte, opts ...Options) ([]byte, error) {
	e := getBufferedEncoder(opts...)
	defer putBufferedEncoder(e)
	e.s.Flags.Set(jsonflags.OmitTopLevelNewline | 1)
	if err := e.s.WriteValue(src); err != nil {
		return append(dst, src...), err
	}
	return append(dst, e.s.Buf...), nil
}

// Value represents a single raw JSON value, which may be one of the following:
//   - a JSON literal (i.e., null, true, or false)
//   - a JSON string (e.g., "hello, world!")
//   - a JSON number (e.g., 123.456)
//   - an entire JSON object (e.g., {"fizz":"buzz"} )
//   - an entire JSON array (e.g., [1,2,3] )
//
// Value can represent entire array or object values, while [Token] cannot.
// Value may contain leading and/or trailing whitespace.
type Value []byte

// Clone returns a copy of v.
func (v Value) Clone() Value {
	return bytes.Clone(v)
}

// String returns the string formatting of v.
func (v Value) String() string {
	if v == nil {
		return "null"
	}
	return string(v)
}

// IsValid reports whether the raw JSON value is syntactically valid
// according to the specified options.
//
// By default (if no options are specified), it validates according to RFC 7493.
// It verifies whether the input is properly encoded as UTF-8,
// that escape sequences within strings decode to valid Unicode codepoints, and
// that all names in each object are unique.
// It does not verify whether numbers are representable within the limits
// of any common numeric type (e.g., float64, int64, or uint64).
//
// Relevant options include:
//   - [AllowDuplicateNames]
//   - [AllowInvalidUTF8]
//
// All other options are ignored.
func (v Value) IsValid(opts ...Options) bool {
	// TODO: Document support for [WithByteLimit] and [WithDepthLimit].
	d := getBufferedDecoder(v, opts...)
	defer putBufferedDecoder(d)
	_, errVal := d.ReadValue()
	_, errEOF := d.ReadToken()
	return errVal == nil && errEOF == io.EOF
}

// Format formats the raw JSON value in place.
//
// By default (if no options are specified), it validates according to RFC 7493
// and produces the minimal JSON representation, where
// all whitespace is elided and JSON strings use the shortest encoding.
//
// Relevant options include:
//   - [AllowDuplicateNames]
//   - [AllowInvalidUTF8]
//   - [EscapeForHTML]
//   - [EscapeForJS]
//   - [PreserveRawStrings]
//   - [CanonicalizeRawInts]
//   - [CanonicalizeRawFloats]
//   - [ReorderRawObjects]
//   - [SpaceAfterColon]
//   - [SpaceAfterComma]
//   - [Multiline]
//   - [WithIndent]
//   - [WithIndentPrefix]
//
// All other options are ignored.
//
// It is guaranteed to succeed if the value is valid according to the same options.
// If the value is already formatted, then the buffer is not mutated.
func (v *Value) Format(opts ...Options) error {
	// TODO: Document support for [WithByteLimit] and [WithDepthLimit].
	return v.format(opts, nil)
}

// format accepts two []Options to avoid the allocation appending them together.
// It is equivalent to v.Format(append(opts1, opts2...)...).
func (v *Value) format(opts1, opts2 []Options) error {
	e := getBufferedEncoder(opts1...)
	defer putBufferedEncoder(e)
	e.s.Join(opts2...)
	e.s.Flags.Set(jsonflags.OmitTopLevelNewline | 1)
	if err := e.s.WriteValue(*v); err != nil {
		return err
	}
	if !bytes.Equal(*v, e.s.Buf) {
		*v = append((*v)[:0], e.s.Buf...)
	}
	return nil
}

// Compact removes all whitespace from the raw JSON value.
//
// It does not reformat JSON strings or numbers to use any other representation.
// To maximize the set of JSON values that can be formatted,
// this permits values with duplicate names and invalid UTF-8.
//
// Compact is equivalent to calling [Value.Format] with the following options:
//   - [AllowDuplicateNames](true)
//   - [AllowInvalidUTF8](true)
//   - [PreserveRawStrings](true)
//
// Any options specified by the caller are applied after the initial set
// and may deliberately override prior options.
func (v *Value) Compact(opts ...Options) error {
	return v.format([]Options{
		AllowDuplicateNames(true),
		AllowInvalidUTF8(true),
		PreserveRawStrings(true),
	}, opts)
}

// Indent reformats the whitespace in the raw JSON value so that each element
// in a JSON object or array begins on a indented line according to the nesting.
//
// It does not reformat JSON strings or numbers to use any other representation.
// To maximize the set of JSON values that can be formatted,
// this permits values with duplicate names and invalid UTF-8.
//
// Indent is equivalent to calling [Value.Format] with the following options:
//   - [AllowDuplicateNames](true)
//   - [AllowInvalidUTF8](true)
//   - [PreserveRawStrings](true)
//   - [Multiline](true)
//
// Any options specified by the caller are applied after the initial set
// and may deliberately override prior options.
func (v *Value) Indent(opts ...Options) error {
	return v.format([]Options{
		AllowDuplicateNames(true),
		AllowInvalidUTF8(true),
		PreserveRawStrings(true),
		Multiline(true),
	}, opts)
}

// Canonicalize canonicalizes the raw JSON value according to the
// JSON Canonicalization Scheme (JCS) as defined by RFC 8785
// where it produces a stable representation of a JSON value.
//
// JSON strings are formatted to use their minimal representation,
// JSON numbers are formatted as double precision numbers according
// to some stable serialization algorithm.
// JSON object members are sorted in ascending order by name.
// All whitespace is removed.
//
// The output stability is dependent on the stability of the application data
// (see RFC 8785, Appendix E). It cannot produce stable output from
// fundamentally unstable input. For example, if the JSON value
// contains ephemeral data (e.g., a frequently changing timestamp),
// then the value is still unstable regardless of whether this is called.
//
// Canonicalize is equivalent to calling [Value.Format] with the following options:
//   - [CanonicalizeRawInts](true)
//   - [CanonicalizeRawFloats](true)
//   - [ReorderRawObjects](true)
//
// Any options specified by the caller are applied after the initial set
// and may deliberately override prior options.
//
// Note that JCS treats all JSON numbers as IEEE 754 double precision numbers.
// Any numbers with precision beyond what is representable by that form
// will lose their precision when canonicalized. For example, integer values
// beyond ±2⁵³ will lose their precision. To preserve the original representation
// of JSON integers, additionally set [CanonicalizeRawInts] to false:
//
//	v.Canonicalize(jsontext.CanonicalizeRawInts(false))
func (v *Value) Canonicalize(opts ...Options) error {
	return v.format([]Options{
		CanonicalizeRawInts(true),
		CanonicalizeRawFloats(true),
		ReorderRawObjects(true),
	}, opts)
}

// MarshalJSON returns v as the JSON encoding of v.
// It returns the stored value as the raw JSON output without any validation.
// If v is nil, then this returns a JSON null.
func (v Value) MarshalJSON() ([]byte, error) {
	// NOTE: This matches the behavior of v1 json.RawMessage.MarshalJSON.
	if v == nil {
		return []byte("null"), nil
	}
	return v, nil
}

// UnmarshalJSON sets v as the JSON encoding of b.
// It stores a copy of the provided raw JSON input without any validation.
func (v *Value) UnmarshalJSON(b []byte) error {
	// NOTE: This matches the behavior of v1 json.RawMessage.UnmarshalJSON.
	if v == nil {
		return errors.New("jsontext.Value: UnmarshalJSON on nil pointer")
	}
	*v = append((*v)[:0], b...)
	return nil
}

// Kind returns the starting token kind.
// For a valid value, this will never include [KindEndObject] or [KindEndArray].
func (v Value) Kind() Kind {
	if v := v[jsonwire.ConsumeWhitespace(v):]; len(v) > 0 {
		return Kind(v[0]).normalize()
	}
	return invalidKind
}

const commaAndWhitespace = ", \n\r\t"

type objectMember struct {
	// name is the unquoted name.
	name []byte // e.g., "name"
	// buffer is the entirety of the raw JSON object member
	// starting from right after the previous member (or opening '{')
	// until right after the member value.
	buffer []byte // e.g., `, \n\r\t"name": "value"`
}

func (x objectMember) Compare(y objectMember) int {
	if c := jsonwire.CompareUTF16(x.name, y.name); c != 0 {
		return c
	}
	// With [AllowDuplicateNames] or [AllowInvalidUTF8],
	// names could be identical, so also sort using the member value.
	return jsonwire.CompareUTF16(
		bytes.TrimLeft(x.buffer, commaAndWhitespace),
		bytes.TrimLeft(y.buffer, commaAndWhitespace))
}

var objectMemberPool = sync.Pool{New: func() any { return new([]objectMember) }}

func getObjectMembers() *[]objectMember {
	ns := objectMemberPool.Get().(*[]objectMember)
	*ns = (*ns)[:0]
	return ns
}
func putObjectMembers(ns *[]objectMember) {
	if cap(*ns) < 1<<10 {
		clear(*ns) // avoid pinning name and buffer
		objectMemberPool.Put(ns)
	}
}

// mustReorderObjects reorders in-place all object members in a JSON value,
// which must be valid otherwise it panics.
func mustReorderObjects(b []byte) {
	// Obtain a buffered encoder just to use its internal buffer as
	// a scratch buffer for reordering object members.
	e2 := getBufferedEncoder()
	defer putBufferedEncoder(e2)

	// Disable unnecessary checks to syntactically parse the JSON value.
	d := getBufferedDecoder(b)
	defer putBufferedDecoder(d)
	d.s.Flags.Set(jsonflags.AllowDuplicateNames | jsonflags.AllowInvalidUTF8 | 1)
	mustReorderObjectsFromDecoder(d, &e2.s.Buf) // per RFC 8785, section 3.2.3
}

// mustReorderObjectsFromDecoder recursively reorders all object members in place
// according to the ordering specified in RFC 8785, section 3.2.3.
//
// Pre-conditions:
//   - The value is valid (i.e., no decoder errors should ever occur).
//   - Initial call is provided a Decoder reading from the start of v.
//
// Post-conditions:
//   - Exactly one JSON value is read from the Decoder.
//   - All fully-parsed JSON objects are reordered by directly moving
//     the members in the value buffer.
//
// The runtime is approximately O(n·log(n)) + O(m·log(m)),
// where n is len(v) and m is the total number of object members.
func mustReorderObjectsFromDecoder(d *Decoder, scratch *[]byte) {
	switch tok, err := d.ReadToken(); tok.Kind() {
	case '{':
		// Iterate and collect the name and offsets for every object member.
		members := getObjectMembers()
		defer putObjectMembers(members)
		var prevMember objectMember
		isSorted := true

		beforeBody := d.InputOffset() // offset after '{'
		for d.PeekKind() != '}' {
			beforeName := d.InputOffset()
			var flags jsonwire.ValueFlags
			name, _ := d.s.ReadValue(&flags)
			name = jsonwire.UnquoteMayCopy(name, flags.IsVerbatim())
			mustReorderObjectsFromDecoder(d, scratch)
			afterValue := d.InputOffset()

			currMember := objectMember{name, d.s.buf[beforeName:afterValue]}
			if isSorted && len(*members) > 0 {
				isSorted = objectMember.Compare(prevMember, currMember) < 0
			}
			*members = append(*members, currMember)
			prevMember = currMember
		}
		afterBody := d.InputOffset() // offset before '}'
		d.ReadToken()

		// Sort the members; return early if it's already sorted.
		if isSorted {
			return
		}
		firstBufferBeforeSorting := (*members)[0].buffer
		slices.SortFunc(*members, objectMember.Compare)
		firstBufferAfterSorting := (*members)[0].buffer

		// Append the reordered members to a new buffer,
		// then copy the reordered members back over the original members.
		// Avoid swapping in place since each member may be a different size
		// where moving a member over a smaller member may corrupt the data
		// for subsequent members before they have been moved.
		//
		// The following invariant must hold:
		//	sum([m.after-m.before for m in members]) == afterBody-beforeBody
		commaAndWhitespacePrefix := func(b []byte) []byte {
			return b[:len(b)-len(bytes.TrimLeft(b, commaAndWhitespace))]
		}
		sorted := (*scratch)[:0]
		for i, member := range *members {
			switch {
			case i == 0 && &member.buffer[0] != &firstBufferBeforeSorting[0]:
				// First member after sorting is not the first member before sorting,
				// so use the prefix of the first member before sorting.
				sorted = append(sorted, commaAndWhitespacePrefix(firstBufferBeforeSorting)...)
				sorted = append(sorted, bytes.TrimLeft(member.buffer, commaAndWhitespace)...)
			case i != 0 && &member.buffer[0] == &firstBufferBeforeSorting[0]:
				// Later member after sorting is the first member before sorting,
				// so use the prefix of the first member after sorting.
				sorted = append(sorted, commaAndWhitespacePrefix(firstBufferAfterSorting)...)
				sorted = append(sorted, bytes.TrimLeft(member.buffer, commaAndWhitespace)...)
			default:
				sorted = append(sorted, member.buffer...)
			}
		}
		if int(afterBody-beforeBody) != len(sorted) {
			panic("BUG: length invariant violated")
		}
		copy(d.s.buf[beforeBody:afterBody], sorted)

		// Update scratch buffer to the largest amount ever used.
		if len(sorted) > len(*scratch) {
			*scratch = sorted
		}
	case '[':
		for d.PeekKind() != ']' {
			mustReorderObjectsFromDecoder(d, scratch)
		}
		d.ReadToken()
	default:
		if err != nil {
			panic("BUG: " + err.Error())
		}
	}
}
