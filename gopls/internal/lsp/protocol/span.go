// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// this file contains protocol<->span converters

// Here's a handy guide for your tour of the location zoo:
//
// Imports: lsppos  -->  protocol  -->  span  -->  token
//
// lsppos.TokenMapper = (token.File, lsppos.Mapper)
// lsppos.Mapper = (line offset table, content)
//
// protocol.ColumnMapper = (URI, Content). Does all offset <=> column conversions.
// protocol.MappedRange = (protocol.ColumnMapper, {start,end} int)
// protocol.Location = (URI, protocol.Range)
// protocol.Range = (start, end Position)
// protocol.Position = (line, char uint32) 0-based UTF-16
//
// span.Point = (line?, col?, offset?) 1-based UTF-8
// span.Span = (uri URI, start, end span.Point)
// span.Range = (file token.File, start, end token.Pos)
//
// token.Pos
// token.FileSet
// token.File
// offset int
//
// TODO(adonovan): simplify this picture:
//   - Eliminate the optionality of span.{Span,Point}'s position and offset fields?
//   - Move span.Range to package safetoken. Can we eliminate it?
//     Without a ColumnMapper it's not really self-contained.
//     It is mostly used by completion. Given access to complete.mapper,
//     it could use a pair byte offsets instead.
//   - Merge lsppos.Mapper and protocol.ColumnMapper.
//   - Replace all uses of lsppos.TokenMapper by the underlying ParsedGoFile,
//     which carries a token.File and a ColumnMapper.
//   - Then delete lsppos package.
//   - ColumnMapper.OffsetPoint and .Position aren't used outside this package.
//     OffsetSpan is barely used, and its user would better off with a MappedRange
//     or protocol.Range. The span package data tyes are mostly used in tests
//     and in argument parsing (without access to file content).

package protocol

import (
	"bytes"
	"fmt"
	"go/token"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
)

// A ColumnMapper wraps the content of a file and provides mapping
// from byte offsets to and from other notations of position:
//
//   - (line, col8) pairs, where col8 is a 1-based UTF-8 column number (bytes),
//     as used by go/token;
//
//   - (line, col16) pairs, where col16 is a 1-based UTF-16 column number,
//     as used by the LSP protocol;
//
//   - (line, colRune) pairs, where colRune is a rune index, as used by ParseWork.
//
// This type does not depend on or use go/token-based representations.
// Use safetoken to map between token.Pos <=> byte offsets.
type ColumnMapper struct {
	URI     span.URI
	Content []byte

	// This field provides a line-number table, nothing more.
	// The public API of ColumnMapper doesn't mention go/token,
	// nor should it. It need not be consistent with any
	// other token.File or FileSet.
	//
	// TODO(adonovan): eliminate this field in a follow-up
	// by inlining the line-number table. Then merge this
	// type with the nearly identical lsspos.Mapper.
	//
	// TODO(adonovan): opt: quick experiments suggest that
	// ColumnMappers are created for thousands of files but the
	// m.lines field is accessed only for a small handful.
	// So it would make sense to allocate it lazily.
	lines *token.File
}

// NewColumnMapper creates a new column mapper for the given uri and content.
func NewColumnMapper(uri span.URI, content []byte) *ColumnMapper {
	fset := token.NewFileSet()
	tf := fset.AddFile(uri.Filename(), -1, len(content))
	tf.SetLinesForContent(content)

	return &ColumnMapper{
		URI:     uri,
		lines:   tf,
		Content: content,
	}
}

func URIFromSpanURI(uri span.URI) DocumentURI {
	return DocumentURI(uri)
}

func URIFromPath(path string) DocumentURI {
	return URIFromSpanURI(span.URIFromPath(path))
}

func (u DocumentURI) SpanURI() span.URI {
	return span.URIFromURI(string(u))
}

func (m *ColumnMapper) Location(s span.Span) (Location, error) {
	rng, err := m.Range(s)
	if err != nil {
		return Location{}, err
	}
	return Location{URI: URIFromSpanURI(s.URI()), Range: rng}, nil
}

func (m *ColumnMapper) Range(s span.Span) (Range, error) {
	// Assert that we aren't using the wrong mapper.
	// We check only the base name, and case insensitively,
	// because we can't assume clean paths, no symbolic links,
	// case-sensitive directories. The authoritative answer
	// requires querying the file system, and we don't want
	// to do that.
	if !strings.EqualFold(filepath.Base(string(m.URI)), filepath.Base(string(s.URI()))) {
		return Range{}, bug.Errorf("column mapper is for file %q instead of %q", m.URI, s.URI())
	}

	s, err := s.WithOffset(m.lines)
	if err != nil {
		return Range{}, err
	}
	start, err := m.Position(s.Start())
	if err != nil {
		return Range{}, err
	}
	end, err := m.Position(s.End())
	if err != nil {
		return Range{}, err
	}
	return Range{Start: start, End: end}, nil
}

// OffsetRange returns a Range for the byte-offset interval Content[start:end],
func (m *ColumnMapper) OffsetRange(start, end int) (Range, error) {
	startPosition, err := m.OffsetPosition(start)
	if err != nil {
		return Range{}, fmt.Errorf("start: %v", err)
	}

	endPosition, err := m.OffsetPosition(end)
	if err != nil {
		return Range{}, fmt.Errorf("end: %v", err)
	}

	return Range{Start: startPosition, End: endPosition}, nil
}

// OffsetSpan converts a pair of byte offsets to a Span.
func (m *ColumnMapper) OffsetSpan(start, end int) (span.Span, error) {
	if start > end {
		return span.Span{}, fmt.Errorf("start offset (%d) > end (%d)", start, end)
	}
	startPoint, err := m.OffsetPoint(start)
	if err != nil {
		return span.Span{}, err
	}
	endPoint, err := m.OffsetPoint(end)
	if err != nil {
		return span.Span{}, err
	}
	return span.New(m.URI, startPoint, endPoint), nil
}

// Position returns the protocol position for the specified point,
// which must have a byte offset.
func (m *ColumnMapper) Position(p span.Point) (Position, error) {
	if !p.HasOffset() {
		return Position{}, fmt.Errorf("point is missing offset")
	}
	return m.OffsetPosition(p.Offset())
}

// OffsetPosition returns the protocol position of the specified
// offset within m.Content.
func (m *ColumnMapper) OffsetPosition(offset int) (Position, error) {
	// We use span.OffsetToLineCol8 for its "line+1 at EOF" workaround.
	line, _, err := span.OffsetToLineCol8(m.lines, offset)
	if err != nil {
		return Position{}, fmt.Errorf("OffsetPosition: %v", err)
	}
	// If that workaround executed, skip the usual column computation.
	char := 0
	if offset != m.lines.Size() {
		char = m.utf16Column(offset)
	}
	return Position{
		Line:      uint32(line - 1),
		Character: uint32(char),
	}, nil
}

// utf16Column returns the zero-based column index of the
// specified file offset, measured in UTF-16 codes.
// Precondition: 0 <= offset <= len(m.Content).
func (m *ColumnMapper) utf16Column(offset int) int {
	s := m.Content[:offset]
	if i := bytes.LastIndex(s, []byte("\n")); i >= 0 {
		s = s[i+1:]
	}
	// s is the prefix of the line before offset.
	return utf16len(s)
}

// utf16len returns the number of codes in the UTF-16 transcoding of s.
func utf16len(s []byte) int {
	var n int
	for len(s) > 0 {
		n++

		// Fast path for ASCII.
		if s[0] < 0x80 {
			s = s[1:]
			continue
		}

		r, size := utf8.DecodeRune(s)
		if r >= 0x10000 {
			n++ // surrogate pair
		}
		s = s[size:]
	}
	return n
}

func (m *ColumnMapper) Span(l Location) (span.Span, error) {
	return m.RangeSpan(l.Range)
}

// RangeSpan converts a UTF-16 range to a Span with both the
// position (line/col) and offset fields populated.
func (m *ColumnMapper) RangeSpan(r Range) (span.Span, error) {
	start, err := m.Point(r.Start)
	if err != nil {
		return span.Span{}, err
	}
	end, err := m.Point(r.End)
	if err != nil {
		return span.Span{}, err
	}
	return span.New(m.URI, start, end).WithAll(m.lines)
}

// Offset returns the utf-8 byte offset of p within the mapped file.
func (m *ColumnMapper) Offset(p Position) (int, error) {
	start, err := m.Point(p)
	if err != nil {
		return 0, err
	}
	return start.Offset(), nil
}

// OffsetPoint returns the span.Point for the given byte offset.
func (m *ColumnMapper) OffsetPoint(offset int) (span.Point, error) {
	// We use span.ToPosition for its "line+1 at EOF" workaround.
	line, col8, err := span.OffsetToLineCol8(m.lines, offset)
	if err != nil {
		return span.Point{}, fmt.Errorf("OffsetPoint: %v", err)
	}
	return span.NewPoint(line, col8, offset), nil
}

// Point returns a span.Point for the protocol position p within the mapped file.
// The resulting point has a valid Position and Offset.
func (m *ColumnMapper) Point(p Position) (span.Point, error) {
	line := int(p.Line) + 1

	// Find byte offset of start of containing line.
	offset, err := span.ToOffset(m.lines, line, 1)
	if err != nil {
		return span.Point{}, err
	}
	lineStart := span.NewPoint(line, 1, offset)
	return span.FromUTF16Column(lineStart, int(p.Character)+1, m.Content)
}

// OffsetMappedRange returns a MappedRange for the given byte offsets.
// A MappedRange can be converted to any other form.
func (m *ColumnMapper) OffsetMappedRange(start, end int) (MappedRange, error) {
	if !(0 <= start && start <= end && end <= len(m.Content)) {
		return MappedRange{}, fmt.Errorf("invalid offsets (%d, %d) (file %s has size %d)", start, end, m.URI, len(m.Content))
	}
	return MappedRange{m, start, end}, nil
}

// A MappedRange represents a valid byte-offset range of a file.
// Through its ColumnMapper it can be converted into other forms such
// as protocol.Range or span.Span.
//
// Construct one by calling ColumnMapper.OffsetMappedRange with start/end offsets.
// From the go/token domain, call safetoken.Offsets first,
// or use a helper such as ParsedGoFile.MappedPosRange.
type MappedRange struct {
	Mapper     *ColumnMapper
	start, end int // valid byte offsets
}

// Offsets returns the (start, end) byte offsets of this range.
func (mr MappedRange) Offsets() (start, end int) { return mr.start, mr.end }

// -- convenience functions --

// URI returns the URI of the range's file.
func (mr MappedRange) URI() span.URI {
	return mr.Mapper.URI
}

// TODO(adonovan): once the fluff is removed from all the
// location-conversion methods, it will be obvious that a properly
// constructed MappedRange is always valid and its Range and Span (and
// other) methods simply cannot fail.
// At that point we might want to provide variants of methods such as
// Range and Span below that don't return an error.

// Range returns the range in protocol form.
func (mr MappedRange) Range() (Range, error) {
	return mr.Mapper.OffsetRange(mr.start, mr.end)
}

// Span returns the range in span form.
func (mr MappedRange) Span() (span.Span, error) {
	return mr.Mapper.OffsetSpan(mr.start, mr.end)
}

func IsPoint(r Range) bool {
	return r.Start.Line == r.End.Line && r.Start.Character == r.End.Character
}

// CompareRange returns -1 if a is before b, 0 if a == b, and 1 if a is after
// b.
//
// A range a is defined to be 'before' b if a.Start is before b.Start, or
// a.Start == b.Start and a.End is before b.End.
func CompareRange(a, b Range) int {
	if r := ComparePosition(a.Start, b.Start); r != 0 {
		return r
	}
	return ComparePosition(a.End, b.End)
}

// ComparePosition returns -1 if a is before b, 0 if a == b, and 1 if a is
// after b.
func ComparePosition(a, b Position) int {
	if a.Line < b.Line {
		return -1
	}
	if a.Line > b.Line {
		return 1
	}
	if a.Character < b.Character {
		return -1
	}
	if a.Character > b.Character {
		return 1
	}
	return 0
}

func Intersect(a, b Range) bool {
	if a.Start.Line > b.End.Line || a.End.Line < b.Start.Line {
		return false
	}
	return !((a.Start.Line == b.End.Line) && a.Start.Character > b.End.Character ||
		(a.End.Line == b.Start.Line) && a.End.Character < b.Start.Character)
}

// Format implements fmt.Formatter.
//
// Note: Formatter is implemented instead of Stringer (presumably) for
// performance reasons, though it is not clear that it matters in practice.
func (r Range) Format(f fmt.State, _ rune) {
	fmt.Fprintf(f, "%v-%v", r.Start, r.End)
}

// Format implements fmt.Formatter.
//
// See Range.Format for discussion of why the Formatter interface is
// implemented rather than Stringer.
func (p Position) Format(f fmt.State, _ rune) {
	fmt.Fprintf(f, "%v:%v", p.Line, p.Character)
}
