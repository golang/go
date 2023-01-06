// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines ColumnMapper, which wraps a file content buffer
// ([]byte) and provides efficient conversion between every kind of
// position representation.
//
// Here's a handy guide for your tour of the location zoo:
//
// Imports: lsppos  -->  protocol  -->  span  -->  token
//
// lsppos.Mapper = (line offset table, content)
//
// protocol: for the LSP protocol.
// protocol.ColumnMapper = (URI, Content). Does all offset <=> column conversions.
// protocol.MappedRange = (protocol.ColumnMapper, {start,end} int)
// protocol.Location = (URI, protocol.Range)
// protocol.Range = (start, end Position)
// protocol.Position = (line, char uint32) 0-based UTF-16
//
// span: for optional fields; useful for CLIs and tests without access to file contents.
// span.Point = (line?, col?, offset?) 1-based UTF-8
// span.Span = (uri URI, start, end span.Point)
// span.Range = (file token.File, start, end token.Pos)
//
// token: for interaction with the go/* syntax packages:
// token.Pos
// token.FileSet
// token.File
// offset int
//
// TODO(adonovan): simplify this picture:
//   - Move span.Range to package safetoken. Can we eliminate most uses in gopls?
//     Without a ColumnMapper it's not really self-contained.
//     It is mostly used by completion. Given access to complete.mapper,
//     it could use a pair of byte offsets instead.
//   - Merge lsppos.Mapper and protocol.ColumnMapper.
//   - Then delete lsppos package.
//   - ColumnMapper.OffsetPoint and .PointPosition aren't used outside this package.
//     OffsetSpan is barely used, and its user would better off with a MappedRange
//     or protocol.Range. The span package data types are mostly used in tests
//     and in argument parsing (without access to file content).
//   - share core conversion primitives with span package.
//   - rename ColumnMapper to just Mapper, since it also maps lines
//     <=> offsets, and move it to file mapper.go.

package protocol

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"golang.org/x/tools/gopls/internal/lsp/safetoken"
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
//     (Not yet implemented, but could easily be.)
//
// All conversion methods are named "FromTo", where From and To are the two types.
// For example, the PointPosition method converts from a Point to a Position.
//
// ColumnMapper does not intrinsically depend on go/token-based
// representations.  Use safetoken to map between token.Pos <=> byte
// offsets, or the convenience methods such as PosPosition,
// NodePosition, or NodeRange.
type ColumnMapper struct {
	URI     span.URI
	Content []byte

	// Line-number information is requested only for a tiny
	// fraction of ColumnMappers, so we compute it lazily.
	// Call initLines() before accessing fields below.
	linesOnce sync.Once
	lineStart []int // byte offset of start of ith line (0-based); last=EOF iff \n-terminated
	nonASCII  bool

	// TODO(adonovan): adding an extra lineStart entry for EOF
	// might simplify every method that accesses it. Try it out.
}

// NewColumnMapper creates a new column mapper for the given URI and content.
func NewColumnMapper(uri span.URI, content []byte) *ColumnMapper {
	return &ColumnMapper{URI: uri, Content: content}
}

// initLines populates the lineStart table.
func (m *ColumnMapper) initLines() {
	m.linesOnce.Do(func() {
		nlines := bytes.Count(m.Content, []byte("\n"))
		m.lineStart = make([]int, 1, nlines+1) // initially []int{0}
		for offset, b := range m.Content {
			if b == '\n' {
				m.lineStart = append(m.lineStart, offset+1)
			}
			if b >= utf8.RuneSelf {
				m.nonASCII = true
			}
		}
	})
}

// -- conversions from span (UTF-8) domain --

// Location converts a (UTF-8) span to a protocol (UTF-16) range.
// Precondition: the URIs of Location and ColumnMapper match.
// TODO(adonovan): rename to SpanLocation.
func (m *ColumnMapper) Location(s span.Span) (Location, error) {
	rng, err := m.Range(s)
	if err != nil {
		return Location{}, err
	}
	return Location{URI: URIFromSpanURI(s.URI()), Range: rng}, nil
}

// Range converts a (UTF-8) span to a protocol (UTF-16) range.
// Precondition: the URIs of Span and ColumnMapper match.
// TODO(adonovan): rename to SpanRange
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
	start, err := m.PointPosition(s.Start())
	if err != nil {
		return Range{}, fmt.Errorf("start: %w", err)
	}
	end, err := m.PointPosition(s.End())
	if err != nil {
		return Range{}, fmt.Errorf("end: %w", err)
	}
	return Range{Start: start, End: end}, nil
}

// PointPosition converts a valid span (UTF-8) point to a protocol (UTF-8) position.
func (m *ColumnMapper) PointPosition(p span.Point) (Position, error) {
	if p.HasPosition() {
		line, col8 := p.Line()-1, p.Column()-1 // both 0-based
		m.initLines()
		if line >= len(m.lineStart) {
			return Position{}, fmt.Errorf("line number %d out of range (max %d)", line, len(m.lineStart))
		}
		offset := m.lineStart[line]
		end := offset + col8

		// Validate column.
		if end > len(m.Content) {
			return Position{}, fmt.Errorf("column is beyond end of file")
		} else if line+1 < len(m.lineStart) && end >= m.lineStart[line+1] {
			return Position{}, fmt.Errorf("column is beyond end of line")
		}

		char := utf16len(m.Content[offset:end])
		return Position{Line: uint32(line), Character: uint32(char)}, nil
	}
	if p.HasOffset() {
		return m.OffsetPosition(p.Offset())
	}
	return Position{}, fmt.Errorf("point has neither offset nor line/column")
}

// -- conversions from byte offsets --

// OffsetRange converts a byte-offset interval to a protocol (UTF-16) range.
func (m *ColumnMapper) OffsetRange(start, end int) (Range, error) {
	if start > end {
		return Range{}, fmt.Errorf("start offset (%d) > end (%d)", start, end)
	}
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

// OffsetSpan converts a byte-offset interval to a (UTF-8) span.
func (m *ColumnMapper) OffsetSpan(start, end int) (span.Span, error) {
	if start > end {
		return span.Span{}, fmt.Errorf("start offset (%d) > end (%d)", start, end)
	}
	startPoint, err := m.OffsetPoint(start)
	if err != nil {
		return span.Span{}, fmt.Errorf("start: %v", err)
	}
	endPoint, err := m.OffsetPoint(end)
	if err != nil {
		return span.Span{}, fmt.Errorf("end: %v", err)
	}
	return span.New(m.URI, startPoint, endPoint), nil
}

// OffsetPosition converts a byte offset to a protocol (UTF-16) position.
func (m *ColumnMapper) OffsetPosition(offset int) (Position, error) {
	if !(0 <= offset && offset <= len(m.Content)) {
		return Position{}, fmt.Errorf("invalid offset %d (want 0-%d)", offset, len(m.Content))
	}
	line, col16 := m.lineCol16(offset)
	return Position{Line: uint32(line), Character: uint32(col16)}, nil
}

// lineCol16 converts a valid byte offset to line and UTF-16 column numbers, both 0-based.
func (m *ColumnMapper) lineCol16(offset int) (int, int) {
	line, start, cr := m.line(offset)
	var col16 int
	if m.nonASCII {
		col16 = utf16len(m.Content[start:offset])
	} else {
		col16 = offset - start
	}
	if cr {
		col16-- // retreat from \r at line end
	}
	return line, col16
}

// lineCol8 converts a valid byte offset to line and UTF-8 column numbers, both 0-based.
func (m *ColumnMapper) lineCol8(offset int) (int, int) {
	line, start, cr := m.line(offset)
	col8 := offset - start
	if cr {
		col8-- // retreat from \r at line end
	}
	return line, col8
}

// line returns:
// - the 0-based index of the line that encloses the (valid) byte offset;
// - the start offset of that line; and
// - whether the offset denotes a carriage return (\r) at line end.
func (m *ColumnMapper) line(offset int) (int, int, bool) {
	m.initLines()
	// In effect, binary search returns a 1-based result.
	line := sort.Search(len(m.lineStart), func(i int) bool {
		return offset < m.lineStart[i]
	})

	// Adjustment for line-endings: \r|\n is the same as |\r\n.
	var eol int
	if line == len(m.lineStart) {
		eol = len(m.Content) // EOF
	} else {
		eol = m.lineStart[line] - 1
	}
	cr := offset == eol && offset > 0 && m.Content[offset-1] == '\r'

	line-- // 0-based

	return line, m.lineStart[line], cr
}

// OffsetPoint converts a byte offset to a span (UTF-8) point.
func (m *ColumnMapper) OffsetPoint(offset int) (span.Point, error) {
	if !(0 <= offset && offset <= len(m.Content)) {
		return span.Point{}, fmt.Errorf("invalid offset %d (want 0-%d)", offset, len(m.Content))
	}
	line, col8 := m.lineCol8(offset)
	return span.NewPoint(line+1, col8+1, offset), nil
}

// -- conversions from protocol domain --

// Span converts a protocol (UTF-16) Location to a (UTF-8) span.
// Precondition: the URIs of Location and ColumnMapper match.
// TODO(adonovan): rename to LocationSpan.
func (m *ColumnMapper) Span(l Location) (span.Span, error) {
	// TODO(adonovan): check that l.URI matches m.URI.
	return m.RangeSpan(l.Range)
}

// RangeSpan converts a protocol (UTF-16) range to a (UTF-8) span.
// The resulting span has valid Positions and Offsets.
func (m *ColumnMapper) RangeSpan(r Range) (span.Span, error) {
	start, end, err := m.RangeOffsets(r)
	if err != nil {
		return span.Span{}, err
	}
	return m.OffsetSpan(start, end)
}

// RangeOffsets converts a protocol (UTF-16) range to start/end byte offsets.
func (m *ColumnMapper) RangeOffsets(r Range) (int, int, error) {
	start, err := m.PositionOffset(r.Start)
	if err != nil {
		return 0, 0, err
	}
	end, err := m.PositionOffset(r.End)
	if err != nil {
		return 0, 0, err
	}
	return start, end, nil
}

// PositionOffset converts a protocol (UTF-16) position to a byte offset.
func (m *ColumnMapper) PositionOffset(p Position) (int, error) {
	m.initLines()

	// Validate line number.
	if p.Line > uint32(len(m.lineStart)) {
		return 0, fmt.Errorf("line number %d out of range 0-%d", p.Line, len(m.lineStart))
	} else if p.Line == uint32(len(m.lineStart)) {
		if p.Character == 0 {
			return len(m.Content), nil // EOF
		}
		return 0, fmt.Errorf("column is beyond end of file")
	}

	offset := m.lineStart[p.Line]
	content := m.Content[offset:] // rest of file from start of enclosing line

	// Advance bytes up to the required number of UTF-16 codes.
	col8 := 0
	for col16 := 0; col16 < int(p.Character); col16++ {
		r, sz := utf8.DecodeRune(content)
		if sz == 0 {
			return 0, fmt.Errorf("column is beyond end of file")
		}
		if r == '\n' {
			// TODO(adonovan): report an error?
			break // column denotes position beyond EOL: truncate
		}
		if sz == 1 && r == utf8.RuneError {
			return 0, fmt.Errorf("buffer contains invalid UTF-8 text")
		}
		content = content[sz:]

		if r >= 0x10000 {
			col16++ // rune was encoded by a pair of surrogate UTF-16 codes

			if col16 == int(p.Character) {
				break // requested position is in the middle of a rune
			}
		}
		col8 += sz
	}
	return offset + col8, nil
}

// PositionPoint converts a protocol (UTF-16) position to a span (UTF-8) point.
// The resulting point has a valid Position and Offset.
func (m *ColumnMapper) PositionPoint(p Position) (span.Point, error) {
	offset, err := m.PositionOffset(p)
	if err != nil {
		return span.Point{}, err
	}
	line, col8 := m.lineCol8(offset)

	return span.NewPoint(line+1, col8+1, offset), nil
}

// -- go/token domain convenience methods --

// PosPosition converts a token pos to a protocol (UTF-16) position.
func (m *ColumnMapper) PosPosition(tf *token.File, pos token.Pos) (Position, error) {
	offset, err := safetoken.Offset(tf, pos)
	if err != nil {
		return Position{}, err
	}
	return m.OffsetPosition(offset)
}

// PosPosition converts a token range to a protocol (UTF-16) range.
func (m *ColumnMapper) PosRange(tf *token.File, start, end token.Pos) (Range, error) {
	startOffset, endOffset, err := safetoken.Offsets(tf, start, end)
	if err != nil {
		return Range{}, err
	}
	return m.OffsetRange(startOffset, endOffset)
}

// PosPosition converts a syntax node range to a protocol (UTF-16) range.
func (m *ColumnMapper) NodeRange(tf *token.File, node ast.Node) (Range, error) {
	return m.PosRange(tf, node.Pos(), node.End())
}

// -- MappedRange --

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
	start, end int // valid byte offsets:  0 <= start <= end <= len(Mapper.Content)
}

// Offsets returns the (start, end) byte offsets of this range.
func (mr MappedRange) Offsets() (start, end int) { return mr.start, mr.end }

// -- convenience functions --

// URI returns the URI of the range's file.
func (mr MappedRange) URI() span.URI {
	return mr.Mapper.URI
}

// TODO(adonovan): the Range and Span methods of a properly
// constructed MappedRange cannot fail. Change them to panic instead
// of returning the error, for convenience of the callers.

// Range returns the range in protocol form.
func (mr MappedRange) Range() (Range, error) {
	return mr.Mapper.OffsetRange(mr.start, mr.end)
}

// Span returns the range in span form.
func (mr MappedRange) Span() (span.Span, error) {
	return mr.Mapper.OffsetSpan(mr.start, mr.end)
}

// -- misc helpers --

func URIFromSpanURI(uri span.URI) DocumentURI {
	return DocumentURI(uri) // simple conversion
}

func URIFromPath(path string) DocumentURI {
	return URIFromSpanURI(span.URIFromPath(path)) // normalizing conversion
}

func (u DocumentURI) SpanURI() span.URI {
	return span.URIFromURI(string(u)) // normalizing conversion
}

func IsPoint(r Range) bool {
	return r.Start.Line == r.End.Line && r.Start.Character == r.End.Character
}

// CompareLocation defines a three-valued comparison over locations,
// lexicographically ordered by (URI, Range).
func CompareLocation(x, y Location) int {
	if x.URI != y.URI {
		if x.URI < y.URI {
			return -1
		} else {
			return +1
		}
	}
	return CompareRange(x.Range, y.Range)
}

// CompareRange returns -1 if a is before b, 0 if a == b, and 1 if a is after b.
//
// A range a is defined to be 'before' b if a.Start is before b.Start, or
// a.Start == b.Start and a.End is before b.End.
func CompareRange(a, b Range) int {
	if r := ComparePosition(a.Start, b.Start); r != 0 {
		return r
	}
	return ComparePosition(a.End, b.End)
}

// ComparePosition returns -1 if a is before b, 0 if a == b, and 1 if a is after b.
func ComparePosition(a, b Position) int {
	if a.Line != b.Line {
		if a.Line < b.Line {
			return -1
		} else {
			return +1
		}
	}
	if a.Character != b.Character {
		if a.Character < b.Character {
			return -1
		} else {
			return +1
		}
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

// -- implementation helpers --

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
