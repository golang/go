// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"bytes"
	"errors"
	"fmt"
	"sort"

	"golang.org/x/text/internal/tag"
)

// isAlpha returns true if the byte is not a digit.
// b must be an ASCII letter or digit.
func isAlpha(b byte) bool {
	return b > '9'
}

// isAlphaNum returns true if the string contains only ASCII letters or digits.
func isAlphaNum(s []byte) bool {
	for _, c := range s {
		if !('a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9') {
			return false
		}
	}
	return true
}

// ErrSyntax is returned by any of the parsing functions when the
// input is not well-formed, according to BCP 47.
// TODO: return the position at which the syntax error occurred?
var ErrSyntax = errors.New("language: tag is not well-formed")

// ErrDuplicateKey is returned when a tag contains the same key twice with
// different values in the -u section.
var ErrDuplicateKey = errors.New("language: different values for same key in -u extension")

// ValueError is returned by any of the parsing functions when the
// input is well-formed but the respective subtag is not recognized
// as a valid value.
type ValueError struct {
	v [8]byte
}

// NewValueError creates a new ValueError.
func NewValueError(tag []byte) ValueError {
	var e ValueError
	copy(e.v[:], tag)
	return e
}

func (e ValueError) tag() []byte {
	n := bytes.IndexByte(e.v[:], 0)
	if n == -1 {
		n = 8
	}
	return e.v[:n]
}

// Error implements the error interface.
func (e ValueError) Error() string {
	return fmt.Sprintf("language: subtag %q is well-formed but unknown", e.tag())
}

// Subtag returns the subtag for which the error occurred.
func (e ValueError) Subtag() string {
	return string(e.tag())
}

// scanner is used to scan BCP 47 tokens, which are separated by _ or -.
type scanner struct {
	b     []byte
	bytes [max99thPercentileSize]byte
	token []byte
	start int // start position of the current token
	end   int // end position of the current token
	next  int // next point for scan
	err   error
	done  bool
}

func makeScannerString(s string) scanner {
	scan := scanner{}
	if len(s) <= len(scan.bytes) {
		scan.b = scan.bytes[:copy(scan.bytes[:], s)]
	} else {
		scan.b = []byte(s)
	}
	scan.init()
	return scan
}

// makeScanner returns a scanner using b as the input buffer.
// b is not copied and may be modified by the scanner routines.
func makeScanner(b []byte) scanner {
	scan := scanner{b: b}
	scan.init()
	return scan
}

func (s *scanner) init() {
	for i, c := range s.b {
		if c == '_' {
			s.b[i] = '-'
		}
	}
	s.scan()
}

// restToLower converts the string between start and end to lower case.
func (s *scanner) toLower(start, end int) {
	for i := start; i < end; i++ {
		c := s.b[i]
		if 'A' <= c && c <= 'Z' {
			s.b[i] += 'a' - 'A'
		}
	}
}

func (s *scanner) setError(e error) {
	if s.err == nil || (e == ErrSyntax && s.err != ErrSyntax) {
		s.err = e
	}
}

// resizeRange shrinks or grows the array at position oldStart such that
// a new string of size newSize can fit between oldStart and oldEnd.
// Sets the scan point to after the resized range.
func (s *scanner) resizeRange(oldStart, oldEnd, newSize int) {
	s.start = oldStart
	if end := oldStart + newSize; end != oldEnd {
		diff := end - oldEnd
		var b []byte
		if n := len(s.b) + diff; n > cap(s.b) {
			b = make([]byte, n)
			copy(b, s.b[:oldStart])
		} else {
			b = s.b[:n]
		}
		copy(b[end:], s.b[oldEnd:])
		s.b = b
		s.next = end + (s.next - s.end)
		s.end = end
	}
}

// replace replaces the current token with repl.
func (s *scanner) replace(repl string) {
	s.resizeRange(s.start, s.end, len(repl))
	copy(s.b[s.start:], repl)
}

// gobble removes the current token from the input.
// Caller must call scan after calling gobble.
func (s *scanner) gobble(e error) {
	s.setError(e)
	if s.start == 0 {
		s.b = s.b[:+copy(s.b, s.b[s.next:])]
		s.end = 0
	} else {
		s.b = s.b[:s.start-1+copy(s.b[s.start-1:], s.b[s.end:])]
		s.end = s.start - 1
	}
	s.next = s.start
}

// deleteRange removes the given range from s.b before the current token.
func (s *scanner) deleteRange(start, end int) {
	s.b = s.b[:start+copy(s.b[start:], s.b[end:])]
	diff := end - start
	s.next -= diff
	s.start -= diff
	s.end -= diff
}

// scan parses the next token of a BCP 47 string.  Tokens that are larger
// than 8 characters or include non-alphanumeric characters result in an error
// and are gobbled and removed from the output.
// It returns the end position of the last token consumed.
func (s *scanner) scan() (end int) {
	end = s.end
	s.token = nil
	for s.start = s.next; s.next < len(s.b); {
		i := bytes.IndexByte(s.b[s.next:], '-')
		if i == -1 {
			s.end = len(s.b)
			s.next = len(s.b)
			i = s.end - s.start
		} else {
			s.end = s.next + i
			s.next = s.end + 1
		}
		token := s.b[s.start:s.end]
		if i < 1 || i > 8 || !isAlphaNum(token) {
			s.gobble(ErrSyntax)
			continue
		}
		s.token = token
		return end
	}
	if n := len(s.b); n > 0 && s.b[n-1] == '-' {
		s.setError(ErrSyntax)
		s.b = s.b[:len(s.b)-1]
	}
	s.done = true
	return end
}

// acceptMinSize parses multiple tokens of the given size or greater.
// It returns the end position of the last token consumed.
func (s *scanner) acceptMinSize(min int) (end int) {
	end = s.end
	s.scan()
	for ; len(s.token) >= min; s.scan() {
		end = s.end
	}
	return end
}

// Parse parses the given BCP 47 string and returns a valid Tag. If parsing
// failed it returns an error and any part of the tag that could be parsed.
// If parsing succeeded but an unknown value was found, it returns
// ValueError. The Tag returned in this case is just stripped of the unknown
// value. All other values are preserved. It accepts tags in the BCP 47 format
// and extensions to this standard defined in
// https://www.unicode.org/reports/tr35/#Unicode_Language_and_Locale_Identifiers.
func Parse(s string) (t Tag, err error) {
	// TODO: consider supporting old-style locale key-value pairs.
	if s == "" {
		return Und, ErrSyntax
	}
	defer func() {
		if recover() != nil {
			t = Und
			err = ErrSyntax
			return
		}
	}()
	if len(s) <= maxAltTaglen {
		b := [maxAltTaglen]byte{}
		for i, c := range s {
			// Generating invalid UTF-8 is okay as it won't match.
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			} else if c == '_' {
				c = '-'
			}
			b[i] = byte(c)
		}
		if t, ok := grandfathered(b); ok {
			return t, nil
		}
	}
	scan := makeScannerString(s)
	return parse(&scan, s)
}

func parse(scan *scanner, s string) (t Tag, err error) {
	t = Und
	var end int
	if n := len(scan.token); n <= 1 {
		scan.toLower(0, len(scan.b))
		if n == 0 || scan.token[0] != 'x' {
			return t, ErrSyntax
		}
		end = parseExtensions(scan)
	} else if n >= 4 {
		return Und, ErrSyntax
	} else { // the usual case
		t, end = parseTag(scan, true)
		if n := len(scan.token); n == 1 {
			t.pExt = uint16(end)
			end = parseExtensions(scan)
		} else if end < len(scan.b) {
			scan.setError(ErrSyntax)
			scan.b = scan.b[:end]
		}
	}
	if int(t.pVariant) < len(scan.b) {
		if end < len(s) {
			s = s[:end]
		}
		if len(s) > 0 && tag.Compare(s, scan.b) == 0 {
			t.str = s
		} else {
			t.str = string(scan.b)
		}
	} else {
		t.pVariant, t.pExt = 0, 0
	}
	return t, scan.err
}

// parseTag parses language, script, region and variants.
// It returns a Tag and the end position in the input that was parsed.
// If doNorm is true, then <lang>-<extlang> will be normalized to <extlang>.
func parseTag(scan *scanner, doNorm bool) (t Tag, end int) {
	var e error
	// TODO: set an error if an unknown lang, script or region is encountered.
	t.LangID, e = getLangID(scan.token)
	scan.setError(e)
	scan.replace(t.LangID.String())
	langStart := scan.start
	end = scan.scan()
	for len(scan.token) == 3 && isAlpha(scan.token[0]) {
		// From http://tools.ietf.org/html/bcp47, <lang>-<extlang> tags are equivalent
		// to a tag of the form <extlang>.
		if doNorm {
			lang, e := getLangID(scan.token)
			if lang != 0 {
				t.LangID = lang
				langStr := lang.String()
				copy(scan.b[langStart:], langStr)
				scan.b[langStart+len(langStr)] = '-'
				scan.start = langStart + len(langStr) + 1
			}
			scan.gobble(e)
		}
		end = scan.scan()
	}
	if len(scan.token) == 4 && isAlpha(scan.token[0]) {
		t.ScriptID, e = getScriptID(script, scan.token)
		if t.ScriptID == 0 {
			scan.gobble(e)
		}
		end = scan.scan()
	}
	if n := len(scan.token); n >= 2 && n <= 3 {
		t.RegionID, e = getRegionID(scan.token)
		if t.RegionID == 0 {
			scan.gobble(e)
		} else {
			scan.replace(t.RegionID.String())
		}
		end = scan.scan()
	}
	scan.toLower(scan.start, len(scan.b))
	t.pVariant = byte(end)
	end = parseVariants(scan, end, t)
	t.pExt = uint16(end)
	return t, end
}

var separator = []byte{'-'}

// parseVariants scans tokens as long as each token is a valid variant string.
// Duplicate variants are removed.
func parseVariants(scan *scanner, end int, t Tag) int {
	start := scan.start
	varIDBuf := [4]uint8{}
	variantBuf := [4][]byte{}
	varID := varIDBuf[:0]
	variant := variantBuf[:0]
	last := -1
	needSort := false
	for ; len(scan.token) >= 4; scan.scan() {
		// TODO: measure the impact of needing this conversion and redesign
		// the data structure if there is an issue.
		v, ok := variantIndex[string(scan.token)]
		if !ok {
			// unknown variant
			// TODO: allow user-defined variants?
			scan.gobble(NewValueError(scan.token))
			continue
		}
		varID = append(varID, v)
		variant = append(variant, scan.token)
		if !needSort {
			if last < int(v) {
				last = int(v)
			} else {
				needSort = true
				// There is no legal combinations of more than 7 variants
				// (and this is by no means a useful sequence).
				const maxVariants = 8
				if len(varID) > maxVariants {
					break
				}
			}
		}
		end = scan.end
	}
	if needSort {
		sort.Sort(variantsSort{varID, variant})
		k, l := 0, -1
		for i, v := range varID {
			w := int(v)
			if l == w {
				// Remove duplicates.
				continue
			}
			varID[k] = varID[i]
			variant[k] = variant[i]
			k++
			l = w
		}
		if str := bytes.Join(variant[:k], separator); len(str) == 0 {
			end = start - 1
		} else {
			scan.resizeRange(start, end, len(str))
			copy(scan.b[scan.start:], str)
			end = scan.end
		}
	}
	return end
}

type variantsSort struct {
	i []uint8
	v [][]byte
}

func (s variantsSort) Len() int {
	return len(s.i)
}

func (s variantsSort) Swap(i, j int) {
	s.i[i], s.i[j] = s.i[j], s.i[i]
	s.v[i], s.v[j] = s.v[j], s.v[i]
}

func (s variantsSort) Less(i, j int) bool {
	return s.i[i] < s.i[j]
}

type bytesSort struct {
	b [][]byte
	n int // first n bytes to compare
}

func (b bytesSort) Len() int {
	return len(b.b)
}

func (b bytesSort) Swap(i, j int) {
	b.b[i], b.b[j] = b.b[j], b.b[i]
}

func (b bytesSort) Less(i, j int) bool {
	for k := 0; k < b.n; k++ {
		if b.b[i][k] == b.b[j][k] {
			continue
		}
		return b.b[i][k] < b.b[j][k]
	}
	return false
}

// parseExtensions parses and normalizes the extensions in the buffer.
// It returns the last position of scan.b that is part of any extension.
// It also trims scan.b to remove excess parts accordingly.
func parseExtensions(scan *scanner) int {
	start := scan.start
	exts := [][]byte{}
	private := []byte{}
	end := scan.end
	for len(scan.token) == 1 {
		extStart := scan.start
		ext := scan.token[0]
		end = parseExtension(scan)
		extension := scan.b[extStart:end]
		if len(extension) < 3 || (ext != 'x' && len(extension) < 4) {
			scan.setError(ErrSyntax)
			end = extStart
			continue
		} else if start == extStart && (ext == 'x' || scan.start == len(scan.b)) {
			scan.b = scan.b[:end]
			return end
		} else if ext == 'x' {
			private = extension
			break
		}
		exts = append(exts, extension)
	}
	sort.Sort(bytesSort{exts, 1})
	if len(private) > 0 {
		exts = append(exts, private)
	}
	scan.b = scan.b[:start]
	if len(exts) > 0 {
		scan.b = append(scan.b, bytes.Join(exts, separator)...)
	} else if start > 0 {
		// Strip trailing '-'.
		scan.b = scan.b[:start-1]
	}
	return end
}

// parseExtension parses a single extension and returns the position of
// the extension end.
func parseExtension(scan *scanner) int {
	start, end := scan.start, scan.end
	switch scan.token[0] {
	case 'u': // https://www.ietf.org/rfc/rfc6067.txt
		attrStart := end
		scan.scan()
		for last := []byte{}; len(scan.token) > 2; scan.scan() {
			if bytes.Compare(scan.token, last) != -1 {
				// Attributes are unsorted. Start over from scratch.
				p := attrStart + 1
				scan.next = p
				attrs := [][]byte{}
				for scan.scan(); len(scan.token) > 2; scan.scan() {
					attrs = append(attrs, scan.token)
					end = scan.end
				}
				sort.Sort(bytesSort{attrs, 3})
				copy(scan.b[p:], bytes.Join(attrs, separator))
				break
			}
			last = scan.token
			end = scan.end
		}
		// Scan key-type sequences. A key is of length 2 and may be followed
		// by 0 or more "type" subtags from 3 to the maximum of 8 letters.
		var last, key []byte
		for attrEnd := end; len(scan.token) == 2; last = key {
			key = scan.token
			end = scan.end
			for scan.scan(); end < scan.end && len(scan.token) > 2; scan.scan() {
				end = scan.end
			}
			// TODO: check key value validity
			if bytes.Compare(key, last) != 1 || scan.err != nil {
				// We have an invalid key or the keys are not sorted.
				// Start scanning keys from scratch and reorder.
				p := attrEnd + 1
				scan.next = p
				keys := [][]byte{}
				for scan.scan(); len(scan.token) == 2; {
					keyStart := scan.start
					end = scan.end
					for scan.scan(); end < scan.end && len(scan.token) > 2; scan.scan() {
						end = scan.end
					}
					keys = append(keys, scan.b[keyStart:end])
				}
				sort.Stable(bytesSort{keys, 2})
				if n := len(keys); n > 0 {
					k := 0
					for i := 1; i < n; i++ {
						if !bytes.Equal(keys[k][:2], keys[i][:2]) {
							k++
							keys[k] = keys[i]
						} else if !bytes.Equal(keys[k], keys[i]) {
							scan.setError(ErrDuplicateKey)
						}
					}
					keys = keys[:k+1]
				}
				reordered := bytes.Join(keys, separator)
				if e := p + len(reordered); e < end {
					scan.deleteRange(e, end)
					end = e
				}
				copy(scan.b[p:], reordered)
				break
			}
		}
	case 't': // https://www.ietf.org/rfc/rfc6497.txt
		scan.scan()
		if n := len(scan.token); n >= 2 && n <= 3 && isAlpha(scan.token[1]) {
			_, end = parseTag(scan, false)
			scan.toLower(start, end)
		}
		for len(scan.token) == 2 && !isAlpha(scan.token[1]) {
			end = scan.acceptMinSize(3)
		}
	case 'x':
		end = scan.acceptMinSize(1)
	default:
		end = scan.acceptMinSize(2)
	}
	return end
}

// getExtension returns the name, body and end position of the extension.
func getExtension(s string, p int) (end int, ext string) {
	if s[p] == '-' {
		p++
	}
	if s[p] == 'x' {
		return len(s), s[p:]
	}
	end = nextExtension(s, p)
	return end, s[p:end]
}

// nextExtension finds the next extension within the string, searching
// for the -<char>- pattern from position p.
// In the fast majority of cases, language tags will have at most
// one extension and extensions tend to be small.
func nextExtension(s string, p int) int {
	for n := len(s) - 3; p < n; {
		if s[p] == '-' {
			if s[p+2] == '-' {
				return p
			}
			p += 3
		} else {
			p++
		}
	}
	return len(s)
}
