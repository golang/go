package scanner

import (
	"go/token"
	"strconv"
	"strings"
	"unicode/utf8"
)

const GoX Mode = 1 << 10
const (
	FormatStart token.Token = 1000 + iota
	MarkupStart
)

// Cursor operations are used only by the extended parser in literal modes.
func (s *Scanner) Char() rune          { return s.ch }
func (s *Scanner) Advance()            { s.next() }
func (s *Scanner) Position() token.Pos { return s.file.Pos(s.offset) }
func (s *Scanner) Semi(v bool)         { s.insertSemi = v }

// Contextual extensions only inspect the next lexical item. Newlines remain
// significant, exactly as for ordinary Go identifiers.
func (s *Scanner) lookRune() rune {
	i := s.offset
	for i < len(s.src) {
		switch s.src[i] {
		case ' ', '\t', '\r':
			i++
			continue
		case '/':
			if i+1 < len(s.src) && s.src[i+1] == '*' {
				end := strings.Index(string(s.src[i+2:]), "*/")
				if end < 0 || strings.ContainsRune(string(s.src[i:i+end+4]), '\n') {
					return -1
				}
				i += end + 4
				continue
			}
		}
		c, _ := utf8.DecodeRune(s.src[i:])
		return c
	}
	return -1
}
func (s *Scanner) LookIdentifier() bool { return isLetter(s.lookRune()) }
func (s *Scanner) StartsType() bool {
	c := s.lookRune()
	return isLetter(c) || strings.ContainsRune("*[]", c)
}

// FormatPart returns decoded text and the next boundary: '{' or closing quote.
func (s *Scanner) FormatPart(quote rune) (string, rune) {
	var b strings.Builder
	for s.ch >= 0 {
		c := s.ch
		if c == quote {
			s.next()
			return decodePart(b.String(), quote, s), quote
		}
		if c == '{' || c == '}' {
			s.next()
			if s.ch == c {
				b.WriteRune(c)
				s.next()
				continue
			}
			if c == '}' {
				s.error(s.offset, "unmatched } in F literal")
				continue
			}
			return decodePart(b.String(), quote, s), '{'
		}
		if c == '\n' && quote == '"' {
			s.error(s.offset, "newline in quoted F text")
		}
		b.WriteRune(c)
		s.next()
		if c == '\\' && quote == '"' && s.ch >= 0 {
			b.WriteRune(s.ch)
			s.next()
		}
	}
	s.error(s.offset, "unterminated F literal")
	return decodePart(b.String(), quote, s), 0
}
func decodePart(v string, q rune, s *Scanner) string {
	if q == '`' {
		return strings.ReplaceAll(v, "\r", "")
	}
	v, err := strconv.Unquote("\"" + v + "\"")
	if err != nil {
		s.error(s.offset, "invalid F escape: "+err.Error())
	}
	return v
}
func (s *Scanner) Space() {
	for s.ch == ' ' || s.ch == '\t' || s.ch == '\r' || s.ch == '\n' {
		s.next()
	}
}
func (s *Scanner) Name() string {
	var b strings.Builder
	for isLetter(s.ch) || isDecimal(s.ch) || s.ch == '.' {
		b.WriteRune(s.ch)
		s.next()
	}
	return b.String()
}
func (s *Scanner) AttributeString() string {
	q := s.ch
	s.next()
	if q == '`' {
		return s.scanRawString()
	}
	return s.scanString()
}
