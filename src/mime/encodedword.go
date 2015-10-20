// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// A WordEncoder is a RFC 2047 encoded-word encoder.
type WordEncoder byte

const (
	// BEncoding represents Base64 encoding scheme as defined by RFC 2045.
	BEncoding = WordEncoder('b')
	// QEncoding represents the Q-encoding scheme as defined by RFC 2047.
	QEncoding = WordEncoder('q')
)

var (
	errInvalidWord = errors.New("mime: invalid RFC 2047 encoded-word")
)

// Encode returns the encoded-word form of s. If s is ASCII without special
// characters, it is returned unchanged. The provided charset is the IANA
// charset name of s. It is case insensitive.
func (e WordEncoder) Encode(charset, s string) string {
	if !needsEncoding(s) {
		return s
	}
	return e.encodeWord(charset, s)
}

func needsEncoding(s string) bool {
	for _, b := range s {
		if (b < ' ' || b > '~') && b != '\t' {
			return true
		}
	}
	return false
}

// encodeWord encodes a string into an encoded-word.
func (e WordEncoder) encodeWord(charset, s string) string {
	buf := getBuffer()
	defer putBuffer(buf)

	e.openWord(buf, charset)
	if e == BEncoding {
		e.bEncode(buf, charset, s)
	} else {
		e.qEncode(buf, charset, s)
	}
	closeWord(buf)

	return buf.String()
}

const (
	// The maximum length of an encoded-word is 75 characters.
	// See RFC 2047, section 2.
	maxEncodedWordLen = 75
	// maxContentLen is how much content can be encoded, ignoring the header and
	// 2-byte footer.
	maxContentLen = maxEncodedWordLen - len("=?UTF-8?") - len("?=")
)

var maxBase64Len = base64.StdEncoding.DecodedLen(maxContentLen)

// bEncode encodes s using base64 encoding and writes it to buf.
func (e WordEncoder) bEncode(buf *bytes.Buffer, charset, s string) {
	w := base64.NewEncoder(base64.StdEncoding, buf)
	// If the charset is not UTF-8 or if the content is short, do not bother
	// splitting the encoded-word.
	if !isUTF8(charset) || base64.StdEncoding.EncodedLen(len(s)) <= maxContentLen {
		io.WriteString(w, s)
		w.Close()
		return
	}

	var currentLen, last, runeLen int
	for i := 0; i < len(s); i += runeLen {
		// Multi-byte characters must not be split accross encoded-words.
		// See RFC 2047, section 5.3.
		_, runeLen = utf8.DecodeRuneInString(s[i:])

		if currentLen+runeLen <= maxBase64Len {
			currentLen += runeLen
		} else {
			io.WriteString(w, s[last:i])
			w.Close()
			e.splitWord(buf, charset)
			last = i
			currentLen = runeLen
		}
	}
	io.WriteString(w, s[last:])
	w.Close()
}

// qEncode encodes s using Q encoding and writes it to buf. It splits the
// encoded-words when necessary.
func (e WordEncoder) qEncode(buf *bytes.Buffer, charset, s string) {
	// We only split encoded-words when the charset is UTF-8.
	if !isUTF8(charset) {
		writeQString(buf, s)
		return
	}

	var currentLen, runeLen int
	for i := 0; i < len(s); i += runeLen {
		b := s[i]
		// Multi-byte characters must not be split accross encoded-words.
		// See RFC 2047, section 5.3.
		var encLen int
		if b >= ' ' && b <= '~' && b != '=' && b != '?' && b != '_' {
			runeLen, encLen = 1, 1
		} else {
			_, runeLen = utf8.DecodeRuneInString(s[i:])
			encLen = 3 * runeLen
		}

		if currentLen+encLen > maxContentLen {
			e.splitWord(buf, charset)
			currentLen = 0
		}
		writeQString(buf, s[i:i+runeLen])
		currentLen += encLen
	}
}

// writeQString encodes s using Q encoding and writes it to buf.
func writeQString(buf *bytes.Buffer, s string) {
	for i := 0; i < len(s); i++ {
		switch b := s[i]; {
		case b == ' ':
			buf.WriteByte('_')
		case b >= '!' && b <= '~' && b != '=' && b != '?' && b != '_':
			buf.WriteByte(b)
		default:
			buf.WriteByte('=')
			buf.WriteByte(upperhex[b>>4])
			buf.WriteByte(upperhex[b&0x0f])
		}
	}
}

// openWord writes the beginning of an encoded-word into buf.
func (e WordEncoder) openWord(buf *bytes.Buffer, charset string) {
	buf.WriteString("=?")
	buf.WriteString(charset)
	buf.WriteByte('?')
	buf.WriteByte(byte(e))
	buf.WriteByte('?')
}

// closeWord writes the end of an encoded-word into buf.
func closeWord(buf *bytes.Buffer) {
	buf.WriteString("?=")
}

// splitWord closes the current encoded-word and opens a new one.
func (e WordEncoder) splitWord(buf *bytes.Buffer, charset string) {
	closeWord(buf)
	buf.WriteByte(' ')
	e.openWord(buf, charset)
}

func isUTF8(charset string) bool {
	return strings.EqualFold(charset, "UTF-8")
}

const upperhex = "0123456789ABCDEF"

// A WordDecoder decodes MIME headers containing RFC 2047 encoded-words.
type WordDecoder struct {
	// CharsetReader, if non-nil, defines a function to generate
	// charset-conversion readers, converting from the provided
	// charset into UTF-8.
	// Charsets are always lower-case. utf-8, iso-8859-1 and us-ascii charsets
	// are handled by default.
	// One of the the CharsetReader's result values must be non-nil.
	CharsetReader func(charset string, input io.Reader) (io.Reader, error)
}

// Decode decodes an RFC 2047 encoded-word.
func (d *WordDecoder) Decode(word string) (string, error) {
	if !strings.HasPrefix(word, "=?") || !strings.HasSuffix(word, "?=") || strings.Count(word, "?") != 4 {
		return "", errInvalidWord
	}
	word = word[2 : len(word)-2]

	// split delimits the first 2 fields
	split := strings.IndexByte(word, '?')
	// the field after split must only be one byte
	if word[split+2] != '?' {
		return "", errInvalidWord
	}

	// split word "UTF-8?q?ascii" into "UTF-8", 'q', and "ascii"
	charset := word[:split]
	encoding := word[split+1]
	text := word[split+3:]

	content, err := decode(encoding, text)
	if err != nil {
		return "", err
	}

	buf := getBuffer()
	defer putBuffer(buf)

	if err := d.convert(buf, charset, content); err != nil {
		return "", err
	}

	return buf.String(), nil
}

// DecodeHeader decodes all encoded-words of the given string. It returns an
// error if and only if CharsetReader of d returns an error.
func (d *WordDecoder) DecodeHeader(header string) (string, error) {
	// If there is no encoded-word, returns before creating a buffer.
	i := strings.Index(header, "=?")
	if i == -1 {
		return header, nil
	}

	buf := getBuffer()
	defer putBuffer(buf)

	buf.WriteString(header[:i])
	header = header[i:]

	betweenWords := false
	for {
		start := strings.Index(header, "=?")
		if start == -1 {
			break
		}
		cur := start + len("=?")

		i := strings.Index(header[cur:], "?")
		if i == -1 {
			break
		}
		charset := header[cur : cur+i]
		cur += i + len("?")

		if len(header) < cur+len("Q??=") {
			break
		}
		encoding := header[cur]
		cur++

		if header[cur] != '?' {
			break
		}
		cur++

		j := strings.Index(header[cur:], "?=")
		if j == -1 {
			break
		}
		text := header[cur : cur+j]
		end := cur + j + len("?=")

		content, err := decode(encoding, text)
		if err != nil {
			betweenWords = false
			buf.WriteString(header[:start+2])
			header = header[start+2:]
			continue
		}

		// Write characters before the encoded-word. White-space and newline
		// characters separating two encoded-words must be deleted.
		if start > 0 && (!betweenWords || hasNonWhitespace(header[:start])) {
			buf.WriteString(header[:start])
		}

		if err := d.convert(buf, charset, content); err != nil {
			return "", err
		}

		header = header[end:]
		betweenWords = true
	}

	if len(header) > 0 {
		buf.WriteString(header)
	}

	return buf.String(), nil
}

func decode(encoding byte, text string) ([]byte, error) {
	switch encoding {
	case 'B', 'b':
		return base64.StdEncoding.DecodeString(text)
	case 'Q', 'q':
		return qDecode(text)
	default:
		return nil, errInvalidWord
	}
}

func (d *WordDecoder) convert(buf *bytes.Buffer, charset string, content []byte) error {
	switch {
	case strings.EqualFold("utf-8", charset):
		buf.Write(content)
	case strings.EqualFold("iso-8859-1", charset):
		for _, c := range content {
			buf.WriteRune(rune(c))
		}
	case strings.EqualFold("us-ascii", charset):
		for _, c := range content {
			if c >= utf8.RuneSelf {
				buf.WriteRune(unicode.ReplacementChar)
			} else {
				buf.WriteByte(c)
			}
		}
	default:
		if d.CharsetReader == nil {
			return fmt.Errorf("mime: unhandled charset %q", charset)
		}
		r, err := d.CharsetReader(strings.ToLower(charset), bytes.NewReader(content))
		if err != nil {
			return err
		}
		if _, err = buf.ReadFrom(r); err != nil {
			return err
		}
	}
	return nil
}

// hasNonWhitespace reports whether s (assumed to be ASCII) contains at least
// one byte of non-whitespace.
func hasNonWhitespace(s string) bool {
	for _, b := range s {
		switch b {
		// Encoded-words can only be separated by linear white spaces which does
		// not include vertical tabs (\v).
		case ' ', '\t', '\n', '\r':
		default:
			return true
		}
	}
	return false
}

// qDecode decodes a Q encoded string.
func qDecode(s string) ([]byte, error) {
	dec := make([]byte, len(s))
	n := 0
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == '_':
			dec[n] = ' '
		case c == '=':
			if i+2 >= len(s) {
				return nil, errInvalidWord
			}
			b, err := readHexByte(s[i+1], s[i+2])
			if err != nil {
				return nil, err
			}
			dec[n] = b
			i += 2
		case (c <= '~' && c >= ' ') || c == '\n' || c == '\r' || c == '\t':
			dec[n] = c
		default:
			return nil, errInvalidWord
		}
		n++
	}

	return dec[:n], nil
}

// readHexByte returns the byte from its quoted-printable representation.
func readHexByte(a, b byte) (byte, error) {
	var hb, lb byte
	var err error
	if hb, err = fromHex(a); err != nil {
		return 0, err
	}
	if lb, err = fromHex(b); err != nil {
		return 0, err
	}
	return hb<<4 | lb, nil
}

func fromHex(b byte) (byte, error) {
	switch {
	case b >= '0' && b <= '9':
		return b - '0', nil
	case b >= 'A' && b <= 'F':
		return b - 'A' + 10, nil
	// Accept badly encoded bytes.
	case b >= 'a' && b <= 'f':
		return b - 'a' + 10, nil
	}
	return 0, fmt.Errorf("mime: invalid hex byte %#02x", b)
}

var bufPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

func getBuffer() *bytes.Buffer {
	return bufPool.Get().(*bytes.Buffer)
}

func putBuffer(buf *bytes.Buffer) {
	if buf.Len() > 1024 {
		return
	}
	buf.Reset()
	bufPool.Put(buf)
}
