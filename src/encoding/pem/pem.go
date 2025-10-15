// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pem implements the PEM data encoding, which originated in Privacy
// Enhanced Mail. The most common use of PEM encoding today is in TLS keys and
// certificates. See RFC 1421.
package pem

import (
	"bytes"
	"encoding/base64"
	"errors"
	"io"
	"slices"
	"strings"
)

// A Block represents a PEM encoded structure.
//
// The encoded form is:
//
//	-----BEGIN Type-----
//	Headers
//	base64-encoded Bytes
//	-----END Type-----
//
// where [Block.Headers] is a possibly empty sequence of Key: Value lines.
type Block struct {
	Type    string            // The type, taken from the preamble (i.e. "RSA PRIVATE KEY").
	Headers map[string]string // Optional headers.
	Bytes   []byte            // The decoded bytes of the contents. Typically a DER encoded ASN.1 structure.
}

// getLine results the first \r\n or \n delineated line from the given byte
// array. The line does not include trailing whitespace or the trailing new
// line bytes. The remainder of the byte array (also not including the new line
// bytes) is also returned and this will always be smaller than the original
// argument.
func getLine(data []byte) (line, rest []byte, consumed int) {
	i := bytes.IndexByte(data, '\n')
	var j int
	if i < 0 {
		i = len(data)
		j = i
	} else {
		j = i + 1
		if i > 0 && data[i-1] == '\r' {
			i--
		}
	}
	return bytes.TrimRight(data[0:i], " \t"), data[j:], j
}

// removeSpacesAndTabs returns a copy of its input with all spaces and tabs
// removed, if there were any. Otherwise, the input is returned unchanged.
//
// The base64 decoder already skips newline characters, so we don't need to
// filter them out here.
func removeSpacesAndTabs(data []byte) []byte {
	if !bytes.ContainsAny(data, " \t") {
		// Fast path; most base64 data within PEM contains newlines, but
		// no spaces nor tabs. Skip the extra alloc and work.
		return data
	}
	result := make([]byte, len(data))
	n := 0

	for _, b := range data {
		if b == ' ' || b == '\t' {
			continue
		}
		result[n] = b
		n++
	}

	return result[0:n]
}

var pemStart = []byte("\n-----BEGIN ")
var pemEnd = []byte("\n-----END ")
var pemEndOfLine = []byte("-----")
var colon = []byte(":")

// Decode will find the next PEM formatted block (certificate, private key
// etc) in the input. It returns that block and the remainder of the input. If
// no PEM data is found, p is nil and the whole of the input is returned in
// rest.
func Decode(data []byte) (p *Block, rest []byte) {
	// pemStart begins with a newline. However, at the very beginning of
	// the byte array, we'll accept the start string without it.
	rest = data

	endTrailerIndex := 0
	for {
		// If we've already tried parsing a block, skip past the END we already
		// saw.
		rest = rest[endTrailerIndex:]

		// Find the first END line, and then find the last BEGIN line before
		// the end line. This lets us skip any repeated BEGIN lines that don't
		// have a matching END.
		endIndex := bytes.Index(rest, pemEnd)
		if endIndex < 0 {
			return nil, data
		}
		endTrailerIndex = endIndex + len(pemEnd)
		beginIndex := bytes.LastIndex(rest[:endIndex], pemStart[1:])
		if beginIndex < 0 || (beginIndex > 0 && rest[beginIndex-1] != '\n') {
			continue
		}
		rest = rest[beginIndex+len(pemStart)-1:]
		endIndex -= beginIndex + len(pemStart) - 1
		endTrailerIndex -= beginIndex + len(pemStart) - 1

		var typeLine []byte
		var consumed int
		typeLine, rest, consumed = getLine(rest)
		if !bytes.HasSuffix(typeLine, pemEndOfLine) {
			continue
		}
		endIndex -= consumed
		endTrailerIndex -= consumed
		typeLine = typeLine[0 : len(typeLine)-len(pemEndOfLine)]

		p = &Block{
			Headers: make(map[string]string),
			Type:    string(typeLine),
		}

		for {
			// This loop terminates because getLine's second result is
			// always smaller than its argument.
			if len(rest) == 0 {
				return nil, data
			}
			line, next, consumed := getLine(rest)

			key, val, ok := bytes.Cut(line, colon)
			if !ok {
				break
			}

			// TODO(agl): need to cope with values that spread across lines.
			key = bytes.TrimSpace(key)
			val = bytes.TrimSpace(val)
			p.Headers[string(key)] = string(val)
			rest = next
			endIndex -= consumed
			endTrailerIndex -= consumed
		}

		// If there were headers, there must be a newline between the headers
		// and the END line, so endIndex should be >= 0.
		if len(p.Headers) > 0 && endIndex < 0 {
			continue
		}

		// After the "-----" of the ending line, there should be the same type
		// and then a final five dashes.
		endTrailer := rest[endTrailerIndex:]
		endTrailerLen := len(typeLine) + len(pemEndOfLine)
		if len(endTrailer) < endTrailerLen {
			continue
		}

		restOfEndLine := endTrailer[endTrailerLen:]
		endTrailer = endTrailer[:endTrailerLen]
		if !bytes.HasPrefix(endTrailer, typeLine) ||
			!bytes.HasSuffix(endTrailer, pemEndOfLine) {
			continue
		}

		// The line must end with only whitespace.
		if s, _, _ := getLine(restOfEndLine); len(s) != 0 {
			continue
		}

		p.Bytes = []byte{}
		if endIndex > 0 {
			base64Data := removeSpacesAndTabs(rest[:endIndex])
			p.Bytes = make([]byte, base64.StdEncoding.DecodedLen(len(base64Data)))
			n, err := base64.StdEncoding.Decode(p.Bytes, base64Data)
			if err != nil {
				continue
			}
			p.Bytes = p.Bytes[:n]
		}

		// the -1 is because we might have only matched pemEnd without the
		// leading newline if the PEM block was empty.
		_, rest, _ = getLine(rest[endIndex+len(pemEnd)-1:])
		return p, rest
	}
}

const pemLineLength = 64

type lineBreaker struct {
	line [pemLineLength]byte
	used int
	out  io.Writer
}

var nl = []byte{'\n'}

func (l *lineBreaker) Write(b []byte) (n int, err error) {
	if l.used+len(b) < pemLineLength {
		copy(l.line[l.used:], b)
		l.used += len(b)
		return len(b), nil
	}

	n, err = l.out.Write(l.line[0:l.used])
	if err != nil {
		return
	}
	excess := pemLineLength - l.used
	l.used = 0

	n, err = l.out.Write(b[0:excess])
	if err != nil {
		return
	}

	n, err = l.out.Write(nl)
	if err != nil {
		return
	}

	return l.Write(b[excess:])
}

func (l *lineBreaker) Close() (err error) {
	if l.used > 0 {
		_, err = l.out.Write(l.line[0:l.used])
		if err != nil {
			return
		}
		_, err = l.out.Write(nl)
	}

	return
}

func writeHeader(out io.Writer, k, v string) error {
	_, err := out.Write([]byte(k + ": " + v + "\n"))
	return err
}

// Encode writes the PEM encoding of b to out.
func Encode(out io.Writer, b *Block) error {
	// Check for invalid block before writing any output.
	for k := range b.Headers {
		if strings.Contains(k, ":") {
			return errors.New("pem: cannot encode a header key that contains a colon")
		}
	}

	// All errors below are relayed from underlying io.Writer,
	// so it is now safe to write data.

	if _, err := out.Write(pemStart[1:]); err != nil {
		return err
	}
	if _, err := out.Write([]byte(b.Type + "-----\n")); err != nil {
		return err
	}

	if len(b.Headers) > 0 {
		const procType = "Proc-Type"
		h := make([]string, 0, len(b.Headers))
		hasProcType := false
		for k := range b.Headers {
			if k == procType {
				hasProcType = true
				continue
			}
			h = append(h, k)
		}
		// The Proc-Type header must be written first.
		// See RFC 1421, section 4.6.1.1
		if hasProcType {
			if err := writeHeader(out, procType, b.Headers[procType]); err != nil {
				return err
			}
		}
		// For consistency of output, write other headers sorted by key.
		slices.Sort(h)
		for _, k := range h {
			if err := writeHeader(out, k, b.Headers[k]); err != nil {
				return err
			}
		}
		if _, err := out.Write(nl); err != nil {
			return err
		}
	}

	var breaker lineBreaker
	breaker.out = out

	b64 := base64.NewEncoder(base64.StdEncoding, &breaker)
	if _, err := b64.Write(b.Bytes); err != nil {
		return err
	}
	b64.Close()
	breaker.Close()

	if _, err := out.Write(pemEnd[1:]); err != nil {
		return err
	}
	_, err := out.Write([]byte(b.Type + "-----\n"))
	return err
}

// EncodeToMemory returns the PEM encoding of b.
//
// If b has invalid headers and cannot be encoded,
// EncodeToMemory returns nil. If it is important to
// report details about this error case, use [Encode] instead.
func EncodeToMemory(b *Block) []byte {
	var buf bytes.Buffer
	if err := Encode(&buf, b); err != nil {
		return nil
	}
	return buf.Bytes()
}
