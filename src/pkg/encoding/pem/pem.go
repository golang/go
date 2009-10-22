// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements the PEM data encoding, which originated in Privacy
// Enhanced Mail. The most common use of PEM encoding today is in TLS keys and
// certificates. See RFC 1421.
package pem

import (
	"bytes";
	"encoding/base64";
	"strings";
)

// A Block represents a PEM encoded structure.
//
// The encoded form is:
//    -----BEGIN Type-----
//    Headers
//    base64-encoded Bytes
//    -----END Type-----
// where Headers is a possibly empty sequence of Key: Value lines.
type Block struct {
	Type	string;			// The type, taken from the preamble (i.e. "RSA PRIVATE KEY").
	Headers	map[string]string;	// Optional headers.
	Bytes	[]byte;			// The decoded bytes of the contents. Typically a DER encoded ASN.1 structure.
}

// getLine results the first \r\n or \n delineated line from the given byte
// array. The line does not include the \r\n or \n. The remainder of the byte
// array (also not including the new line bytes) is also returned and this will
// always be smaller than the original argument.
func getLine(data []byte) (line, rest []byte) {
	i := bytes.Index(data, []byte{'\n'});
	var j int;
	if i < 0 {
		i = len(data);
		j = i;
	} else {
		j = i+1;
		if i > 0 && data[i-1] == '\r' {
			i--;
		}
	}
	return data[0:i], data[j:len(data)];
}

// removeWhitespace returns a copy of its input with all spaces, tab and
// newline characters removed.
func removeWhitespace(data []byte) []byte {
	result := make([]byte, len(data));
	n := 0;

	for _, b := range data {
		if b == ' ' || b == '\t' || b == '\r' || b == '\n' {
			continue;
		}
		result[n] = b;
		n++;
	}

	return result[0:n];
}

var pemStart = strings.Bytes("\n-----BEGIN ")
var pemEnd = strings.Bytes("\n-----END ")
var pemEndOfLine = strings.Bytes("-----")

// Decode will find the next PEM formatted block (certificate, private key
// etc) in the input. It returns that block and the remainder of the input. If
// no PEM data is found, p is nil and the whole of the input is returned in
// rest.
func Decode(data []byte) (p *Block, rest []byte) {
	// pemStart begins with a newline. However, at the very beginning of
	// the byte array, we'll accept the start string without it.
	rest = data;
	if bytes.HasPrefix(data, pemStart[1:len(pemStart)]) {
		rest = rest[len(pemStart)-1 : len(data)];
	} else if i := bytes.Index(data, pemStart); i >= 0 {
		rest = rest[i+len(pemStart) : len(data)];
	} else {
		return nil, data;
	}

	typeLine, rest := getLine(rest);
	if !bytes.HasSuffix(typeLine, pemEndOfLine) {
		goto Error;
	}
	typeLine = typeLine[0 : len(typeLine)-len(pemEndOfLine)];

	p = &Block{
		Headers: make(map[string]string),
		Type: string(typeLine),
	};

	for {
		// This loop terminates because getLine's second result is
		// always smaller than it's argument.
		if len(rest) == 0 {
			return nil, data;
		}
		line, next := getLine(rest);

		i := bytes.Index(line, []byte{':'});
		if i == -1 {
			break;
		}

		// TODO(agl): need to cope with values that spread across lines.
		key, val := line[0:i], line[i+1 : len(line)];
		key = bytes.TrimSpace(key);
		val = bytes.TrimSpace(val);
		p.Headers[string(key)] = string(val);
		rest = next;
	}

	i := bytes.Index(rest, pemEnd);
	if i < 0 {
		goto Error;
	}
	base64Data := removeWhitespace(rest[0:i]);

	p.Bytes = make([]byte, base64.StdEncoding.DecodedLen(len(base64Data)));
	n, err := base64.StdEncoding.Decode(base64Data, p.Bytes);
	if err != nil {
		goto Error;
	}
	p.Bytes = p.Bytes[0:n];

	_, rest = getLine(rest[i+len(pemEnd) : len(rest)]);

	return;

Error:
	// If we get here then we have rejected a likely looking, but
	// ultimately invalid PEM block. We need to start over from a new
	// position.  We have consumed the preamble line and will have consumed
	// any lines which could be header lines. However, a valid preamble
	// line is not a valid header line, therefore we cannot have consumed
	// the preamble line for the any subsequent block. Thus, we will always
	// find any valid block, no matter what bytes preceed it.
	//
	// For example, if the input is
	//
	//    -----BEGIN MALFORMED BLOCK-----
	//    junk that may look like header lines
	//   or data lines, but no END line
	//
	//    -----BEGIN ACTUAL BLOCK-----
	//    realdata
	//    -----END ACTUAL BLOCK-----
	//
	// we've failed to parse using the first BEGIN line
	// and now will try again, using the second BEGIN line.
	p, rest = Decode(rest);
	if p == nil {
		rest = data;
	}
	return;
}
