// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"encoding/binary"
)

// Content-type sniffing algorithm.
// References in this file refer to this draft specification:
//   http://mimesniff.spec.whatwg.org/

// The algorithm prefers to use sniffLen bytes to make its decision.
const sniffLen = 512

// DetectContentType returns the sniffed Content-Type string
// for the given data. This function always returns a valid MIME type.
func DetectContentType(data []byte) string {
	if len(data) > sniffLen {
		data = data[:sniffLen]
	}

	// Index of the first non-whitespace byte in data.
	firstNonWS := 0
	for ; firstNonWS < len(data) && isWS(data[firstNonWS]); firstNonWS++ {
	}

	for _, sig := range sniffSignatures {
		if ct := sig.match(data, firstNonWS); ct != "" {
			return ct
		}
	}

	return "application/octet-stream" // fallback
}

func isWS(b byte) bool {
	return bytes.IndexByte([]byte("\t\n\x0C\r "), b) != -1
}

type sniffSig interface {
	// match returns the MIME type of the data, or "" if unknown.
	match(data []byte, firstNonWS int) string
}

// Data matching the table in section 6.
var sniffSignatures = []sniffSig{
	htmlSig([]byte("<!DOCTYPE HTML")),
	htmlSig([]byte("<HTML")),
	htmlSig([]byte("<HEAD")),
	htmlSig([]byte("<SCRIPT")),
	htmlSig([]byte("<IFRAME")),
	htmlSig([]byte("<H1")),
	htmlSig([]byte("<DIV")),
	htmlSig([]byte("<FONT")),
	htmlSig([]byte("<TABLE")),
	htmlSig([]byte("<A")),
	htmlSig([]byte("<STYLE")),
	htmlSig([]byte("<TITLE")),
	htmlSig([]byte("<B")),
	htmlSig([]byte("<BODY")),
	htmlSig([]byte("<BR")),
	htmlSig([]byte("<P")),
	htmlSig([]byte("<!--")),

	&maskedSig{mask: []byte("\xFF\xFF\xFF\xFF\xFF"), pat: []byte("<?xml"), skipWS: true, ct: "text/xml; charset=utf-8"},

	&exactSig{[]byte("%PDF-"), "application/pdf"},
	&exactSig{[]byte("%!PS-Adobe-"), "application/postscript"},

	// UTF BOMs.
	&maskedSig{mask: []byte("\xFF\xFF\x00\x00"), pat: []byte("\xFE\xFF\x00\x00"), ct: "text/plain; charset=utf-16be"},
	&maskedSig{mask: []byte("\xFF\xFF\x00\x00"), pat: []byte("\xFF\xFE\x00\x00"), ct: "text/plain; charset=utf-16le"},
	&maskedSig{mask: []byte("\xFF\xFF\xFF\x00"), pat: []byte("\xEF\xBB\xBF\x00"), ct: "text/plain; charset=utf-8"},

	&exactSig{[]byte("GIF87a"), "image/gif"},
	&exactSig{[]byte("GIF89a"), "image/gif"},
	&exactSig{[]byte("\x89\x50\x4E\x47\x0D\x0A\x1A\x0A"), "image/png"},
	&exactSig{[]byte("\xFF\xD8\xFF"), "image/jpeg"},
	&exactSig{[]byte("BM"), "image/bmp"},
	&maskedSig{
		mask: []byte("\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\xFF\xFF"),
		pat:  []byte("RIFF\x00\x00\x00\x00WEBPVP"),
		ct:   "image/webp",
	},
	&exactSig{[]byte("\x00\x00\x01\x00"), "image/vnd.microsoft.icon"},
	&exactSig{[]byte("\x4F\x67\x67\x53\x00"), "application/ogg"},
	&maskedSig{
		mask: []byte("\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF"),
		pat:  []byte("RIFF\x00\x00\x00\x00WAVE"),
		ct:   "audio/wave",
	},
	&exactSig{[]byte("\x1A\x45\xDF\xA3"), "video/webm"},
	&exactSig{[]byte("\x52\x61\x72\x20\x1A\x07\x00"), "application/x-rar-compressed"},
	&exactSig{[]byte("\x50\x4B\x03\x04"), "application/zip"},
	&exactSig{[]byte("\x1F\x8B\x08"), "application/x-gzip"},

	// TODO(dsymonds): Re-enable this when the spec is sorted w.r.t. MP4.
	//mp4Sig(0),

	textSig(0), // should be last
}

type exactSig struct {
	sig []byte
	ct  string
}

func (e *exactSig) match(data []byte, firstNonWS int) string {
	if bytes.HasPrefix(data, e.sig) {
		return e.ct
	}
	return ""
}

type maskedSig struct {
	mask, pat []byte
	skipWS    bool
	ct        string
}

func (m *maskedSig) match(data []byte, firstNonWS int) string {
	if m.skipWS {
		data = data[firstNonWS:]
	}
	if len(data) < len(m.mask) {
		return ""
	}
	for i, mask := range m.mask {
		db := data[i] & mask
		if db != m.pat[i] {
			return ""
		}
	}
	return m.ct
}

type htmlSig []byte

func (h htmlSig) match(data []byte, firstNonWS int) string {
	data = data[firstNonWS:]
	if len(data) < len(h)+1 {
		return ""
	}
	for i, b := range h {
		db := data[i]
		if 'A' <= b && b <= 'Z' {
			db &= 0xDF
		}
		if b != db {
			return ""
		}
	}
	// Next byte must be space or right angle bracket.
	if db := data[len(h)]; db != ' ' && db != '>' {
		return ""
	}
	return "text/html; charset=utf-8"
}

type mp4Sig int

func (mp4Sig) match(data []byte, firstNonWS int) string {
	// c.f. section 6.1.
	if len(data) < 8 {
		return ""
	}
	boxSize := int(binary.BigEndian.Uint32(data[:4]))
	if boxSize%4 != 0 || len(data) < boxSize {
		return ""
	}
	if !bytes.Equal(data[4:8], []byte("ftyp")) {
		return ""
	}
	for st := 8; st < boxSize; st += 4 {
		if st == 12 {
			// minor version number
			continue
		}
		seg := string(data[st : st+3])
		switch seg {
		case "mp4", "iso", "M4V", "M4P", "M4B":
			return "video/mp4"
			/* The remainder are not in the spec.
			case "M4A":
				return "audio/mp4"
			case "3gp":
				return "video/3gpp"
			case "jp2":
				return "image/jp2" // JPEG 2000
			*/
		}
	}
	return ""
}

type textSig int

func (textSig) match(data []byte, firstNonWS int) string {
	// c.f. section 5, step 4.
	for _, b := range data[firstNonWS:] {
		switch {
		case 0x00 <= b && b <= 0x08,
			b == 0x0B,
			0x0E <= b && b <= 0x1A,
			0x1C <= b && b <= 0x1F:
			return ""
		}
	}
	return "text/plain; charset=utf-8"
}
