// Code generated from gen_encoding_table.go using 'go generate'; DO NOT EDIT.

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package url

type encoding uint8

const (
	encodePath encoding = 1 << iota
	encodePathSegment
	encodeHost
	encodeZone
	encodeUserPassword
	encodeQueryComponent
	encodeFragment

	// hexChar is actually NOT an encoding mode, but there are only seven
	// encoding modes. We might as well abuse the otherwise unused most
	// significant bit in uint8 to indicate whether a character is
	// hexadecimal.
	hexChar
)

var table = [256]encoding{
	'!':  encodeFragment | encodeZone | encodeHost,
	'"':  encodeZone | encodeHost,
	'$':  encodeFragment | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'&':  encodeFragment | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'\'': encodeZone | encodeHost,
	'(':  encodeFragment | encodeZone | encodeHost,
	')':  encodeFragment | encodeZone | encodeHost,
	'*':  encodeFragment | encodeZone | encodeHost,
	'+':  encodeFragment | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	',':  encodeFragment | encodeUserPassword | encodeZone | encodeHost | encodePath,
	'-':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'.':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'/':  encodeFragment | encodePath,
	'0':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'1':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'2':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'3':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'4':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'5':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'6':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'7':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'8':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'9':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	':':  encodeFragment | encodeZone | encodeHost | encodePathSegment | encodePath,
	';':  encodeFragment | encodeUserPassword | encodeZone | encodeHost | encodePath,
	'<':  encodeZone | encodeHost,
	'=':  encodeFragment | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'>':  encodeZone | encodeHost,
	'?':  encodeFragment,
	'@':  encodeFragment | encodePathSegment | encodePath,
	'A':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'B':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'C':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'D':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'E':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'F':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'G':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'H':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'I':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'J':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'K':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'L':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'M':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'N':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'O':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'P':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'Q':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'R':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'S':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'T':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'U':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'V':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'W':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'X':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'Y':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'Z':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'[':  encodeZone | encodeHost,
	']':  encodeZone | encodeHost,
	'_':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'a':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'b':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'c':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'd':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'e':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'f':  hexChar | encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'g':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'h':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'i':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'j':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'k':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'l':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'm':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'n':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'o':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'p':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'q':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'r':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	's':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	't':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'u':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'v':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'w':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'x':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'y':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'z':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
	'~':  encodeFragment | encodeQueryComponent | encodeUserPassword | encodeZone | encodeHost | encodePathSegment | encodePath,
}
