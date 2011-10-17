// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package csv reads and writes comma-separated values (CSV) files.
//
// A csv file contains zero or more records of one or more fields per record.
// Each record is separated by the newline character. The final record may
// optionally be followed by a newline character.
//
//	field1,field2,field3
//
// White space is considered part of a field.
//
// Carriage returns before newline characters are silently removed.
//
// Blank lines are ignored.  A line with only whitespace characters (excluding
// the ending newline character) is not considered a blank line.
//
// Fields which start and stop with the quote character " are called
// quoted-fields.  The beginning and ending quote are not part of the
// field.
//
// The source:
//
//	normal string,"quoted-field"
//
// results in the fields
//
//	{`normal string`, `quoted-field`}
//
// Within a quoted-field a quote character followed by a second quote
// character is considered a single quote.
//
//	"the ""word"" is true","a ""quoted-field"""
//
// results in
//
//	{`the "word" is true`, `a "quoted-field"`}
//
// Newlines and commas may be included in a quoted-field
//
//	"Multi-line
//	field","comma is ,"
//
// results in
//
//	{`Multi-line
//	field`, `comma is ,`}
package csv

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"unicode"
)

// A ParseError is returned for parsing errors.
// The first line is 1.  The first column is 0.
type ParseError struct {
	Line   int      // Line where the error occurred
	Column int      // Column (rune index) where the error occurred
	Error  os.Error // The actual error
}

func (e *ParseError) String() string {
	return fmt.Sprintf("line %d, column %d: %s", e.Line, e.Column, e.Error)
}

// These are the errors that can be returned in ParseError.Error
var (
	ErrTrailingComma = os.NewError("extra delimiter at end of line")
	ErrBareQuote     = os.NewError("bare \" in non-quoted-field")
	ErrQuote         = os.NewError("extraneous \" in field")
	ErrFieldCount    = os.NewError("wrong number of fields in line")
)

// A Reader reads records from a CSV-encoded file.
//
// As returned by NewReader, a Reader expects input conforming to RFC 4180.
// The exported fields can be changed to customize the details before the
// first call to Read or ReadAll.
//
// Comma is the field delimiter.  It defaults to ','.
//
// Comment, if not 0, is the comment character. Lines beginning with the
// Comment character are ignored.
//
// If FieldsPerRecord is positive, Read requires each record to
// have the given number of fields.  If FieldsPerRecord is 0, Read sets it to
// the number of fields in the first record, so that future records must
// have the same field count.
//
// If LazyQuotes is true, a quote may appear in an unquoted field and a
// non-doubled quote may appear in a quoted field.
//
// If TrailingComma is true, the last field may be an unquoted empty field.
//
// If TrimLeadingSpace is true, leading white space in a field is ignored.
type Reader struct {
	Comma            int  // Field delimiter (set to ',' by NewReader)
	Comment          int  // Comment character for start of line
	FieldsPerRecord  int  // Number of expected fields per record
	LazyQuotes       bool // Allow lazy quotes
	TrailingComma    bool // Allow trailing comma
	TrimLeadingSpace bool // Trim leading space
	line             int
	column           int
	r                *bufio.Reader
	field            bytes.Buffer
}

// NewReader returns a new Reader that reads from r.
func NewReader(r io.Reader) *Reader {
	return &Reader{
		Comma: ',',
		r:     bufio.NewReader(r),
	}
}

// error creates a new ParseError based on err.
func (r *Reader) error(err os.Error) os.Error {
	return &ParseError{
		Line:   r.line,
		Column: r.column,
		Error:  err,
	}
}

// Read reads one record from r.  The record is a slice of strings with each
// string representing one field.
func (r *Reader) Read() (record []string, err os.Error) {
	for {
		record, err = r.parseRecord()
		if record != nil {
			break
		}
		if err != nil {
			return nil, err
		}
	}

	if r.FieldsPerRecord > 0 {
		if len(record) != r.FieldsPerRecord {
			r.column = 0 // report at start of record
			return record, r.error(ErrFieldCount)
		}
	} else if r.FieldsPerRecord == 0 {
		r.FieldsPerRecord = len(record)
	}
	return record, nil
}

// ReadAll reads all the remaining records from r.
// Each record is a slice of fields.
func (r *Reader) ReadAll() (records [][]string, err os.Error) {
	for {
		record, err := r.Read()
		if err == os.EOF {
			return records, nil
		}
		if err != nil {
			return nil, err
		}
		records = append(records, record)
	}
	panic("unreachable")
}

// readRune reads one rune from r, folding \r\n to \n and keeping track
// of how far into the line we have read.  r.column will point to the start
// of this rune, not the end of this rune.
func (r *Reader) readRune() (int, os.Error) {
	rune, _, err := r.r.ReadRune()

	// Handle \r\n here.  We make the simplifying assumption that
	// anytime \r is followed by \n that it can be folded to \n.
	// We will not detect files which contain both \r\n and bare \n.
	if rune == '\r' {
		rune, _, err = r.r.ReadRune()
		if err == nil {
			if rune != '\n' {
				r.r.UnreadRune()
				rune = '\r'
			}
		}
	}
	r.column++
	return rune, err
}

// unreadRune puts the last rune read from r back.
func (r *Reader) unreadRune() {
	r.r.UnreadRune()
	r.column--
}

// skip reads runes up to and including the rune delim or until error.
func (r *Reader) skip(delim int) os.Error {
	for {
		rune, err := r.readRune()
		if err != nil {
			return err
		}
		if rune == delim {
			return nil
		}
	}
	panic("unreachable")
}

// parseRecord reads and parses a single csv record from r.
func (r *Reader) parseRecord() (fields []string, err os.Error) {
	// Each record starts on a new line.  We increment our line
	// number (lines start at 1, not 0) and set column to -1
	// so as we increment in readRune it points to the character we read.
	r.line++
	r.column = -1

	// Peek at the first rune.  If it is an error we are done.
	// If we are support comments and it is the comment character
	// then skip to the end of line.

	rune, _, err := r.r.ReadRune()
	if err != nil {
		return nil, err
	}

	if r.Comment != 0 && rune == r.Comment {
		return nil, r.skip('\n')
	}
	r.r.UnreadRune()

	// At this point we have at least one field.
	for {
		haveField, delim, err := r.parseField()
		if haveField {
			fields = append(fields, r.field.String())
		}
		if delim == '\n' || err == os.EOF {
			return fields, err
		} else if err != nil {
			return nil, err
		}
	}
	panic("unreachable")
}

// parseField parses the next field in the record.  The read field is
// located in r.field.  Delim is the first character not part of the field
// (r.Comma or '\n').
func (r *Reader) parseField() (haveField bool, delim int, err os.Error) {
	r.field.Reset()

	rune, err := r.readRune()
	if err != nil {
		// If we have EOF and are not at the start of a line
		// then we return the empty field.  We have already
		// checked for trailing commas if needed.
		if err == os.EOF && r.column != 0 {
			return true, 0, err
		}
		return false, 0, err
	}

	if r.TrimLeadingSpace {
		for rune != '\n' && unicode.IsSpace(rune) {
			rune, err = r.readRune()
			if err != nil {
				return false, 0, err
			}
		}
	}

	switch rune {
	case r.Comma:
		// will check below

	case '\n':
		// We are a trailing empty field or a blank line
		if r.column == 0 {
			return false, rune, nil
		}
		return true, rune, nil

	case '"':
		// quoted field
	Quoted:
		for {
			rune, err = r.readRune()
			if err != nil {
				if err == os.EOF {
					if r.LazyQuotes {
						return true, 0, err
					}
					return false, 0, r.error(ErrQuote)
				}
				return false, 0, err
			}
			switch rune {
			case '"':
				rune, err = r.readRune()
				if err != nil || rune == r.Comma {
					break Quoted
				}
				if rune == '\n' {
					return true, rune, nil
				}
				if rune != '"' {
					if !r.LazyQuotes {
						r.column--
						return false, 0, r.error(ErrQuote)
					}
					// accept the bare quote
					r.field.WriteRune('"')
				}
			case '\n':
				r.line++
				r.column = -1
			}
			r.field.WriteRune(rune)
		}

	default:
		// unquoted field
		for {
			r.field.WriteRune(rune)
			rune, err = r.readRune()
			if err != nil || rune == r.Comma {
				break
			}
			if rune == '\n' {
				return true, rune, nil
			}
			if !r.LazyQuotes && rune == '"' {
				return false, 0, r.error(ErrBareQuote)
			}
		}
	}

	if err != nil {
		if err == os.EOF {
			return true, 0, err
		}
		return false, 0, err
	}

	if !r.TrailingComma {
		// We don't allow trailing commas.  See if we
		// are at the end of the line (being mindful
		// of trimming spaces).
		c := r.column
		rune, err = r.readRune()
		if r.TrimLeadingSpace {
			for rune != '\n' && unicode.IsSpace(rune) {
				rune, err = r.readRune()
				if err != nil {
					break
				}
			}
		}
		if err == os.EOF || rune == '\n' {
			r.column = c // report the comma
			return false, 0, r.error(ErrTrailingComma)
		}
		r.unreadRune()
	}
	return true, rune, nil
}
