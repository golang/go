// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fmtstr defines a parser for format strings as used by [fmt.Printf].
package fmtstr

import (
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Operation holds the parsed representation of a printf operation such as "%3.*[4]d".
// It is constructed by [Parse].
type Operation struct {
	Text  string // full text of the operation, e.g. "%[2]*.3d"
	Verb  Verb   // verb specifier, guaranteed to exist, e.g., 'd' in '%[1]d'
	Range Range  // the range of Text within the overall format string
	Flags string // formatting flags, e.g. "-0"
	Width Size   // width specifier, e.g., '3' in '%3d'
	Prec  Size   // precision specifier, e.g., '.4' in '%.4f'
}

// Size describes an optional width or precision in a format operation.
// It may represent no value, a literal number, an asterisk, or an indexed asterisk.
type Size struct {
	// At most one of these two fields is non-negative.
	Fixed   int // e.g. 4 from "%4d", otherwise -1
	Dynamic int // index of argument providing dynamic size (e.g. %*d or %[3]*d), otherwise -1

	Index int   // If the width or precision uses an indexed argument (e.g. 2 in %[2]*d), this is the index, otherwise -1
	Range Range // position of the size specifier within the operation
}

// Verb represents the verb character of a format operation (e.g., 'd', 's', 'f').
// It also includes positional information and any explicit argument indexing.
type Verb struct {
	Verb     rune
	Range    Range // positional range of the verb in the format string
	Index    int   // index of an indexed argument, (e.g. 2 in %[2]d), otherwise -1
	ArgIndex int   // argument index (0-based) associated with this verb, relative to CallExpr
}

// byte offsets of format string
type Range struct {
	Start, End int
}

// Parse takes a format string and its index in the printf-like call,
// parses out all format operations, returns a slice of parsed
// [Operation] which describes flags, width, precision, verb, and argument indexing,
// or an error if parsing fails.
//
// All error messages are in predicate form ("call has a problem")
// so that they may be affixed into a subject ("log.Printf ").
//
// The flags will only be a subset of ['#', '0', '+', '-', ' '].
// It does not perform any validation of verbs, nor the
// existence of corresponding arguments (obviously it can't). The provided format string may differ
// from the one in CallExpr, such as a concatenated string or a string
// referred to by the argument in the CallExpr.
func Parse(format string, idx int) ([]*Operation, error) {
	if !strings.Contains(format, "%") {
		return nil, fmt.Errorf("call has arguments but no formatting directives")
	}

	firstArg := idx + 1 // Arguments are immediately after format string.
	argNum := firstArg
	var operations []*Operation
	for i, w := 0, 0; i < len(format); i += w {
		w = 1
		if format[i] != '%' {
			continue
		}
		state, err := parseOperation(format[i:], firstArg, argNum)
		if err != nil {
			return nil, err
		}

		state.operation.addOffset(i)
		operations = append(operations, state.operation)

		w = len(state.operation.Text)
		// Do not waste an argument for '%'.
		if state.operation.Verb.Verb != '%' {
			argNum = state.argNum + 1
		}
	}
	return operations, nil
}

// Internal parsing state to operation.
type state struct {
	operation    *Operation
	firstArg     int  // index of the first argument after the format string
	argNum       int  // which argument we're expecting to format now
	hasIndex     bool // whether the argument is indexed
	index        int  // the encountered index
	indexPos     int  // the encountered index's offset
	indexPending bool // whether we have an indexed argument that has not resolved
	nbytes       int  // number of bytes of the format string consumed
}

// parseOperation parses one format operation starting at the given substring `format`,
// which should begin with '%'. It returns a fully populated state or an error
// if the operation is malformed. The firstArg and argNum parameters help determine how
// arguments map to this operation.
//
// Parse sequence: '%' -> flags -> {[N]* or width} -> .{[N]* or precision} -> [N] -> verb.
func parseOperation(format string, firstArg, argNum int) (*state, error) {
	state := &state{
		operation: &Operation{
			Text: format,
			Width: Size{
				Fixed:   -1,
				Dynamic: -1,
				Index:   -1,
			},
			Prec: Size{
				Fixed:   -1,
				Dynamic: -1,
				Index:   -1,
			},
		},
		firstArg:     firstArg,
		argNum:       argNum,
		hasIndex:     false,
		index:        0,
		indexPos:     0,
		indexPending: false,
		nbytes:       len("%"), // There's guaranteed to be a percent sign.
	}
	// There may be flags.
	state.parseFlags()
	// There may be an index.
	if err := state.parseIndex(); err != nil {
		return nil, err
	}
	// There may be a width.
	state.parseSize(Width)
	// There may be a precision.
	if err := state.parsePrecision(); err != nil {
		return nil, err
	}
	// Now a verb, possibly prefixed by an index (which we may already have).
	if !state.indexPending {
		if err := state.parseIndex(); err != nil {
			return nil, err
		}
	}
	if state.nbytes == len(state.operation.Text) {
		return nil, fmt.Errorf("format %s is missing verb at end of string", state.operation.Text)
	}
	verb, w := utf8.DecodeRuneInString(state.operation.Text[state.nbytes:])

	// Ensure there must be a verb.
	if state.indexPending {
		state.operation.Verb = Verb{
			Verb: verb,
			Range: Range{
				Start: state.indexPos,
				End:   state.nbytes + w,
			},
			Index:    state.index,
			ArgIndex: state.argNum,
		}
	} else {
		state.operation.Verb = Verb{
			Verb: verb,
			Range: Range{
				Start: state.nbytes,
				End:   state.nbytes + w,
			},
			Index:    -1,
			ArgIndex: state.argNum,
		}
	}

	state.nbytes += w
	state.operation.Text = state.operation.Text[:state.nbytes]
	return state, nil
}

// addOffset adjusts the recorded positions in Verb, Width, Prec, and the
// operation's overall Range to be relative to the position in the full format string.
func (s *Operation) addOffset(parsedLen int) {
	s.Verb.Range.Start += parsedLen
	s.Verb.Range.End += parsedLen

	s.Range.Start = parsedLen
	s.Range.End = s.Verb.Range.End

	// one of Fixed or Dynamic is non-negative means existence.
	if s.Prec.Fixed != -1 || s.Prec.Dynamic != -1 {
		s.Prec.Range.Start += parsedLen
		s.Prec.Range.End += parsedLen
	}
	if s.Width.Fixed != -1 || s.Width.Dynamic != -1 {
		s.Width.Range.Start += parsedLen
		s.Width.Range.End += parsedLen
	}
}

// parseFlags accepts any printf flags.
func (s *state) parseFlags() {
	s.operation.Flags = prefixOf(s.operation.Text[s.nbytes:], "#0+- ")
	s.nbytes += len(s.operation.Flags)
}

// prefixOf returns the prefix of s composed only of runes from the specified set.
func prefixOf(s, set string) string {
	rest := strings.TrimLeft(s, set)
	return s[:len(s)-len(rest)]
}

// parseIndex parses an argument index of the form "[n]" that can appear
// in a printf operation (e.g., "%[2]d"). Returns an error if syntax is
// malformed or index is invalid.
func (s *state) parseIndex() error {
	if s.nbytes == len(s.operation.Text) || s.operation.Text[s.nbytes] != '[' {
		return nil
	}
	// Argument index present.
	s.nbytes++ // skip '['
	start := s.nbytes
	if num, ok := s.scanNum(); ok {
		// Later consumed/stored by a '*' or verb.
		s.index = num
		s.indexPos = start - 1
	}

	ok := true
	if s.nbytes == len(s.operation.Text) || s.nbytes == start || s.operation.Text[s.nbytes] != ']' {
		ok = false // syntax error is either missing "]" or invalid index.
		s.nbytes = strings.Index(s.operation.Text[start:], "]")
		if s.nbytes < 0 {
			return fmt.Errorf("format %s is missing closing ]", s.operation.Text)
		}
		s.nbytes = s.nbytes + start
	}
	arg32, err := strconv.ParseInt(s.operation.Text[start:s.nbytes], 10, 32)
	if err != nil || !ok || arg32 <= 0 {
		return fmt.Errorf("format has invalid argument index [%s]", s.operation.Text[start:s.nbytes])
	}

	s.nbytes++ // skip ']'
	arg := int(arg32)
	arg += s.firstArg - 1 // We want to zero-index the actual arguments.
	s.argNum = arg
	s.hasIndex = true
	s.indexPending = true
	return nil
}

// scanNum advances through a decimal number if present, which represents a [Size] or [Index].
func (s *state) scanNum() (int, bool) {
	start := s.nbytes
	for ; s.nbytes < len(s.operation.Text); s.nbytes++ {
		c := s.operation.Text[s.nbytes]
		if c < '0' || '9' < c {
			if start < s.nbytes {
				num, _ := strconv.ParseInt(s.operation.Text[start:s.nbytes], 10, 32)
				return int(num), true
			} else {
				return 0, false
			}
		}
	}
	return 0, false
}

type sizeType int

const (
	Width sizeType = iota
	Precision
)

// parseSize parses a width or precision specifier. It handles literal numeric
// values (e.g., "%3d"), asterisk values (e.g., "%*d"), or indexed asterisk values (e.g., "%[2]*d").
func (s *state) parseSize(kind sizeType) {
	if s.nbytes < len(s.operation.Text) && s.operation.Text[s.nbytes] == '*' {
		s.nbytes++
		if s.indexPending {
			// Absorb it.
			s.indexPending = false
			size := Size{
				Fixed:   -1,
				Dynamic: s.argNum,
				Index:   s.index,
				Range: Range{
					Start: s.indexPos,
					End:   s.nbytes,
				},
			}
			switch kind {
			case Width:
				s.operation.Width = size
			case Precision:
				// Include the leading '.'.
				size.Range.Start -= len(".")
				s.operation.Prec = size
			default:
				panic(kind)
			}
		} else {
			// Non-indexed asterisk: "%*d".
			size := Size{
				Dynamic: s.argNum,
				Index:   -1,
				Fixed:   -1,
				Range: Range{
					Start: s.nbytes - 1,
					End:   s.nbytes,
				},
			}
			switch kind {
			case Width:
				s.operation.Width = size
			case Precision:
				// For precision, include the '.' in the range.
				size.Range.Start -= 1
				s.operation.Prec = size
			default:
				panic(kind)
			}
		}
		s.argNum++
	} else { // Literal number, e.g. "%10d"
		start := s.nbytes
		if num, ok := s.scanNum(); ok {
			size := Size{
				Fixed:   num,
				Index:   -1,
				Dynamic: -1,
				Range: Range{
					Start: start,
					End:   s.nbytes,
				},
			}
			switch kind {
			case Width:
				s.operation.Width = size
			case Precision:
				// Include the leading '.'.
				size.Range.Start -= 1
				s.operation.Prec = size
			default:
				panic(kind)
			}
		}
	}
}

// parsePrecision checks if there's a precision specified after a '.' character.
// If found, it may also parse an index or an asterisk. Returns an error if any index
// parsing fails.
func (s *state) parsePrecision() error {
	// If there's a period, there may be a precision.
	if s.nbytes < len(s.operation.Text) && s.operation.Text[s.nbytes] == '.' {
		s.nbytes++
		if err := s.parseIndex(); err != nil {
			return err
		}
		s.parseSize(Precision)
	}
	return nil
}
