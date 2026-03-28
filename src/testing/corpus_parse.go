// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
)

// encVersion1 is the first line of a Go corpus file with version 1 encoding.
const encVersion1 = "go test fuzz v1"

// parseGoCorpusFile parses a Go corpus file and returns the values.
// The format is:
//
//	go test fuzz v1
//	type(value)
//	type(value)
//	...
func parseGoCorpusFile(data []byte, types []reflect.Type) ([]any, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("cannot parse empty corpus file")
	}
	lines := bytes.Split(data, []byte("\n"))
	if len(lines) < 2 {
		return nil, fmt.Errorf("must include version and at least one value")
	}
	version := strings.TrimSuffix(string(lines[0]), "\r")
	if version != encVersion1 {
		return nil, fmt.Errorf("unknown encoding version: %s", version)
	}

	var vals []any
	for _, line := range lines[1:] {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		v, err := parseCorpusValue(line)
		if err != nil {
			return nil, fmt.Errorf("malformed line %q: %v", line, err)
		}
		vals = append(vals, v)
	}

	// Check that types match
	if len(vals) != len(types) {
		return nil, fmt.Errorf("wrong number of values: got %d, want %d", len(vals), len(types))
	}
	for i, v := range vals {
		if reflect.TypeOf(v) != types[i] {
			return nil, fmt.Errorf("type mismatch at arg %d: got %T, want %v", i, v, types[i])
		}
	}

	return vals, nil
}

// parseCorpusValue parses a single value from a corpus file line.
// The format is: type(literal) where type is a Go type name.
// This is a simplified parser that doesn't use go/parser to avoid import cycles.
func parseCorpusValue(line []byte) (any, error) {
	s := string(line)

	// Find the opening paren to split type and value
	parenIdx := strings.Index(s, "(")
	if parenIdx == -1 {
		return nil, fmt.Errorf("expected type(value) format")
	}
	if !strings.HasSuffix(s, ")") {
		return nil, fmt.Errorf("expected closing paren")
	}

	typeName := s[:parenIdx]
	literal := s[parenIdx+1 : len(s)-1]

	// Handle []byte type
	if typeName == "[]byte" {
		str, err := strconv.Unquote(literal)
		if err != nil {
			return nil, fmt.Errorf("invalid string literal for []byte: %v", err)
		}
		return []byte(str), nil
	}

	// Handle math.Float32frombits and math.Float64frombits
	if typeName == "math.Float32frombits" {
		bits, err := strconv.ParseUint(literal, 0, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid uint32 for math.Float32frombits: %v", err)
		}
		return math.Float32frombits(uint32(bits)), nil
	}
	if typeName == "math.Float64frombits" {
		bits, err := strconv.ParseUint(literal, 0, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid uint64 for math.Float64frombits: %v", err)
		}
		return math.Float64frombits(bits), nil
	}

	// Handle primitive types
	switch typeName {
	case "bool":
		switch literal {
		case "true":
			return true, nil
		case "false":
			return false, nil
		default:
			return nil, fmt.Errorf("invalid bool: %s", literal)
		}

	case "string":
		return strconv.Unquote(literal)

	case "byte":
		// Can be character literal like 'a' or integer like 65
		if len(literal) > 0 && literal[0] == '\'' {
			r, err := parseCharLiteral(literal)
			if err != nil {
				return nil, err
			}
			if r >= 256 {
				return nil, fmt.Errorf("can only encode single byte to a byte type")
			}
			return byte(r), nil
		}
		i, err := strconv.ParseUint(literal, 0, 8)
		return uint8(i), err

	case "rune":
		// Can be character literal like '★' or integer like 9733
		if len(literal) > 0 && literal[0] == '\'' {
			return parseCharLiteral(literal)
		}
		i, err := strconv.ParseInt(literal, 0, 32)
		return int32(i), err

	case "int":
		i, err := strconv.ParseInt(literal, 0, 64)
		return int(i), err
	case "int8":
		i, err := strconv.ParseInt(literal, 0, 8)
		return int8(i), err
	case "int16":
		i, err := strconv.ParseInt(literal, 0, 16)
		return int16(i), err
	case "int32":
		i, err := strconv.ParseInt(literal, 0, 32)
		return int32(i), err
	case "int64":
		return strconv.ParseInt(literal, 0, 64)

	case "uint":
		i, err := strconv.ParseUint(literal, 0, 64)
		return uint(i), err
	case "uint8":
		i, err := strconv.ParseUint(literal, 0, 8)
		return uint8(i), err
	case "uint16":
		i, err := strconv.ParseUint(literal, 0, 16)
		return uint16(i), err
	case "uint32":
		i, err := strconv.ParseUint(literal, 0, 32)
		return uint32(i), err
	case "uint64":
		return strconv.ParseUint(literal, 0, 64)

	case "float32":
		// Handle special values: NaN, +Inf, -Inf
		switch literal {
		case "NaN":
			return float32(math.NaN()), nil
		case "+Inf":
			return float32(math.Inf(1)), nil
		case "-Inf":
			return float32(math.Inf(-1)), nil
		}
		v, err := strconv.ParseFloat(literal, 32)
		return float32(v), err

	case "float64":
		// Handle special values: NaN, +Inf, -Inf
		switch literal {
		case "NaN":
			return math.NaN(), nil
		case "+Inf":
			return math.Inf(1), nil
		case "-Inf":
			return math.Inf(-1), nil
		}
		return strconv.ParseFloat(literal, 64)

	default:
		return nil, fmt.Errorf("unsupported type: %s", typeName)
	}
}

// parseCharLiteral parses a Go character literal like 'a' or '\n' or '\x00'.
func parseCharLiteral(s string) (rune, error) {
	if len(s) < 2 || s[0] != '\'' || s[len(s)-1] != '\'' {
		return 0, fmt.Errorf("malformed character literal")
	}
	// Use strconv.UnquoteChar to handle escape sequences
	r, _, _, err := strconv.UnquoteChar(s[1:len(s)-1], '\'')
	return r, err
}

// marshalGoCorpusFile encodes typed Go values into the Go corpus file format.
// This is the inverse of parseGoCorpusFile.
//
// The format is:
//
//	go test fuzz v1
//	type(value)
//	type(value)
//	...
func marshalGoCorpusFile(vals []any) []byte {
	if len(vals) == 0 {
		panic("must have at least one value to marshal")
	}
	var buf bytes.Buffer
	buf.WriteString(encVersion1)
	buf.WriteByte('\n')

	for _, val := range vals {
		switch v := val.(type) {
		case bool:
			fmt.Fprintf(&buf, "bool(%v)\n", v)
		case int:
			fmt.Fprintf(&buf, "int(%d)\n", v)
		case int8:
			fmt.Fprintf(&buf, "int8(%d)\n", v)
		case int16:
			fmt.Fprintf(&buf, "int16(%d)\n", v)
		case int32:
			// Distinguish rune from int32: use rune syntax for valid UTF-8 runes
			if v >= 0 && v <= 0x10FFFF && (v < 0xD800 || v > 0xDFFF) {
				fmt.Fprintf(&buf, "rune(%q)\n", v)
			} else {
				fmt.Fprintf(&buf, "int32(%d)\n", v)
			}
		case int64:
			fmt.Fprintf(&buf, "int64(%d)\n", v)
		case uint:
			fmt.Fprintf(&buf, "uint(%d)\n", v)
		case uint8:
			// Use byte with character syntax for printable ASCII
			fmt.Fprintf(&buf, "byte(%q)\n", v)
		case uint16:
			fmt.Fprintf(&buf, "uint16(%d)\n", v)
		case uint32:
			fmt.Fprintf(&buf, "uint32(%d)\n", v)
		case uint64:
			fmt.Fprintf(&buf, "uint64(%d)\n", v)
		case float32:
			if math.IsNaN(float64(v)) {
				// Check for non-standard NaN representations
				if math.Float32bits(v) != math.Float32bits(float32(math.NaN())) {
					fmt.Fprintf(&buf, "math.Float32frombits(0x%x)\n", math.Float32bits(v))
				} else {
					buf.WriteString("float32(NaN)\n")
				}
			} else if math.IsInf(float64(v), 1) {
				buf.WriteString("float32(+Inf)\n")
			} else if math.IsInf(float64(v), -1) {
				buf.WriteString("float32(-Inf)\n")
			} else {
				fmt.Fprintf(&buf, "float32(%v)\n", v)
			}
		case float64:
			if math.IsNaN(v) {
				// Check for non-standard NaN representations
				if math.Float64bits(v) != math.Float64bits(math.NaN()) {
					fmt.Fprintf(&buf, "math.Float64frombits(0x%x)\n", math.Float64bits(v))
				} else {
					buf.WriteString("float64(NaN)\n")
				}
			} else if math.IsInf(v, 1) {
				buf.WriteString("float64(+Inf)\n")
			} else if math.IsInf(v, -1) {
				buf.WriteString("float64(-Inf)\n")
			} else {
				fmt.Fprintf(&buf, "float64(%v)\n", v)
			}
		case string:
			fmt.Fprintf(&buf, "string(%q)\n", v)
		case []byte:
			fmt.Fprintf(&buf, "[]byte(%q)\n", v)
		default:
			panic(fmt.Sprintf("unsupported type for corpus marshaling: %T", val))
		}
	}

	return buf.Bytes()
}
