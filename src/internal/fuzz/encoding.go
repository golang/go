// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"strconv"
	"unicode/utf8"
)

// encVersion1 will be the first line of a file with version 1 encoding.
var encVersion1 = "go test fuzz v1"

// marshalCorpusFile encodes an arbitrary number of arguments into the file format for the
// corpus.
func marshalCorpusFile(vals ...any) []byte {
	if len(vals) == 0 {
		panic("must have at least one value to marshal")
	}
	b := bytes.NewBuffer([]byte(encVersion1 + "\n"))
	// TODO(katiehockman): keep uint8 and int32 encoding where applicable,
	// instead of changing to byte and rune respectively.
	for _, val := range vals {
		switch t := val.(type) {
		case int, int8, int16, int64, uint, uint16, uint32, uint64, bool:
			fmt.Fprintf(b, "%T(%v)\n", t, t)
		case float32:
			if math.IsNaN(float64(t)) && math.Float32bits(t) != math.Float32bits(float32(math.NaN())) {
				// We encode unusual NaNs as hex values, because that is how users are
				// likely to encounter them in literature about floating-point encoding.
				// This allows us to reproduce fuzz failures that depend on the specific
				// NaN representation (for float32 there are about 2^24 possibilities!),
				// not just the fact that the value is *a* NaN.
				//
				// Note that the specific value of float32(math.NaN()) can vary based on
				// whether the architecture represents signaling NaNs using a low bit
				// (as is common) or a high bit (as commonly implemented on MIPS
				// hardware before around 2012). We believe that the increase in clarity
				// from identifying "NaN" with math.NaN() is worth the slight ambiguity
				// from a platform-dependent value.
				fmt.Fprintf(b, "math.Float32frombits(0x%x)\n", math.Float32bits(t))
			} else {
				// We encode all other values — including the NaN value that is
				// bitwise-identical to float32(math.Nan()) — using the default
				// formatting, which is equivalent to strconv.FormatFloat with format
				// 'g' and can be parsed by strconv.ParseFloat.
				//
				// For an ordinary floating-point number this format includes
				// sufficiently many digits to reconstruct the exact value. For positive
				// or negative infinity it is the string "+Inf" or "-Inf". For positive
				// or negative zero it is "0" or "-0". For NaN, it is the string "NaN".
				fmt.Fprintf(b, "%T(%v)\n", t, t)
			}
		case float64:
			if math.IsNaN(t) && math.Float64bits(t) != math.Float64bits(math.NaN()) {
				fmt.Fprintf(b, "math.Float64frombits(0x%x)\n", math.Float64bits(t))
			} else {
				fmt.Fprintf(b, "%T(%v)\n", t, t)
			}
		case string:
			fmt.Fprintf(b, "string(%q)\n", t)
		case rune: // int32
			// Although rune and int32 are represented by the same type, only a subset
			// of valid int32 values can be expressed as rune literals. Notably,
			// negative numbers, surrogate halves, and values above unicode.MaxRune
			// have no quoted representation.
			//
			// fmt with "%q" (and the corresponding functions in the strconv package)
			// would quote out-of-range values to the Unicode replacement character
			// instead of the original value (see https://go.dev/issue/51526), so
			// they must be treated as int32 instead.
			//
			// We arbitrarily draw the line at UTF-8 validity, which biases toward the
			// "rune" interpretation. (However, we accept either format as input.)
			if utf8.ValidRune(t) {
				fmt.Fprintf(b, "rune(%q)\n", t)
			} else {
				fmt.Fprintf(b, "int32(%v)\n", t)
			}
		case byte: // uint8
			// For bytes, we arbitrarily prefer the character interpretation.
			// (Every byte has a valid character encoding.)
			fmt.Fprintf(b, "byte(%q)\n", t)
		case []byte: // []uint8
			fmt.Fprintf(b, "[]byte(%q)\n", t)
		default:
			panic(fmt.Sprintf("unsupported type: %T", t))
		}
	}
	return b.Bytes()
}

// unmarshalCorpusFile decodes corpus bytes into their respective values.
func unmarshalCorpusFile(b []byte) ([]any, error) {
	if len(b) == 0 {
		return nil, fmt.Errorf("cannot unmarshal empty string")
	}
	lines := bytes.Split(b, []byte("\n"))
	if len(lines) < 2 {
		return nil, fmt.Errorf("must include version and at least one value")
	}
	if string(lines[0]) != encVersion1 {
		return nil, fmt.Errorf("unknown encoding version: %s", lines[0])
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
	return vals, nil
}

func parseCorpusValue(line []byte) (any, error) {
	fs := token.NewFileSet()
	expr, err := parser.ParseExprFrom(fs, "(test)", line, 0)
	if err != nil {
		return nil, err
	}
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return nil, fmt.Errorf("expected call expression")
	}
	if len(call.Args) != 1 {
		return nil, fmt.Errorf("expected call expression with 1 argument; got %d", len(call.Args))
	}
	arg := call.Args[0]

	if arrayType, ok := call.Fun.(*ast.ArrayType); ok {
		if arrayType.Len != nil {
			return nil, fmt.Errorf("expected []byte or primitive type")
		}
		elt, ok := arrayType.Elt.(*ast.Ident)
		if !ok || elt.Name != "byte" {
			return nil, fmt.Errorf("expected []byte")
		}
		lit, ok := arg.(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return nil, fmt.Errorf("string literal required for type []byte")
		}
		s, err := strconv.Unquote(lit.Value)
		if err != nil {
			return nil, err
		}
		return []byte(s), nil
	}

	var idType *ast.Ident
	if selector, ok := call.Fun.(*ast.SelectorExpr); ok {
		xIdent, ok := selector.X.(*ast.Ident)
		if !ok || xIdent.Name != "math" {
			return nil, fmt.Errorf("invalid selector type")
		}
		switch selector.Sel.Name {
		case "Float64frombits":
			idType = &ast.Ident{Name: "float64-bits"}
		case "Float32frombits":
			idType = &ast.Ident{Name: "float32-bits"}
		default:
			return nil, fmt.Errorf("invalid selector type")
		}
	} else {
		idType, ok = call.Fun.(*ast.Ident)
		if !ok {
			return nil, fmt.Errorf("expected []byte or primitive type")
		}
		if idType.Name == "bool" {
			id, ok := arg.(*ast.Ident)
			if !ok {
				return nil, fmt.Errorf("malformed bool")
			}
			if id.Name == "true" {
				return true, nil
			} else if id.Name == "false" {
				return false, nil
			} else {
				return nil, fmt.Errorf("true or false required for type bool")
			}
		}
	}

	var (
		val  string
		kind token.Token
	)
	if op, ok := arg.(*ast.UnaryExpr); ok {
		switch lit := op.X.(type) {
		case *ast.BasicLit:
			if op.Op != token.SUB {
				return nil, fmt.Errorf("unsupported operation on int/float: %v", op.Op)
			}
			// Special case for negative numbers.
			val = op.Op.String() + lit.Value // e.g. "-" + "124"
			kind = lit.Kind
		case *ast.Ident:
			if lit.Name != "Inf" {
				return nil, fmt.Errorf("expected operation on int or float type")
			}
			if op.Op == token.SUB {
				val = "-Inf"
			} else {
				val = "+Inf"
			}
			kind = token.FLOAT
		default:
			return nil, fmt.Errorf("expected operation on int or float type")
		}
	} else {
		switch lit := arg.(type) {
		case *ast.BasicLit:
			val, kind = lit.Value, lit.Kind
		case *ast.Ident:
			if lit.Name != "NaN" {
				return nil, fmt.Errorf("literal value required for primitive type")
			}
			val, kind = "NaN", token.FLOAT
		default:
			return nil, fmt.Errorf("literal value required for primitive type")
		}
	}

	switch typ := idType.Name; typ {
	case "string":
		if kind != token.STRING {
			return nil, fmt.Errorf("string literal value required for type string")
		}
		return strconv.Unquote(val)
	case "byte", "rune":
		if kind == token.INT {
			switch typ {
			case "rune":
				return parseInt(val, typ)
			case "byte":
				return parseUint(val, typ)
			}
		}
		if kind != token.CHAR {
			return nil, fmt.Errorf("character literal required for byte/rune types")
		}
		n := len(val)
		if n < 2 {
			return nil, fmt.Errorf("malformed character literal, missing single quotes")
		}
		code, _, _, err := strconv.UnquoteChar(val[1:n-1], '\'')
		if err != nil {
			return nil, err
		}
		if typ == "rune" {
			return code, nil
		}
		if code >= 256 {
			return nil, fmt.Errorf("can only encode single byte to a byte type")
		}
		return byte(code), nil
	case "int", "int8", "int16", "int32", "int64":
		if kind != token.INT {
			return nil, fmt.Errorf("integer literal required for int types")
		}
		return parseInt(val, typ)
	case "uint", "uint8", "uint16", "uint32", "uint64":
		if kind != token.INT {
			return nil, fmt.Errorf("integer literal required for uint types")
		}
		return parseUint(val, typ)
	case "float32":
		if kind != token.FLOAT && kind != token.INT {
			return nil, fmt.Errorf("float or integer literal required for float32 type")
		}
		v, err := strconv.ParseFloat(val, 32)
		return float32(v), err
	case "float64":
		if kind != token.FLOAT && kind != token.INT {
			return nil, fmt.Errorf("float or integer literal required for float64 type")
		}
		return strconv.ParseFloat(val, 64)
	case "float32-bits":
		if kind != token.INT {
			return nil, fmt.Errorf("integer literal required for math.Float32frombits type")
		}
		bits, err := parseUint(val, "uint32")
		if err != nil {
			return nil, err
		}
		return math.Float32frombits(bits.(uint32)), nil
	case "float64-bits":
		if kind != token.FLOAT && kind != token.INT {
			return nil, fmt.Errorf("integer literal required for math.Float64frombits type")
		}
		bits, err := parseUint(val, "uint64")
		if err != nil {
			return nil, err
		}
		return math.Float64frombits(bits.(uint64)), nil
	default:
		return nil, fmt.Errorf("expected []byte or primitive type")
	}
}

// parseInt returns an integer of value val and type typ.
func parseInt(val, typ string) (any, error) {
	switch typ {
	case "int":
		// The int type may be either 32 or 64 bits. If 32, the fuzz tests in the
		// corpus may include 64-bit values produced by fuzzing runs on 64-bit
		// architectures. When running those tests, we implicitly wrap the values to
		// fit in a regular int. (The test case is still “interesting”, even if the
		// specific values of its inputs are platform-dependent.)
		i, err := strconv.ParseInt(val, 0, 64)
		return int(i), err
	case "int8":
		i, err := strconv.ParseInt(val, 0, 8)
		return int8(i), err
	case "int16":
		i, err := strconv.ParseInt(val, 0, 16)
		return int16(i), err
	case "int32", "rune":
		i, err := strconv.ParseInt(val, 0, 32)
		return int32(i), err
	case "int64":
		return strconv.ParseInt(val, 0, 64)
	default:
		panic("unreachable")
	}
}

// parseInt returns an unsigned integer of value val and type typ.
func parseUint(val, typ string) (any, error) {
	switch typ {
	case "uint":
		i, err := strconv.ParseUint(val, 0, 64)
		return uint(i), err
	case "uint8", "byte":
		i, err := strconv.ParseUint(val, 0, 8)
		return uint8(i), err
	case "uint16":
		i, err := strconv.ParseUint(val, 0, 16)
		return uint16(i), err
	case "uint32":
		i, err := strconv.ParseUint(val, 0, 32)
		return uint32(i), err
	case "uint64":
		return strconv.ParseUint(val, 0, 64)
	default:
		panic("unreachable")
	}
}
