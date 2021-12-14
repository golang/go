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
	"strconv"
)

// encVersion1 will be the first line of a file with version 1 encoding.
var encVersion1 = "go test fuzz v1"

// marshalCorpusFile encodes an arbitrary number of arguments into the file format for the
// corpus.
func marshalCorpusFile(vals ...interface{}) []byte {
	if len(vals) == 0 {
		panic("must have at least one value to marshal")
	}
	b := bytes.NewBuffer([]byte(encVersion1 + "\n"))
	// TODO(katiehockman): keep uint8 and int32 encoding where applicable,
	// instead of changing to byte and rune respectively.
	for _, val := range vals {
		switch t := val.(type) {
		case int, int8, int16, int64, uint, uint16, uint32, uint64, float32, float64, bool:
			fmt.Fprintf(b, "%T(%v)\n", t, t)
		case string:
			fmt.Fprintf(b, "string(%q)\n", t)
		case rune: // int32
			fmt.Fprintf(b, "rune(%q)\n", t)
		case byte: // uint8
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
func unmarshalCorpusFile(b []byte) ([]interface{}, error) {
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
	var vals []interface{}
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

func parseCorpusValue(line []byte) (interface{}, error) {
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

	idType, ok := call.Fun.(*ast.Ident)
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
	var (
		val  string
		kind token.Token
	)
	if op, ok := arg.(*ast.UnaryExpr); ok {
		// Special case for negative numbers.
		lit, ok := op.X.(*ast.BasicLit)
		if !ok || (lit.Kind != token.INT && lit.Kind != token.FLOAT) {
			return nil, fmt.Errorf("expected operation on int or float type")
		}
		if op.Op != token.SUB {
			return nil, fmt.Errorf("unsupported operation on int: %v", op.Op)
		}
		val = op.Op.String() + lit.Value // e.g. "-" + "124"
		kind = lit.Kind
	} else {
		lit, ok := arg.(*ast.BasicLit)
		if !ok {
			return nil, fmt.Errorf("literal value required for primitive type")
		}
		val, kind = lit.Value, lit.Kind
	}

	switch typ := idType.Name; typ {
	case "string":
		if kind != token.STRING {
			return nil, fmt.Errorf("string literal value required for type string")
		}
		return strconv.Unquote(val)
	case "byte", "rune":
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
	default:
		return nil, fmt.Errorf("expected []byte or primitive type")
	}
}

// parseInt returns an integer of value val and type typ.
func parseInt(val, typ string) (interface{}, error) {
	switch typ {
	case "int":
		return strconv.Atoi(val)
	case "int8":
		i, err := strconv.ParseInt(val, 10, 8)
		return int8(i), err
	case "int16":
		i, err := strconv.ParseInt(val, 10, 16)
		return int16(i), err
	case "int32":
		i, err := strconv.ParseInt(val, 10, 32)
		return int32(i), err
	case "int64":
		return strconv.ParseInt(val, 10, 64)
	default:
		panic("unreachable")
	}
}

// parseInt returns an unsigned integer of value val and type typ.
func parseUint(val, typ string) (interface{}, error) {
	switch typ {
	case "uint":
		i, err := strconv.ParseUint(val, 10, 0)
		return uint(i), err
	case "uint8":
		i, err := strconv.ParseUint(val, 10, 8)
		return uint8(i), err
	case "uint16":
		i, err := strconv.ParseUint(val, 10, 16)
		return uint16(i), err
	case "uint32":
		i, err := strconv.ParseUint(val, 10, 32)
		return uint32(i), err
	case "uint64":
		return strconv.ParseUint(val, 10, 64)
	default:
		panic("unreachable")
	}
}
