// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"strings"
)

type scalarCodec struct {
	kind   scalarKind
	quoted bool
}

func (f *jsonField) codec() scalarCodec {
	return scalarCodec{kind: f.shape.scalar, quoted: f.quoted}
}

func (f *jsonField) elemCodec() scalarCodec {
	return scalarCodec{kind: f.shape.scalar}
}

func (c scalarCodec) open() string {
	if c.kind == scalarString || c.quoted {
		return `"`
	}
	return ""
}

func (c scalarCodec) isFloat() bool {
	return c.kind == scalarFloat32 || c.kind == scalarFloat64
}

func (c scalarCodec) floatBits() int {
	if c.kind == scalarFloat32 {
		return 32
	}
	return 64
}

func (c scalarCodec) checkStmt(x string) string {
	if !c.isFloat() {
		return ""
	}
	return fmt.Sprintf("if %[1]s-%[1]s != 0 { return nil, %[2]s.FloatErr(%[1]s, float64(%[1]s), %[3]d) }", x, jsonAlias, c.floatBits())
}

func (c scalarCodec) sizeStmt(x string) string {
	switch c.kind {
	case scalarBool:
		return fmt.Sprintf("if %s { _logos_n += 4 } else { _logos_n += 5 }", x)
	case scalarInt:
		return fmt.Sprintf("_logos_n += %s.IntLen(int64(%s))", jsonAlias, x)
	case scalarUint:
		return fmt.Sprintf("_logos_n += %s.UintLen(uint64(%s))", jsonAlias, x)
	case scalarFloat32:
		return "_logos_n += " + jsonAlias + ".MaxFloatLen32"
	case scalarFloat64:
		return "_logos_n += " + jsonAlias + ".MaxFloatLen64"
	}
	return fmt.Sprintf("_logos_n += %s.%s(string(%s))", jsonAlias, c.stringLen(), x)
}

func (c scalarCodec) stringLen() string {
	if c.quoted {
		return "QuotedLen"
	}
	return "EscapedLen"
}

func (c scalarCodec) stringAppend() string {
	if c.quoted {
		return "AppendQuoted"
	}
	return "AppendEscaped"
}

func (c scalarCodec) emitStmt(x string) string {
	switch c.kind {
	case scalarBool:
		return fmt.Sprintf("if %s { _logos_buf = append(_logos_buf, \"true\"...) } else { _logos_buf = append(_logos_buf, \"false\"...) }", x)
	case scalarInt:
		return fmt.Sprintf("_logos_buf = %s.AppendInt(_logos_buf, int64(%s))", jsonAlias, x)
	case scalarUint:
		return fmt.Sprintf("_logos_buf = %s.AppendUint(_logos_buf, uint64(%s))", jsonAlias, x)
	case scalarFloat32, scalarFloat64:
		return fmt.Sprintf("_logos_buf = %s.AppendFloat(_logos_buf, float64(%s), %d)", jsonAlias, x, c.floatBits())
	}
	return fmt.Sprintf("_logos_buf = %s.%s(_logos_buf, string(%s))", jsonAlias, c.stringAppend(), x)
}

func (c scalarCodec) decodeFn() string {
	name := [...]string{"Bool", "Int", "Uint", "Float", "Float", "String"}[c.kind]
	if c.quoted {
		name += "Q"
	}
	return jsonAlias + "." + name
}

func (c scalarCodec) isZeroCond(x string) string {
	switch c.kind {
	case scalarBool:
		return "!" + x
	case scalarString:
		return x + ` == ""`
	}
	return x + " == 0"
}

func keyLiteral(key string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range key {
		switch r {
		case '<':
			b.WriteString(`\u003c`)
		case '>':
			b.WriteString(`\u003e`)
		case '&':
			b.WriteString(`\u0026`)
		case '\u2028':
			b.WriteString(`\u2028`)
		case '\u2029':
			b.WriteString(`\u2029`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteString(`":`)
	return b.String()
}
