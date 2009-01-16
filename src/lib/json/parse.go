// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// JSON (JavaScript Object Notation) parser.
// See http://www.json.org/

package json

import (
	"array";
	"fmt";
	"io";
	"math";
	"strconv";
	"strings";
	"utf8";
)

// Strings
//
//   Double quoted with escapes: \" \\ \/ \b \f \n \r \t \uXXXX.
//   No literal control characters, supposedly.
//   Have also seen \' and embedded newlines.

func UnHex(p string, r, l int) (v int, ok bool) {
	v = 0;
	for i := r; i < l; i++ {
		if i >= len(p) {
			return 0, false
		}
		v *= 16;
		switch {
		case '0' <= p[i] && p[i] <= '9':
			v += int(p[i] - '0');
		case 'a' <= p[i] && p[i] <= 'f':
			v += int(p[i] - 'a' + 10);
		case 'A' <= p[i] && p[i] <= 'F':
			v += int(p[i] - 'A' + 10);
		default:
			return 0, false;
		}
	}
	return v, true;
}

export func Unquote(s string) (t string, ok bool) {
	if len(s) < 2 || s[0] != '"' || s[len(s)-1] != '"' {
		return
	}
	b := make([]byte, len(s));
	w := 0;
	for r := 1; r < len(s)-1; {
		switch {
		case s[r] == '\\':
			r++;
			if r >= len(s)-1 {
				return
			}
			switch s[r] {
			default:
				return;
			case '"', '\\', '/', '\'':
				b[w] = s[r];
				r++;
				w++;
			case 'b':
				b[w] = '\b';
				r++;
				w++;
			case 'f':
				b[w] = '\f';
				r++;
				w++;
			case 'n':
				b[w] = '\n';
				r++;
				w++;
			case 'r':
				b[w] = '\r';
				r++;
				w++;
			case 't':
				b[w] = '\t';
				r++;
				w++;
			case 'u':
				r++;
				rune, ok := UnHex(s, r, 4);
				if !ok {
					return
				}
				r += 4;
				w += utf8.EncodeRune(rune, b[w:len(b)]);
			}
		// Control characters are invalid, but we've seen raw \n.
		case s[r] < ' ' && s[r] != '\n':
			if s[r] == '\n' {
				b[w] = '\n';
				r++;
				w++;
				break;
			}
			return;
		// ASCII
		case s[r] < utf8.RuneSelf:
			b[w] = s[r];
			r++;
			w++;
		// Coerce to well-formed UTF-8.
		default:
			rune, size := utf8.DecodeRuneInString(s, r);
			r += size;
			w += utf8.EncodeRune(rune, b[w:len(b)]);
		}
	}
	return string(b[0:w]), true
}

export func Quote(s string) string {
	chr := make([]byte, utf8.UTFMax);
	chr0 := chr[0:1];
	b := new(io.ByteBuffer);
	chr[0] = '"';
	b.Write(chr0);
	for i := 0; i < len(s); i++ {
		switch {
		case s[i]=='"' || s[i]=='\\':
			chr[0] = '\\';
			chr[1] = s[i];
			b.Write(chr[0:2]);

		case s[i] == '\b':
			chr[0] = '\\';
			chr[1] = 'b';
			b.Write(chr[0:2]);

		case s[i] == '\f':
			chr[0] = '\\';
			chr[1] = 'f';
			b.Write(chr[0:2]);

		case s[i] == '\n':
			chr[0] = '\\';
			chr[1] = 'n';
			b.Write(chr[0:2]);

		case s[i] == '\r':
			chr[0] = '\\';
			chr[1] = 'r';
			b.Write(chr[0:2]);

		case s[i] == '\t':
			chr[0] = '\\';
			chr[1] = 't';
			b.Write(chr[0:2]);

		case 0x20 <= s[i] && s[i] < utf8.RuneSelf:
			chr[0] = s[i];
			b.Write(chr0);
		}
	}
	chr[0] = '"';
	b.Write(chr0);
	return string(b.Data());
}


// Lexer

type Lexer struct {
	s string;
	i int;
	kind int;
	token string;
}

func Punct(c byte) bool {
	return c=='"' || c=='[' || c==']' || c==':' || c=='{' || c=='}' || c==','
}

func White(c byte) bool {
	return c==' ' || c=='\t' || c=='\n' || c=='\v'
}

func SkipWhite(p string, i int) int {
	for i < len(p) && White(p[i]) {
		i++
	}
	return i
}

func SkipToken(p string, i int) int {
	for i < len(p) && !Punct(p[i]) && !White(p[i]) {
		i++
	}
	return i
}

func SkipString(p string, i int) int {
	for i++; i < len(p) && p[i] != '"'; i++ {
		if p[i] == '\\' {
			i++
		}
	}
	if i >= len(p) {
		return i
	}
	return i+1
}

func (t *Lexer) Next() {
	i, s := t.i, t.s;
	i = SkipWhite(s, i);
	if i >= len(s) {
		t.kind = 0;
		t.token = "";
		t.i = len(s);
		return;
	}

	c := s[i];
	switch {
	case c == '-' || '0' <= c && c <= '9':
		j := SkipToken(s, i);
		t.kind = '1';
		t.token = s[i:j];
		i = j;

	case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
		j := SkipToken(s, i);
		t.kind = 'a';
		t.token = s[i:j];
		i = j;

	case c == '"':
		j := SkipString(s, i);
		t.kind = '"';
		t.token = s[i:j];
		i = j;

	case c == '[', c == ']', c == ':', c == '{', c == '}', c == ',':
		t.kind = int(c);
		t.token = s[i:i+1];
		i++;

	default:
		t.kind = '?';
		t.token = s[i:i+1];
	}

	t.i = i;
}


// Parser
//
// Implements parsing but not the actions.  Those are
// carried out by the implementation of the Builder interface.
// A Builder represents the object being created.
// Calling a method like Int64(i) sets that object to i.
// Calling a method like Elem(i) or Key(s) creates a
// new builder for a subpiece of the object (logically,
// an array element or a map key).
//
// There are two Builders, in other files.
// The JsonBuilder builds a generic Json structure
// in which maps are maps.
// The StructBuilder copies data into a possibly
// nested data structure, using the "map keys"
// as struct field names.

type Value interface {}

export type Builder interface {
	// Set value
	Int64(i int64);
	Uint64(i uint64);
	Float64(f float64);
	String(s string);
	Bool(b bool);
	Null();
	Array();
	Map();

	// Create sub-Builders
	Elem(i int) Builder;
	Key(s string) Builder;
}

func ParseValue(lex *Lexer, build Builder) bool {
	ok := false;
Switch:
	switch lex.kind {
	case 0:
		break;
	case '1':
		// If the number is exactly an integer, use that.
		if i, err := strconv.Atoi64(lex.token); err == nil {
			build.Int64(i);
			ok = true;
		}
		else if i, err := strconv.Atoui64(lex.token); err == nil {
			build.Uint64(i);
			ok = true;
		}
		// Fall back to floating point.
		else if f, err := strconv.Atof64(lex.token); err == nil {
			build.Float64(f);
			ok = true;
		}

	case 'a':
		switch lex.token {
		case "true":
			build.Bool(true);
			ok = true;
		case "false":
			build.Bool(false);
			ok = true;
		case "null":
			build.Null();
			ok = true;
		}

	case '"':
		if str, ok1 := Unquote(lex.token); ok1 {
			build.String(str);
			ok = true;
		}

	case '[':
		// array
		build.Array();
		lex.Next();
		n := 0;
		for lex.kind != ']' {
			if n > 0 {
				if lex.kind != ',' {
					break Switch;
				}
				lex.Next();
			}
			if !ParseValue(lex, build.Elem(n)) {
				break Switch;
			}
			n++;
		}
		ok = true;

	case '{':
		// map
		lex.Next();
		build.Map();
		n := 0;
		for lex.kind != '}' {
			if n > 0 {
				if lex.kind != ',' {
					break Switch;
				}
				lex.Next();
			}
			if lex.kind != '"' {
				break Switch;
			}
			key, ok := Unquote(lex.token);
			if !ok {
				break Switch;
			}
			lex.Next();
			if lex.kind != ':' {
				break Switch;
			}
			lex.Next();
			if !ParseValue(lex, build.Key(key)) {
				break Switch;
			}
			n++;
		}
		ok = true;
	}

	if ok {
		lex.Next();
	}
	return ok;
}

export func Parse(s string, build Builder) (ok bool, errindx int, errtok string) {
	lex := new(Lexer);
	lex.s = s;
	lex.Next();
	if ParseValue(lex, build) {
		if lex.kind == 0 {	// EOF
			return true, 0, ""
		}
	}
	return false, lex.i, lex.token
}

