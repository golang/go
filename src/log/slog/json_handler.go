// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog/internal/buffer"
	"strconv"
	"sync"
	"time"
	"unicode/utf8"
)

// JSONHandler is a [Handler] that writes Records to an [io.Writer] as
// line-delimited JSON objects.
type JSONHandler struct {
	*commonHandler
}

// NewJSONHandler creates a [JSONHandler] that writes to w,
// using the given options.
// If opts is nil, the default options are used.
func NewJSONHandler(w io.Writer, opts *HandlerOptions) *JSONHandler {
	if opts == nil {
		opts = &HandlerOptions{}
	}
	return &JSONHandler{
		&commonHandler{
			json: true,
			w:    w,
			opts: *opts,
			mu:   &sync.Mutex{},
		},
	}
}

// Enabled reports whether the handler handles records at the given level.
// The handler ignores records whose level is lower.
func (h *JSONHandler) Enabled(_ context.Context, level Level) bool {
	return h.commonHandler.enabled(level)
}

// WithAttrs returns a new [JSONHandler] whose attributes consists
// of h's attributes followed by attrs.
func (h *JSONHandler) WithAttrs(attrs []Attr) Handler {
	return &JSONHandler{commonHandler: h.commonHandler.withAttrs(attrs)}
}

func (h *JSONHandler) WithGroup(name string) Handler {
	return &JSONHandler{commonHandler: h.commonHandler.withGroup(name)}
}

// Handle formats its argument [Record] as a JSON object on a single line.
//
// If the Record's time is zero, the time is omitted.
// Otherwise, the key is "time"
// and the value is output as with json.Marshal.
//
// The level's key is "level" and its value is the result of calling [Level.String].
//
// If the AddSource option is set and source information is available,
// the key is "source", and the value is a record of type [Source].
//
// The message's key is "msg".
//
// To modify these or other attributes, or remove them from the output, use
// [HandlerOptions.ReplaceAttr].
//
// Values are formatted as with an [encoding/json.Encoder] with SetEscapeHTML(false),
// with two exceptions.
//
// First, an Attr whose Value is of type error is formatted as a string, by
// calling its Error method. Only errors in Attrs receive this special treatment,
// not errors embedded in structs, slices, maps or other data structures that
// are processed by the [encoding/json] package.
//
// Second, an encoding failure does not cause Handle to return an error.
// Instead, the error message is formatted as a string.
//
// Each call to Handle results in a single serialized call to io.Writer.Write.
func (h *JSONHandler) Handle(_ context.Context, r Record) error {
	return h.commonHandler.handle(r)
}

// Adapted from time.Time.MarshalJSON to avoid allocation.
func appendJSONTime(s *handleState, t time.Time) {
	if y := t.Year(); y < 0 || y >= 10000 {
		// RFC 3339 is clear that years are 4 digits exactly.
		// See golang.org/issue/4556#c15 for more discussion.
		s.appendError(errors.New("time.Time year outside of range [0,9999]"))
	}
	s.buf.WriteByte('"')
	*s.buf = t.AppendFormat(*s.buf, time.RFC3339Nano)
	s.buf.WriteByte('"')
}

func appendJSONValue(s *handleState, v Value) error {
	switch v.Kind() {
	case KindString:
		s.appendString(v.str())
	case KindInt64:
		*s.buf = strconv.AppendInt(*s.buf, v.Int64(), 10)
	case KindUint64:
		*s.buf = strconv.AppendUint(*s.buf, v.Uint64(), 10)
	case KindFloat64:
		// json.Marshal is funny about floats; it doesn't
		// always match strconv.AppendFloat. So just call it.
		// That's expensive, but floats are rare.
		if err := appendJSONMarshal(s.buf, v.Float64()); err != nil {
			return err
		}
	case KindBool:
		*s.buf = strconv.AppendBool(*s.buf, v.Bool())
	case KindDuration:
		// Do what json.Marshal does.
		*s.buf = strconv.AppendInt(*s.buf, int64(v.Duration()), 10)
	case KindTime:
		s.appendTime(v.Time())
	case KindAny:
		a := v.Any()
		_, jm := a.(json.Marshaler)
		if err, ok := a.(error); ok && !jm {
			s.appendString(err.Error())
		} else {
			return appendJSONMarshal(s.buf, a)
		}
	default:
		panic(fmt.Sprintf("bad kind: %s", v.Kind()))
	}
	return nil
}

type jsonEncoder struct {
	buf *bytes.Buffer
	// Use a json.Encoder to avoid escaping HTML.
	json *json.Encoder
}

var jsonEncoderPool = &sync.Pool{
	New: func() any {
		enc := &jsonEncoder{
			buf: new(bytes.Buffer),
		}
		enc.json = json.NewEncoder(enc.buf)
		enc.json.SetEscapeHTML(false)
		return enc
	},
}

func appendJSONMarshal(buf *buffer.Buffer, v any) error {
	j := jsonEncoderPool.Get().(*jsonEncoder)
	defer func() {
		// To reduce peak allocation, return only smaller buffers to the pool.
		const maxBufferSize = 16 << 10
		if j.buf.Cap() > maxBufferSize {
			return
		}
		j.buf.Reset()
		jsonEncoderPool.Put(j)
	}()

	if err := j.json.Encode(v); err != nil {
		return err
	}

	bs := j.buf.Bytes()
	buf.Write(bs[:len(bs)-1]) // remove final newline
	return nil
}

// appendEscapedJSONString escapes s for JSON and appends it to buf.
// It does not surround the string in quotation marks.
//
// Modified from encoding/json/encode.go:encodeState.string,
// with escapeHTML set to false.
func appendEscapedJSONString(buf []byte, s string) []byte {
	char := func(b byte) { buf = append(buf, b) }
	str := func(s string) { buf = append(buf, s...) }

	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if safeSet[b] {
				i++
				continue
			}
			if start < i {
				str(s[start:i])
			}
			char('\\')
			switch b {
			case '\\', '"':
				char(b)
			case '\n':
				char('n')
			case '\r':
				char('r')
			case '\t':
				char('t')
			default:
				// This encodes bytes < 0x20 except for \t, \n and \r.
				str(`u00`)
				char(hex[b>>4])
				char(hex[b&0xF])
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRuneInString(s[i:])
		if c == utf8.RuneError && size == 1 {
			if start < i {
				str(s[start:i])
			}
			str(`\ufffd`)
			i += size
			start = i
			continue
		}
		// U+2028 is LINE SEPARATOR.
		// U+2029 is PARAGRAPH SEPARATOR.
		// They are both technically valid characters in JSON strings,
		// but don't work in JSONP, which has to be evaluated as JavaScript,
		// and can lead to security holes there. It is valid JSON to
		// escape them, so we do so unconditionally.
		// See http://timelessrepo.com/json-isnt-a-javascript-subset for discussion.
		if c == '\u2028' || c == '\u2029' {
			if start < i {
				str(s[start:i])
			}
			str(`\u202`)
			char(hex[c&0xF])
			i += size
			start = i
			continue
		}
		i += size
	}
	if start < len(s) {
		str(s[start:])
	}
	return buf
}

const hex = "0123456789abcdef"

// Copied from encoding/json/tables.go.
//
// safeSet holds the value true if the ASCII character with the given array
// position can be represented inside a JSON string without any further
// escaping.
//
// All values are true except for the ASCII control characters (0-31), the
// double quote ("), and the backslash character ("\").
var safeSet = [utf8.RuneSelf]bool{
	' ':      true,
	'!':      true,
	'"':      false,
	'#':      true,
	'$':      true,
	'%':      true,
	'&':      true,
	'\'':     true,
	'(':      true,
	')':      true,
	'*':      true,
	'+':      true,
	',':      true,
	'-':      true,
	'.':      true,
	'/':      true,
	'0':      true,
	'1':      true,
	'2':      true,
	'3':      true,
	'4':      true,
	'5':      true,
	'6':      true,
	'7':      true,
	'8':      true,
	'9':      true,
	':':      true,
	';':      true,
	'<':      true,
	'=':      true,
	'>':      true,
	'?':      true,
	'@':      true,
	'A':      true,
	'B':      true,
	'C':      true,
	'D':      true,
	'E':      true,
	'F':      true,
	'G':      true,
	'H':      true,
	'I':      true,
	'J':      true,
	'K':      true,
	'L':      true,
	'M':      true,
	'N':      true,
	'O':      true,
	'P':      true,
	'Q':      true,
	'R':      true,
	'S':      true,
	'T':      true,
	'U':      true,
	'V':      true,
	'W':      true,
	'X':      true,
	'Y':      true,
	'Z':      true,
	'[':      true,
	'\\':     false,
	']':      true,
	'^':      true,
	'_':      true,
	'`':      true,
	'a':      true,
	'b':      true,
	'c':      true,
	'd':      true,
	'e':      true,
	'f':      true,
	'g':      true,
	'h':      true,
	'i':      true,
	'j':      true,
	'k':      true,
	'l':      true,
	'm':      true,
	'n':      true,
	'o':      true,
	'p':      true,
	'q':      true,
	'r':      true,
	's':      true,
	't':      true,
	'u':      true,
	'v':      true,
	'w':      true,
	'x':      true,
	'y':      true,
	'z':      true,
	'{':      true,
	'|':      true,
	'}':      true,
	'~':      true,
	'\u007f': true,
}
