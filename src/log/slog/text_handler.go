// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"context"
	"encoding"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"sync"
	"unicode"
	"unicode/utf8"
)

// TextHandler is a [Handler] that writes Records to an [io.Writer] as a
// sequence of key=value pairs separated by spaces and followed by a newline.
type TextHandler struct {
	*commonHandler
}

// NewTextHandler creates a [TextHandler] that writes to w,
// using the given options.
// If opts is nil, the default options are used.
func NewTextHandler(w io.Writer, opts *HandlerOptions) *TextHandler {
	if opts == nil {
		opts = &HandlerOptions{}
	}
	return &TextHandler{
		&commonHandler{
			json: false,
			w:    w,
			opts: *opts,
			mu:   &sync.Mutex{},
		},
	}
}

// Enabled reports whether the handler handles records at the given level.
// The handler ignores records whose level is lower.
func (h *TextHandler) Enabled(_ context.Context, level Level) bool {
	return h.commonHandler.enabled(level)
}

// WithAttrs returns a new [TextHandler] whose attributes consists
// of h's attributes followed by attrs.
func (h *TextHandler) WithAttrs(attrs []Attr) Handler {
	return &TextHandler{commonHandler: h.commonHandler.withAttrs(attrs)}
}

func (h *TextHandler) WithGroup(name string) Handler {
	return &TextHandler{commonHandler: h.commonHandler.withGroup(name)}
}

// Handle formats its argument [Record] as a single line of space-separated
// key=value items.
//
// If the Record's time is zero, the time is omitted.
// Otherwise, the key is "time"
// and the value is output in RFC3339 format with millisecond precision.
//
// The level's key is "level" and its value is the result of calling [Level.String].
//
// If the AddSource option is set and source information is available,
// the key is "source" and the value is output as FILE:LINE.
//
// The message's key is "msg".
//
// To modify these or other attributes, or remove them from the output, use
// [HandlerOptions.ReplaceAttr].
//
// If a value implements [encoding.TextMarshaler], the result of MarshalText is
// written. Otherwise, the result of [fmt.Sprint] is written.
//
// Keys and values are quoted with [strconv.Quote] if they contain Unicode space
// characters, non-printing characters, '"' or '='.
//
// Keys inside groups consist of components (keys or group names) separated by
// dots. No further escaping is performed.
// Thus there is no way to determine from the key "a.b.c" whether there
// are two groups "a" and "b" and a key "c", or a single group "a.b" and a key "c",
// or single group "a" and a key "b.c".
// If it is necessary to reconstruct the group structure of a key
// even in the presence of dots inside components, use
// [HandlerOptions.ReplaceAttr] to encode that information in the key.
//
// Each call to Handle results in a single serialized call to
// io.Writer.Write.
func (h *TextHandler) Handle(_ context.Context, r Record) error {
	return h.commonHandler.handle(r)
}

func appendTextValue(s *handleState, v Value) error {
	switch v.Kind() {
	case KindString:
		s.appendString(v.str())
	case KindTime:
		s.appendTime(v.time())
	case KindAny:
		if tm, ok := v.any.(encoding.TextMarshaler); ok {
			data, err := tm.MarshalText()
			if err != nil {
				return err
			}
			// TODO: avoid the conversion to string.
			s.appendString(string(data))
			return nil
		}
		if bs, ok := byteSlice(v.any); ok {
			// As of Go 1.19, this only allocates for strings longer than 32 bytes.
			s.buf.WriteString(strconv.Quote(string(bs)))
			return nil
		}
		s.appendString(fmt.Sprintf("%+v", v.Any()))
	default:
		*s.buf = v.append(*s.buf)
	}
	return nil
}

// byteSlice returns its argument as a []byte if the argument's
// underlying type is []byte, along with a second return value of true.
// Otherwise it returns nil, false.
func byteSlice(a any) ([]byte, bool) {
	if bs, ok := a.([]byte); ok {
		return bs, true
	}
	// Like Printf's %s, we allow both the slice type and the byte element type to be named.
	t := reflect.TypeOf(a)
	if t != nil && t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
		return reflect.ValueOf(a).Bytes(), true
	}
	return nil, false
}

func needsQuoting(s string) bool {
	if len(s) == 0 {
		return true
	}
	for i := 0; i < len(s); {
		b := s[i]
		if b < utf8.RuneSelf {
			// Quote anything except a backslash that would need quoting in a
			// JSON string, as well as space and '='
			if b != '\\' && (b == ' ' || b == '=' || !safeSet[b]) {
				return true
			}
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError || unicode.IsSpace(r) || !unicode.IsPrint(r) {
			return true
		}
		i += size
	}
	return false
}
