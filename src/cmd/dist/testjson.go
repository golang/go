// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"
)

// lockedWriter serializes Write calls to an underlying Writer.
type lockedWriter struct {
	lock sync.Mutex
	w    io.Writer
}

func (w *lockedWriter) Write(b []byte) (int, error) {
	w.lock.Lock()
	defer w.lock.Unlock()
	return w.w.Write(b)
}

// testJSONFilter is an io.Writer filter that replaces the Package field in
// test2json output.
type testJSONFilter struct {
	w       io.Writer // Underlying writer
	variant string    // Add ":variant" to Package field

	lineBuf bytes.Buffer // Buffer for incomplete lines
}

func (f *testJSONFilter) Write(b []byte) (int, error) {
	bn := len(b)

	// Process complete lines, and buffer any incomplete lines.
	for len(b) > 0 {
		nl := bytes.IndexByte(b, '\n')
		if nl < 0 {
			f.lineBuf.Write(b)
			break
		}
		var line []byte
		if f.lineBuf.Len() > 0 {
			// We have buffered data. Add the rest of the line from b and
			// process the complete line.
			f.lineBuf.Write(b[:nl+1])
			line = f.lineBuf.Bytes()
		} else {
			// Process a complete line from b.
			line = b[:nl+1]
		}
		b = b[nl+1:]
		f.process(line)
		f.lineBuf.Reset()
	}

	return bn, nil
}

func (f *testJSONFilter) Flush() {
	// Write any remaining partial line to the underlying writer.
	if f.lineBuf.Len() > 0 {
		f.w.Write(f.lineBuf.Bytes())
		f.lineBuf.Reset()
	}
}

func (f *testJSONFilter) process(line []byte) {
	if len(line) > 0 && line[0] == '{' {
		// Plausible test2json output. Parse it generically.
		//
		// We go to some effort here to preserve key order while doing this
		// generically. This will stay robust to changes in the test2json
		// struct, or other additions outside of it. If humans are ever looking
		// at the output, it's really nice to keep field order because it
		// preserves a lot of regularity in the output.
		dec := json.NewDecoder(bytes.NewBuffer(line))
		dec.UseNumber()
		val, err := decodeJSONValue(dec)
		if err == nil && val.atom == json.Delim('{') {
			// Rewrite the Package field.
			found := false
			for i := 0; i < len(val.seq); i += 2 {
				if val.seq[i].atom == "Package" {
					if pkg, ok := val.seq[i+1].atom.(string); ok {
						val.seq[i+1].atom = pkg + ":" + f.variant
						found = true
						break
					}
				}
			}
			if found {
				data, err := json.Marshal(val)
				if err != nil {
					// Should never happen.
					panic(fmt.Sprintf("failed to round-trip JSON %q: %s", line, err))
				}
				f.w.Write(data)
				// Copy any trailing text. We expect at most a "\n" here, but
				// there could be other text and we want to feed that through.
				io.Copy(f.w, dec.Buffered())
				return
			}
		}
	}

	// Something went wrong. Just pass the line through.
	f.w.Write(line)
}

type jsonValue struct {
	atom json.Token  // If json.Delim, then seq will be filled
	seq  []jsonValue // If atom == json.Delim('{'), alternating pairs
}

var jsonPop = errors.New("end of JSON sequence")

func decodeJSONValue(dec *json.Decoder) (jsonValue, error) {
	t, err := dec.Token()
	if err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return jsonValue{}, err
	}

	switch t := t.(type) {
	case json.Delim:
		if t == '}' || t == ']' {
			return jsonValue{}, jsonPop
		}

		var seq []jsonValue
		for {
			val, err := decodeJSONValue(dec)
			if err == jsonPop {
				break
			} else if err != nil {
				return jsonValue{}, err
			}
			seq = append(seq, val)
		}
		return jsonValue{t, seq}, nil
	default:
		return jsonValue{t, nil}, nil
	}
}

func (v jsonValue) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	var marshal1 func(v jsonValue) error
	marshal1 = func { v ->
		if t, ok := v.atom.(json.Delim); ok {
			buf.WriteRune(rune(t))
			for i, v2 := range v.seq {
				if t == '{' && i%2 == 1 {
					buf.WriteByte(':')
				} else if i > 0 {
					buf.WriteByte(',')
				}
				if err := marshal1(v2); err != nil {
					return err
				}
			}
			if t == '{' {
				buf.WriteByte('}')
			} else {
				buf.WriteByte(']')
			}
			return nil
		}
		bytes, err := json.Marshal(v.atom)
		if err != nil {
			return err
		}
		buf.Write(bytes)
		return nil
	}
	err := marshal1(v)
	return buf.Bytes(), err
}

func synthesizeSkipEvent(enc *json.Encoder, pkg, msg string) {
	type event struct {
		Time    time.Time
		Action  string
		Package string
		Output  string `json:",omitempty"`
	}
	ev := event{Time: time.Now(), Package: pkg, Action: "start"}
	enc.Encode(ev)
	ev.Action = "output"
	ev.Output = msg
	enc.Encode(ev)
	ev.Action = "skip"
	ev.Output = ""
	enc.Encode(ev)
}
