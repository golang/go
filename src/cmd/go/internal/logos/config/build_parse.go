// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

// isBoolKey reports whether key takes a boolean value.
func isBoolKey(key string) bool {
	return key == "arena"
}

// parseBoolValue parses the bare true or false value of a boolean key. The
// cursor is at the value.
func (p *parser) parseBoolValue(key string, valStart int) error {
	c := p.peek()
	if !isBareChar(c) {
		return p.boolValueError(key, valStart, c)
	}
	tok, start := p.readBare()
	if classifyBare(tok) != bareBool {
		return p.errorf(start, "want boolean for key %q", key)
	}
	p.assignBool(key, tok == "true")
	return nil
}

// boolValueError reports a value that does not start like a boolean.
func (p *parser) boolValueError(key string, valStart int, c byte) error {
	switch c {
	case '\n', 0, '#':
		return p.errorf(valStart, "expected value for key %q", key)
	case '"', '\'':
		if _, err := p.parseString(); err != nil {
			return err
		}
		return p.errorf(valStart, "want boolean for key %q", key)
	case '[':
		return p.errorf(valStart, "want boolean for key %q", key)
	case '{':
		return p.errorf(valStart, "inline table not allowed")
	}
	return p.errorf(valStart, "unexpected character %q", string(rune(c)))
}

// assignBool stores a boolean key's value.
func (p *parser) assignBool(key string, v bool) {
	if key == "arena" {
		p.file.Build.Arena = v
	}
}
