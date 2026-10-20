// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import "strconv"

// lintsOwner returns the table that owns a [lints] key: "lints" for a group
// key, "lints.thresholds" for a threshold key, and "" for any other key.
func lintsOwner(key string) string {
	switch {
	case isLintGroup(key):
		return "lints"
	case isLintThreshold(key):
		return "lints.thresholds"
	}
	return ""
}

// assignLintGroup validates and stores one group severity. A group that
// defaults to forbid cannot be set to anything else.
func (p *parser) assignLintGroup(group, s string, valStart int) error {
	switch s {
	case "forbid", "deny", "warn", "allow":
	default:
		return p.errorf(valStart, "invalid %s value %q (want forbid, deny, warn, or allow)", group, s)
	}
	if isForbidGroup(group) && s != "forbid" {
		return p.errorf(valStart, "[configuration::forbid_lowered] %s cannot be lowered below forbid", group)
	}
	if p.file.Lints.Groups == nil {
		p.file.Lints.Groups = make(map[string]string)
	}
	p.file.Lints.Groups[group] = s
	return nil
}

// parseIntValue parses the positive decimal integer of a [lints.thresholds]
// key. The cursor is at the value.
func (p *parser) parseIntValue(key string, valStart int) error {
	c := p.peek()
	if !isBareChar(c) {
		return p.intValueError(key, valStart, c)
	}
	tok, start := p.readBare()
	if classifyBare(tok) != bareInt {
		return p.errorf(start, "want integer for key %q", key)
	}
	n, err := strconv.Atoi(tok)
	if err != nil || n < 1 {
		return p.errorf(start, "invalid %s value %q (want a positive integer)", key, tok)
	}
	if p.file.Lints.Thresholds == nil {
		p.file.Lints.Thresholds = make(map[string]int)
	}
	p.file.Lints.Thresholds[key] = n
	return nil
}

// intValueError reports a value that does not start like an integer.
func (p *parser) intValueError(key string, valStart int, c byte) error {
	switch c {
	case '\n', 0, '#':
		return p.errorf(valStart, "expected value for key %q", key)
	case '"', '\'':
		return p.errorf(valStart, "want integer for key %q", key)
	case '[':
		return p.errorf(valStart, "want integer for key %q", key)
	case '{':
		return p.errorf(valStart, "inline table not allowed")
	}
	return p.errorf(valStart, "unexpected character %q", string(rune(c)))
}
