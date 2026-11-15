// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

func (p *parser) parseBoolValue(key string, valStart int) error {
	c := p.peek()
	if !isBareChar(c) {
		return p.scalarValueError(key, valStart, "boolean")
	}
	tok, start := p.readBare()
	if classifyBare(tok) != bareBool {
		return p.errorf(start, "want boolean for key %q", key)
	}
	v := tok == "true"
	p.file.Build.Arena = &v
	return nil
}
