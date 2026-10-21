// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import (
	"fmt"
	"path"
	"slices"
	"strings"
	"unicode/utf8"
)

// maxConfigSize caps logos.toml input at 1 MB; larger inputs fail parsing.
const maxConfigSize = 1 << 20

// strictFilesWant is the accepted-values suffix of the strict_files entry error.
const strictFilesWant = `".lgo", ".📜", ".⚡", or ".ʕ◔ϖ◔ʔ"`

// Parse parses data as a logos.toml document. Errors read
// name:line:column: message with byte-based columns. Keys must appear
// under their [project] or [build] table; keys at the file root are
// rejected as unknown keys. Unknown tables and keys are rejected.
func Parse(name string, data []byte) (*File, error) {
	if len(data) > maxConfigSize {
		return nil, fmt.Errorf("%s:1:1: file too large", name)
	}
	if off, c, ok := firstForbiddenControl(data); !ok {
		line, col := lineCol(data, off)
		return nil, fmt.Errorf("%s:%d:%d: invalid control character U+%04X", name, line, col, c)
	}
	if off, ok := firstInvalidUTF8(data); !ok {
		line, col := lineCol(data, off)
		return nil, fmt.Errorf("%s:%d:%d: invalid utf-8", name, line, col)
	}
	p := &parser{
		name:   name,
		src:    data,
		file:   defaultFile(),
		tables: make(map[string]bool),
		keys:   make(map[[2]string]bool),
	}
	if err := p.parse(); err != nil {
		return nil, err
	}
	return p.file, nil
}

// firstForbiddenControl returns the offset of the first forbidden control
// byte: NUL and other C0 controls except tab, LF, and CR in CRLF, plus DEL
// (0x7F), consistent with TOML. ok is false when such a byte exists.
func firstForbiddenControl(data []byte) (off int, c byte, ok bool) {
	for i := range data {
		b := data[i]
		if b == '\r' {
			if i+1 < len(data) && data[i+1] == '\n' {
				continue
			}
			return i, b, false
		}
		if b == '\t' || b == '\n' {
			continue
		}
		if b < 0x20 || b == 0x7F {
			return i, b, false
		}
	}
	return 0, 0, true
}

// firstInvalidUTF8 returns the offset of the first byte that is not valid
// UTF-8. ok is false when such a byte exists.
func firstInvalidUTF8(data []byte) (off int, ok bool) {
	for off < len(data) {
		if data[off] < utf8.RuneSelf {
			off++
			continue
		}
		r, size := utf8.DecodeRune(data[off:])
		if r == utf8.RuneError && size == 1 {
			return off, false
		}
		off += size
	}
	return 0, true
}

// lineCol maps a byte offset to 1-based line and byte-based column.
func lineCol(data []byte, off int) (line, col int) {
	line = 1
	start := 0
	for i := range off {
		if data[i] == '\n' {
			line++
			start = i + 1
		}
	}
	return line, off - start + 1
}

// parser is a cursor over the raw input. Line and column numbers are derived
// from byte offsets on demand, so the cursor only tracks its position.
type parser struct {
	name   string
	src    []byte
	pos    int
	table  string // "", "project", or "build"
	file   *File
	tables map[string]bool
	keys   map[[2]string]bool
}

// errorf reports an error at a byte offset.
func (p *parser) errorf(off int, format string, args ...any) error {
	line, col := lineCol(p.src, off)
	return fmt.Errorf("%s:%d:%d: %s", p.name, line, col, fmt.Sprintf(format, args...))
}

// peek returns the byte under the cursor, or 0 at EOF.
func (p *parser) peek() byte {
	if p.pos < len(p.src) {
		return p.src[p.pos]
	}
	return 0
}

// next consumes one byte.
func (p *parser) next() byte {
	c := p.peek()
	if c != 0 {
		p.pos++
	}
	return c
}

// skipInlineWS skips spaces, tabs, and carriage returns (so CRLF line
// endings work without shifting content columns).
func (p *parser) skipInlineWS() {
	for p.peek() == ' ' || p.peek() == '\t' || p.peek() == '\r' {
		p.next()
	}
}

// skipComment skips a # comment to (not including) the newline or EOF.
func (p *parser) skipComment() {
	for p.peek() != '\n' && p.peek() != 0 {
		p.next()
	}
}

// skipArraySpace skips whitespace, newlines, and comments inside arrays.
func (p *parser) skipArraySpace() {
	for {
		p.skipInlineWS()
		c := p.peek()
		if c == '\n' {
			p.next()
			continue
		}
		if c == '#' {
			p.skipComment()
			continue
		}
		return
	}
}

// parse runs the top-level loop over lines.
func (p *parser) parse() error {
	for {
		p.skipInlineWS()
		c := p.peek()
		switch {
		case c == 0:
			return nil
		case c == '\n':
			p.next()
		case c == '#':
			p.skipComment()
		case c == '[':
			if err := p.parseTable(); err != nil {
				return err
			}
		case isKeyChar(c):
			if err := p.parseKeyVal(); err != nil {
				return err
			}
		default:
			return p.errorf(p.pos, "unexpected character %q", string(rune(c)))
		}
	}
}

// parseTable parses a [table] header. The cursor is at '['.
func (p *parser) parseTable() error {
	start := p.pos
	p.next()
	if p.peek() == '[' {
		return p.errorf(start, "array of tables not allowed")
	}
	p.skipInlineWS()
	nameStart := p.pos
	for isTableChar(p.peek()) {
		p.next()
	}
	name := string(p.src[nameStart:p.pos])
	if name == "" {
		return p.errorf(start, "invalid table header")
	}
	p.skipInlineWS()
	if p.peek() != ']' {
		return p.errorf(p.pos, "expected \"]\" in table header")
	}
	p.next()
	p.skipInlineWS()
	if c := p.peek(); c != '\n' && c != 0 && c != '#' {
		return p.errorf(p.pos, "unexpected character %q", string(rune(c)))
	}
	if p.tables[name] {
		return p.errorf(start, "duplicate table %q", name)
	}
	p.tables[name] = true
	switch name {
	case "project", "build", "capabilities", "lints", "lints.thresholds":
		p.table = name
	default:
		return p.errorf(start, "unknown table %q", name)
	}
	return nil
}

// keyOwner maps a key to its owning table. The second result is false for
// unknown keys and for known keys outside their owning table, including
// keys at the file root before any table header.
func keyOwner(table, key string) (string, bool) {
	var owner string
	switch key {
	case "name", "edition", "strict_files":
		owner = "project"
	case "cgo", "cgo_allowlist", "opt_level", "synthetic_pgo", "target_cpu", "arena", "gc":
		owner = "build"
	case "direct_reflect", "transitive_reflect", "transitive_exec", "transitive_net", "transitive_unsafe",
		"reflect_allowlist", "transitive_exec_allowlist", "transitive_net_allowlist", "transitive_unsafe_allowlist":
		owner = "capabilities"
	default:
		owner = lintsOwner(key)
	}
	if owner == "" {
		return "", false
	}
	if table != owner {
		return "", false
	}
	return owner, true
}

// isArrayKey reports whether key takes a string array value.
func isArrayKey(key string) bool {
	switch key {
	case "strict_files", "cgo_allowlist",
		"reflect_allowlist", "transitive_exec_allowlist", "transitive_net_allowlist", "transitive_unsafe_allowlist":
		return true
	}
	return false
}

// parseKeyVal parses a key = value line. The cursor is at the key start.
func (p *parser) parseKeyVal() error {
	keyStart := p.pos
	for isKeyChar(p.peek()) {
		p.next()
	}
	key := string(p.src[keyStart:p.pos])
	p.skipInlineWS()
	if p.peek() != '=' {
		return p.errorf(p.pos, "expected \"=\" after key %q", key)
	}
	p.next()
	p.skipInlineWS()
	valStart := p.pos
	owner, known := keyOwner(p.table, key)
	if !known {
		return p.errorf(keyStart, "unknown key %q", key)
	}
	id := [2]string{owner, key}
	if p.keys[id] {
		return p.errorf(keyStart, "duplicate key %q", key)
	}
	p.keys[id] = true
	var err error
	switch {
	case isArrayKey(key):
		err = p.parseArrayValue(key)
	case isBoolKey(key):
		err = p.parseBoolValue(key, valStart)
	case isLintThreshold(key):
		err = p.parseIntValue(key, valStart)
	default:
		err = p.parseStringValue(key, valStart)
	}
	if err != nil {
		return err
	}
	p.skipInlineWS()
	if c := p.peek(); c != '\n' && c != 0 && c != '#' {
		return p.errorf(p.pos, "unexpected character %q", string(rune(c)))
	}
	return nil
}

// parseStringValue parses a scalar value for a string key at valStart.
func (p *parser) parseStringValue(key string, valStart int) error {
	c := p.peek()
	switch {
	case c == '"' || c == '\'':
		s, err := p.parseString()
		if err != nil {
			return err
		}
		return p.assignString(key, s, valStart)
	case c == '{':
		return p.errorf(valStart, "inline table not allowed")
	case c == '[':
		return p.errorf(valStart, "want string for key %q", key)
	case c == '\n' || c == 0 || c == '#':
		return p.errorf(valStart, "expected value for key %q", key)
	case isBareChar(c):
		tok, start := p.readBare()
		switch classifyBare(tok) {
		case bareBool, bareInt:
			return p.errorf(start, "want string for key %q", key)
		case bareFloat:
			return p.errorf(start, "float not allowed")
		case bareDate:
			return p.errorf(start, "date not allowed")
		default:
			return p.errorf(start, "invalid value")
		}
	default:
		return p.errorf(valStart, "unexpected character %q", string(rune(c)))
	}
}

// parseArrayValue parses an array value for an array key.
func (p *parser) parseArrayValue(key string) error {
	valStart := p.pos
	c := p.peek()
	switch {
	case c == '"' || c == '\'':
		// Parse the string first so a lexical error (such as an unterminated
		// string) wins over the type error.
		if _, err := p.parseString(); err != nil {
			return err
		}
		return p.errorf(valStart, "want string array for key %q", key)
	case c == '{':
		return p.errorf(p.pos, "inline table not allowed")
	case c == '\n' || c == 0 || c == '#':
		return p.errorf(p.pos, "expected value for key %q", key)
	case isBareChar(c):
		tok, start := p.readBare()
		switch classifyBare(tok) {
		case bareBool, bareInt:
			return p.errorf(start, "want string array for key %q", key)
		case bareFloat:
			return p.errorf(start, "float not allowed")
		case bareDate:
			return p.errorf(start, "date not allowed")
		default:
			return p.errorf(start, "invalid value")
		}
	case c != '[':
		return p.errorf(p.pos, "unexpected character %q", string(rune(c)))
	}
	arrStart := p.pos
	p.next() // consume '['
	out := []string{}
	for {
		p.skipArraySpace()
		elemStart := p.pos
		c := p.peek()
		switch {
		case c == 0:
			return p.errorf(arrStart, "unterminated array")
		case c == ']':
			p.next()
			return p.assignArray(key, out)
		case c == '"' || c == '\'':
			s, err := p.parseString()
			if err != nil {
				return err
			}
			if key == "strict_files" && !slices.Contains(strictFileExts, s) {
				return p.errorf(elemStart, "invalid strict_files entry %q (want %s)", s, strictFilesWant)
			}
			if key == "cgo_allowlist" {
				if _, err := path.Match(s, ""); err != nil {
					return p.errorf(elemStart, "[configuration::invalid_cgo_allowlist] cgo_allowlist pattern %q: %s", s, err)
				}
			}
			if isCapabilitiesAllowlist(key) {
				if _, err := path.Match(s, ""); err != nil {
					return p.errorf(elemStart, "[configuration::invalid_capabilities_allowlist] %s pattern %q: %s", key, s, err)
				}
			}
			out = append(out, s)
		case c == '{':
			return p.errorf(elemStart, "inline table not allowed")
		case c == '[':
			return p.errorf(elemStart, "nested array not allowed")
		case c == ',':
			return p.errorf(elemStart, "expected string in array")
		case isBareChar(c):
			tok, start := p.readBare()
			switch classifyBare(tok) {
			case bareBool, bareInt:
				return p.errorf(start, "want string for key %q", key)
			case bareFloat:
				return p.errorf(start, "float not allowed")
			case bareDate:
				return p.errorf(start, "date not allowed")
			default:
				return p.errorf(start, "invalid value")
			}
		default:
			return p.errorf(elemStart, "unexpected character %q", string(rune(c)))
		}
		p.skipArraySpace()
		switch p.peek() {
		case ',':
			p.next()
		case ']':
			p.next()
			return p.assignArray(key, out)
		case 0:
			return p.errorf(arrStart, "unterminated array")
		default:
			return p.errorf(p.pos, "expected \",\" or \"]\" in array")
		}
	}
}

// assignString validates and stores a scalar value.
func (p *parser) assignString(key, s string, valStart int) error {
	switch key {
	case "name":
		if s == "" {
			return p.errorf(valStart, "project name must not be empty")
		}
		p.file.Project.Name = s
	case "edition":
		if s != "2026" {
			return p.errorf(valStart, "unsupported edition %q (want \"2026\")", s)
		}
		p.file.Project.Edition = s
	case "cgo":
		if s != "forbid" && s != "restricted" && s != "allow" {
			return p.errorf(valStart, "invalid cgo value %q (want forbid, restricted, or allow)", s)
		}
		p.file.Build.Cgo = s
		p.file.Build.CgoSet = true
	case "gc":
		p.file.Build.GC = s
		p.file.Build.GCSet = true
	case "direct_reflect", "transitive_reflect", "transitive_exec", "transitive_net", "transitive_unsafe":
		if s != "deny" && s != "warn" && s != "allow" {
			return p.errorf(valStart, "invalid %s value %q (want deny, warn, or allow)", key, s)
		}
		assignSeverity(&p.file.Capabilities, key, s)
	case "opt_level", "synthetic_pgo", "target_cpu":
		return p.assignBuildEnum(key, s, valStart)
	default:
		if isLintGroup(key) {
			return p.assignLintGroup(key, s, valStart)
		}
	}
	return nil
}

// buildEnums lists the accepted spellings of the enumerated [build] keys.
var buildEnums = map[string][]string{
	"opt_level":     {"O0", "O1", "O2", "O3", "Os", "Oz"},
	"synthetic_pgo": {"off", "standard", "aggressive"},
	"target_cpu":    {"deterministic", "baseline", "native"},
}

// assignBuildEnum validates one enumerated [build] value against its
// accepted spellings and stores it.
func (p *parser) assignBuildEnum(key, s string, valStart int) error {
	if !slices.Contains(buildEnums[key], s) {
		return p.errorf(valStart, "%s must be one of %s", key, strings.Join(buildEnums[key], ", "))
	}
	switch key {
	case "opt_level":
		p.file.Build.OptLevel = s
	case "synthetic_pgo":
		p.file.Build.SyntheticPGO = s
		p.file.Build.SynthPGOSet = true
	case "target_cpu":
		p.file.Build.TargetCPU = s
	}
	return nil
}

// assignSeverity stores a validated capabilities severity value.
func assignSeverity(c *Capabilities, key, s string) {
	switch key {
	case "direct_reflect":
		c.DirectReflect = s
	case "transitive_reflect":
		c.TransitiveReflect = s
	case "transitive_exec":
		c.TransitiveExec = s
	case "transitive_net":
		c.TransitiveNet = s
	case "transitive_unsafe":
		c.TransitiveUnsafe = s
	}
}

// isCapabilitiesAllowlist reports whether key is a capabilities allowlist.
func isCapabilitiesAllowlist(key string) bool {
	switch key {
	case "reflect_allowlist", "transitive_exec_allowlist", "transitive_net_allowlist", "transitive_unsafe_allowlist":
		return true
	}
	return false
}

// assignArray stores an array value.
func (p *parser) assignArray(key string, out []string) error {
	switch key {
	case "strict_files":
		p.file.Project.StrictFiles = out
	case "cgo_allowlist":
		p.file.Build.CgoAllowlist = out
	case "reflect_allowlist":
		p.file.Capabilities.ReflectAllowlist = out
	case "transitive_exec_allowlist":
		p.file.Capabilities.TransitiveExecAllowlist = out
	case "transitive_net_allowlist":
		p.file.Capabilities.TransitiveNetAllowlist = out
	case "transitive_unsafe_allowlist":
		p.file.Capabilities.TransitiveUnsafeAllowlist = out
	}
	return nil
}

// parseString parses a basic or literal string. The cursor is at the quote.
func (p *parser) parseString() (string, error) {
	if p.peek() == '\'' {
		return p.parseLiteral()
	}
	start := p.pos
	p.next() // consume opening quote
	var b strings.Builder
	for {
		c := p.peek()
		if c == 0 || c == '\n' {
			return "", p.errorf(start, "unterminated string")
		}
		if c == '"' {
			p.next()
			return b.String(), nil
		}
		if c != '\\' {
			b.WriteByte(c)
			p.next()
			continue
		}
		escOff := p.pos
		p.next()
		e := p.peek()
		switch e {
		case 'b':
			b.WriteByte('\b')
			p.next()
		case 't':
			b.WriteByte('\t')
			p.next()
		case 'n':
			b.WriteByte('\n')
			p.next()
		case 'f':
			b.WriteByte('\f')
			p.next()
		case 'r':
			b.WriteByte('\r')
			p.next()
		case '"':
			b.WriteByte('"')
			p.next()
		case '\\':
			b.WriteByte('\\')
			p.next()
		case 'u', 'U':
			n := 4
			if e == 'U' {
				n = 8
			}
			p.next()
			v := 0
			for range n {
				h := p.peek()
				if !isHex(h) {
					return "", p.errorf(escOff, "invalid escape %q", rune(e))
				}
				v = v*16 + hexVal(h)
				p.next()
			}
			if !utf8.ValidRune(rune(v)) {
				return "", p.errorf(escOff, "invalid escape %q", rune(e))
			}
			b.WriteRune(rune(v))
		case 0:
			return "", p.errorf(start, "unterminated string")
		default:
			return "", p.errorf(escOff, "invalid escape %q", rune(e))
		}
	}
}

// parseLiteral parses a literal string with no escapes.
func (p *parser) parseLiteral() (string, error) {
	start := p.pos
	p.next() // consume opening quote
	var b strings.Builder
	for {
		c := p.peek()
		if c == 0 || c == '\n' {
			return "", p.errorf(start, "unterminated string")
		}
		if c == '\'' {
			p.next()
			return b.String(), nil
		}
		b.WriteByte(c)
		p.next()
	}
}

// bareKind classifies an unquoted value token.
type bareKind int

const (
	bareInvalid bareKind = iota
	bareBool
	bareInt
	bareFloat
	bareDate
)

// readBare reads an unquoted value token, returning it and its start offset.
func (p *parser) readBare() (string, int) {
	start := p.pos
	for isBareChar(p.peek()) {
		p.next()
	}
	return string(p.src[start:p.pos]), start
}

// classifyBare sorts a token into booleans, dates, floats, integers, or
// anything else (which is not a value in this subset).
func classifyBare(tok string) bareKind {
	if tok == "true" || tok == "false" {
		return bareBool
	}
	if isDatePrefix(tok) {
		return bareDate
	}
	if isFloatTok(tok) {
		return bareFloat
	}
	if isIntTok(tok) {
		return bareInt
	}
	return bareInvalid
}

// isDatePrefix reports a TOML date or time prefix: YYYY-MM-DD or HH:MM:SS,
// which also covers full datetimes starting that way.
func isDatePrefix(tok string) bool {
	if len(tok) >= 10 &&
		isDigit(tok[0]) && isDigit(tok[1]) && isDigit(tok[2]) && isDigit(tok[3]) &&
		tok[4] == '-' && isDigit(tok[5]) && isDigit(tok[6]) &&
		tok[7] == '-' && isDigit(tok[8]) && isDigit(tok[9]) {
		return true
	}
	return len(tok) >= 8 &&
		isDigit(tok[0]) && isDigit(tok[1]) && tok[2] == ':' &&
		isDigit(tok[3]) && isDigit(tok[4]) && tok[5] == ':' &&
		isDigit(tok[6]) && isDigit(tok[7])
}

// isFloatTok reports TOML float shapes: inf/nan, a dot between digits, or a
// decimal exponent.
func isFloatTok(tok string) bool {
	s := tok
	if s != "" && (s[0] == '+' || s[0] == '-') {
		s = s[1:]
	}
	if s == "" {
		return false
	}
	low := strings.ToLower(s)
	if low == "inf" || low == "nan" {
		return true
	}
	hasDigit := false
	for i := range s {
		if isDigit(s[i]) {
			hasDigit = true
		}
	}
	if !hasDigit {
		return false
	}
	if strings.Contains(s, ".") || strings.ContainsAny(s, "eE") {
		return isFloatChars(s)
	}
	return false
}

// isFloatChars reports whether every character fits a float literal.
func isFloatChars(s string) bool {
	for i := range s {
		c := s[i]
		if isDigit(c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' || c == '_' {
			continue
		}
		return false
	}
	return true
}

// isIntTok reports decimal and 0x/0o/0b integer shapes.
func isIntTok(tok string) bool {
	s := tok
	if s != "" && (s[0] == '+' || s[0] == '-') {
		s = s[1:]
	}
	if len(s) >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') {
		return len(s) > 2 && allIn(s[2:], isHex)
	}
	if len(s) >= 2 && s[0] == '0' && (s[1] == 'o' || s[1] == 'O') {
		return len(s) > 2 && allIn(s[2:], isOct)
	}
	if len(s) >= 2 && s[0] == '0' && (s[1] == 'b' || s[1] == 'B') {
		return len(s) > 2 && allIn(s[2:], isBin)
	}
	if s == "" || !isDigit(s[0]) {
		return false
	}
	return allIn(s, func(c byte) bool { return isDigit(c) || c == '_' })
}

// allIn reports whether s is non-empty and every byte satisfies f.
func allIn(s string, f func(byte) bool) bool {
	if s == "" {
		return false
	}
	for i := range s {
		if !f(s[i]) {
			return false
		}
	}
	return true
}

// isKeyChar reports bare-key characters.
func isKeyChar(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '-'
}

// isTableChar reports table-name characters, including dots for dotted names
// (which fail as unknown tables).
func isTableChar(c byte) bool {
	return isKeyChar(c) || c == '.'
}

// isBareChar reports unquoted value-token characters. Delimiters (whitespace,
// newlines, comments, commas, brackets, braces, quotes, equals) end the token.
func isBareChar(c byte) bool {
	return isKeyChar(c) || c == '+' || c == '.' || c == ':' || c == '/'
}

// isDigit reports ASCII digits.
func isDigit(c byte) bool {
	return c >= '0' && c <= '9'
}

// isHex reports hexadecimal digits.
func isHex(c byte) bool {
	return isDigit(c) || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
}

// isOct reports octal digits.
func isOct(c byte) bool {
	return c >= '0' && c <= '7'
}

// isBin reports binary digits.
func isBin(c byte) bool {
	return c == '0' || c == '1'
}

// hexVal returns a hex digit's value.
func hexVal(c byte) int {
	switch {
	case isDigit(c):
		return int(c - '0')
	case c >= 'a':
		return int(c-'a') + 10
	default:
		return int(c-'A') + 10
	}
}
