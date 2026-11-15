// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import (
	"fmt"
	"internal/logosfile"
	"path"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

const maxConfigSize = 1 << 20

func strictFilesWant() string {
	exts := logosfile.Exts()
	for i, e := range exts {
		exts[i] = strconv.Quote(e)
	}
	last := len(exts) - 1
	return strings.Join(exts[:last], ", ") + ", or " + exts[last]
}

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

type parser struct {
	name   string
	src    []byte
	pos    int
	table  string
	file   *File
	tables map[string]bool
	keys   map[[2]string]bool
}

func (p *parser) errorf(off int, format string, args ...any) error {
	line, col := lineCol(p.src, off)
	return fmt.Errorf("%s:%d:%d: %s", p.name, line, col, fmt.Sprintf(format, args...))
}

func (p *parser) peek() byte {
	if p.pos < len(p.src) {
		return p.src[p.pos]
	}
	return 0
}

func (p *parser) next() {
	if p.peek() != 0 {
		p.pos++
	}
}

func (p *parser) skipInlineWS() {
	for p.peek() == ' ' || p.peek() == '\t' || p.peek() == '\r' {
		p.next()
	}
}

func (p *parser) skipComment() {
	for p.peek() != '\n' && p.peek() != 0 {
		p.next()
	}
}

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

func keyOwner(table, key string) (string, bool) {
	var owner string
	switch key {
	case "strict_files":
		owner = "project"
	case "cgo", "cgo_allowlist", "opt_level", "target_cpu", "arena":
		owner = "build"
	case "transitive_reflect", "transitive_exec", "transitive_net", "transitive_unsafe",
		"reflect_allowlist", "transitive_exec_allowlist", "transitive_net_allowlist", "transitive_unsafe_allowlist":
		owner = "capabilities"
	default:
		switch {
		case isLintGroup(key):
			owner = "lints"
		case isLintThreshold(key):
			owner = "lints.thresholds"
		}
	}
	if owner == "" {
		return "", false
	}
	if table != owner {
		return "", false
	}
	return owner, true
}

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
	case key == "strict_files" || key == "cgo_allowlist" ||
		key == "reflect_allowlist" || key == "transitive_exec_allowlist" ||
		key == "transitive_net_allowlist" || key == "transitive_unsafe_allowlist":
		err = p.parseArrayValue(key)
	case key == "arena":
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
		return p.rejectBare(key, "string")
	default:
		return p.errorf(valStart, "unexpected character %q", string(rune(c)))
	}
}

func (p *parser) rejectBare(key, want string) error {
	tok, start := p.readBare()
	switch classifyBare(tok) {
	case bareBool, bareInt:
		return p.errorf(start, "want %s for key %q", want, key)
	case bareFloat:
		return p.errorf(start, "float not allowed")
	case bareDate:
		return p.errorf(start, "date not allowed")
	default:
		return p.errorf(start, "invalid value")
	}
}

func (p *parser) parseArrayValue(key string) error {
	valStart := p.pos
	c := p.peek()
	switch {
	case c == '"' || c == '\'':
		if _, err := p.parseString(); err != nil {
			return err
		}
		return p.errorf(valStart, "want string array for key %q", key)
	case c == '{':
		return p.errorf(p.pos, "inline table not allowed")
	case c == '\n' || c == 0 || c == '#':
		return p.errorf(p.pos, "expected value for key %q", key)
	case isBareChar(c):
		return p.rejectBare(key, "string array")
	case c != '[':
		return p.errorf(p.pos, "unexpected character %q", string(rune(c)))
	}
	arrStart := p.pos
	p.next()
	out := []string{}
	for {
		p.skipArraySpace()
		switch p.peek() {
		case 0:
			return p.errorf(arrStart, "unterminated array")
		case ']':
			p.next()
			return p.assignArray(key, out)
		}
		elemStart := p.pos
		c := p.peek()
		switch {
		case c == '"' || c == '\'':
			s, err := p.parseString()
			if err != nil {
				return err
			}
			if err := p.checkArrayElem(key, elemStart, s); err != nil {
				return err
			}
			out = append(out, s)
		case c == '{':
			return p.errorf(elemStart, "inline table not allowed")
		case c == '[':
			return p.errorf(elemStart, "nested array not allowed")
		case c == ',':
			return p.errorf(elemStart, "expected string in array")
		case isBareChar(c):
			return p.rejectBare(key, "string")
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

func (p *parser) checkArrayElem(key string, elemStart int, s string) error {
	if key == "strict_files" && !slices.Contains(logosfile.Exts(), s) {
		return p.errorf(elemStart, "invalid strict_files entry %q (want %s)", s, strictFilesWant())
	}
	var tag string
	switch key {
	case "cgo_allowlist":
		tag = "configuration::invalid_cgo_allowlist"
	case "reflect_allowlist", "transitive_exec_allowlist", "transitive_net_allowlist", "transitive_unsafe_allowlist":
		tag = "configuration::invalid_capabilities_allowlist"
	default:
		return nil
	}
	if _, err := path.Match(s, ""); err != nil {
		return p.errorf(elemStart, "[%s] %s pattern %q: %s", tag, key, s, err)
	}
	return nil
}

func (p *parser) assignString(key, s string, valStart int) error {
	switch key {
	case "cgo":
		if s != "forbid" && s != "restricted" && s != "allow" {
			return p.errorf(valStart, "invalid cgo value %q (want forbid, restricted, or allow)", s)
		}
		p.file.Build.Cgo = s
		p.file.Build.CgoSet = true
	case "transitive_reflect", "transitive_exec", "transitive_net", "transitive_unsafe":
		if s != "deny" && s != "warn" && s != "allow" {
			return p.errorf(valStart, "invalid %s value %q (want deny, warn, or allow)", key, s)
		}
		assignSeverity(&p.file.Capabilities, key, s)
	case "opt_level", "target_cpu":
		return p.assignBuildEnum(key, s, valStart)
	default:
		if isLintGroup(key) {
			return p.assignLintGroup(key, s, valStart)
		}
	}
	return nil
}

var buildEnums = map[string][]string{
	"opt_level":  {"O0", "O1", "O2", "O3", "Os"},
	"target_cpu": {"deterministic", "baseline", "native"},
}

func (p *parser) assignBuildEnum(key, s string, valStart int) error {
	if !slices.Contains(buildEnums[key], s) {
		return p.errorf(valStart, "%s must be one of %s", key, strings.Join(buildEnums[key], ", "))
	}
	switch key {
	case "opt_level":
		p.file.Build.OptLevel = s
	case "target_cpu":
		p.file.Build.TargetCPU = s
	}
	return nil
}

func assignSeverity(c *Capabilities, key, s string) {
	switch key {
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

func (p *parser) parseString() (string, error) {
	if p.peek() == '\'' {
		return p.parseLiteral()
	}
	start := p.pos
	p.next()
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
			r, err := p.unicodeEscape(escOff, e)
			if err != nil {
				return "", err
			}
			b.WriteRune(r)
		case 0:
			return "", p.errorf(start, "unterminated string")
		default:
			return "", p.errorf(escOff, "invalid escape %q", rune(e))
		}
	}
}

func (p *parser) unicodeEscape(escOff int, e byte) (rune, error) {
	n := 4
	if e == 'U' {
		n = 8
	}
	p.next()
	v := 0
	for range n {
		h := p.peek()
		if !isHex(h) {
			return 0, p.errorf(escOff, "invalid escape %q", rune(e))
		}
		v = v*16 + hexVal(h)
		p.next()
	}
	if !utf8.ValidRune(rune(v)) {
		return 0, p.errorf(escOff, "invalid escape %q", rune(e))
	}
	return rune(v), nil
}

func (p *parser) parseLiteral() (string, error) {
	start := p.pos
	p.next()
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

type bareKind int

const (
	bareInvalid bareKind = iota
	bareBool
	bareInt
	bareFloat
	bareDate
)

func (p *parser) readBare() (string, int) {
	start := p.pos
	for isBareChar(p.peek()) {
		p.next()
	}
	return string(p.src[start:p.pos]), start
}

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

func isKeyChar(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '-'
}

func isTableChar(c byte) bool {
	return isKeyChar(c) || c == '.'
}

func isBareChar(c byte) bool {
	return isKeyChar(c) || c == '+' || c == '.' || c == ':' || c == '/'
}

func isDigit(c byte) bool {
	return c >= '0' && c <= '9'
}

func isHex(c byte) bool {
	return isDigit(c) || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
}

func isOct(c byte) bool {
	return c >= '0' && c <= '7'
}

func isBin(c byte) bool {
	return c == '0' || c == '1'
}

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
