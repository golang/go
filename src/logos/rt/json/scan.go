// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"errors"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
	"unsafe"
)

const (
	stateIdle uint8 = iota
	stateFirst
	stateMember
	stateDone
)

var errRejected = errors.New("logos/rt/json: input rejected by the scanner but accepted by encoding/json")

// A Scanner is a cursor over one complete JSON value held in memory. It
// validates every byte it consumes, so after Next returns false the whole
// input has been checked. Generated UnmarshalJSON methods drive it: Object
// opens the root object, Next and Key walk the members, a typed reader or
// Skip or Fallback consumes each value, and Err and Saved report the
// outcome. Create one with NewScanner and use it through its address.
type Scanner struct {
	data  []byte
	pos   int
	depth int
	state uint8
	typ   any
	key   string
	fold  [32]byte
	err   error
	saved error
	kind  byte
	flags int
	start int
	end   int
}

// NewScanner returns a Scanner over data. typ is a typed nil pointer to the
// target struct, for example (*User)(nil); it is used only to build error
// values.
func NewScanner(data []byte, typ any) Scanner {
	return Scanner{data: data, typ: typ}
}

func (s *Scanner) structType() reflect.Type {
	t := reflect.TypeOf(s.typ)
	if t != nil && t.Kind() == reflect.Pointer {
		return t.Elem()
	}
	return t
}

func (s *Scanner) structName() string {
	if t := s.structType(); t != nil {
		return t.Name()
	}
	return ""
}

func (s *Scanner) fail() {
	if s.err == nil {
		s.err = syntaxErrorOf(s.data)
	}
}

func syntaxErrorOf(data []byte) error {
	var raw stdjson.RawMessage
	if err := stdjson.Unmarshal(data, &raw); err != nil {
		return err
	}
	return errRejected
}

func (s *Scanner) fatal(err error) {
	if s.err == nil {
		s.err = err
	}
}

func (s *Scanner) save(err error) {
	if s.saved == nil {
		s.saved = err
	}
}

func kindWord(kind byte) string {
	switch kind {
	case '"':
		return "string"
	case '0':
		return "number"
	case 't', 'f':
		return "bool"
	case '[':
		return "array"
	}
	return "object"
}

func mismatchOffset(kind byte, start, end int) int {
	if kind == '[' || kind == '{' {
		return start + 1
	}
	return end
}

func (s *Scanner) typeError(value string, typ reflect.Type) *stdjson.UnmarshalTypeError {
	return &stdjson.UnmarshalTypeError{
		Value:  value,
		Type:   typ,
		Offset: int64(mismatchOffset(s.kind, s.start, s.end)),
		Struct: s.structName(),
		Field:  detach(s.key),
	}
}

func detach(key string) string {
	return strings.Clone(key)
}

func (s *Scanner) rejectKind(typ reflect.Type) {
	if s.saved == nil {
		s.saved = s.typeError(kindWord(s.kind), typ)
	}
}

func (s *Scanner) rejectNumber(lit string, typ reflect.Type) {
	if s.saved == nil {
		s.saved = s.typeError("number "+lit, typ)
	}
}

func (s *Scanner) indexNewError(before error, i int) {
	if indexedFieldPaths && before == nil && s.saved != nil {
		if ute, ok := s.saved.(*stdjson.UnmarshalTypeError); ok && len(ute.Field) >= len(s.key) && ute.Field[:len(s.key)] == s.key {
			ute.Field = s.key + "." + strconv.Itoa(i) + ute.Field[len(s.key):]
		}
	}
}

func (s *Scanner) scalar() byte {
	s.kind = 0
	if s.err != nil {
		return 0
	}
	i := skipSpace(s.data, s.pos)
	if i >= len(s.data) {
		s.fail()
		return 0
	}
	end := 0
	if s.data[i] == '"' {
		end, s.flags = scanString(s.data, i)
	} else {
		end = scanValue(s.data, i, s.depth)
	}
	if end < 0 {
		s.fail()
		return 0
	}
	s.start, s.end, s.pos = i, end, end
	s.kind = valueStart[s.data[i]]
	return s.kind
}

func (s *Scanner) stringBody() string {
	body := s.data[s.start+1 : s.end-1]
	if plainBody(body, s.flags) {
		return string(body)
	}
	out, _ := appendUnquoted(make([]byte, 0, len(body)), body)
	return unsafeString(out)
}

// Object opens the value as an object: it skips space and, if the next byte
// is '{', consumes it and returns true. Otherwise it returns false after
// fully validating the remaining value and the trailing space. JSON null
// leaves no error state, so the caller returns End() and UnmarshalJSON of
// null is a no-op. Any other valid value saves an *UnmarshalTypeError whose
// Type is the struct and whose Struct and Field are empty, as
// encoding/json reports it. Invalid JSON is a fatal syntax error.
func (s *Scanner) Object() bool {
	s.state = stateIdle
	i := skipSpace(s.data, s.pos)
	if i < len(s.data) && s.data[i] == '{' {
		s.pos = i + 1
		s.depth = 1
		s.state = stateFirst
		return true
	}
	s.rejectObject()
	return false
}

func (s *Scanner) rejectObject() {
	kind := s.scalar()
	if kind == 0 {
		return
	}
	if skipSpace(s.data, s.pos) != len(s.data) {
		s.fail()
		return
	}
	if kind == 'n' {
		return
	}
	s.save(&stdjson.UnmarshalTypeError{
		Value:  kindWord(kind),
		Type:   s.structType(),
		Offset: int64(mismatchOffset(kind, s.start, s.end)),
	})
}

func (s *Scanner) closeObject(i int) {
	s.pos = skipSpace(s.data, i)
	if s.pos != len(s.data) {
		s.fail()
		return
	}
	s.state = stateDone
}

func (s *Scanner) setKey(start, end, flags int) {
	body := s.data[start+1 : end-1]
	if plainBody(body, flags) {
		s.key = unsafeString(body)
		return
	}
	out, _ := appendUnquoted(make([]byte, 0, len(body)), body)
	s.key = unsafeString(out)
}

func (s *Scanner) member(i int) bool {
	if i >= len(s.data) || s.data[i] != '"' {
		s.fail()
		return false
	}
	end, flags := scanString(s.data, i)
	if end < 0 {
		s.fail()
		return false
	}
	j := skipSpace(s.data, end)
	if j >= len(s.data) || s.data[j] != ':' {
		s.fail()
		return false
	}
	s.setKey(i, end, flags)
	s.pos = skipSpace(s.data, j+1)
	s.state = stateMember
	return true
}

// Next advances to the next member of the object: it parses `"key":` and
// returns true, with the cursor on the value. The key is unescaped and
// validated, and invalid UTF-8 in it becomes U+FFFD like encoding/json. At
// the closing '}' it consumes it, checks that only space follows and
// returns false. A syntax error is recorded as the fatal error and also
// returns false.
func (s *Scanner) Next() bool {
	if s.err != nil || (s.state != stateFirst && s.state != stateMember) {
		return false
	}
	i := skipSpace(s.data, s.pos)
	if i >= len(s.data) {
		s.fail()
		return false
	}
	c := s.data[i]
	if c == '}' {
		s.closeObject(i + 1)
		return false
	}
	if s.state == stateMember {
		if c != ',' {
			s.fail()
			return false
		}
		i = skipSpace(s.data, i+1)
	}
	return s.member(i)
}

// Key returns the current key. It is valid after a true Next until the next
// Next, and it aliases the input when the key had no escapes and was valid
// UTF-8.
func (s *Scanner) Key() string {
	return s.key
}

func foldRune(r rune) rune {
	for {
		r2 := unicode.SimpleFold(r)
		if r2 <= r {
			return r2
		}
		r = r2
	}
}

func appendFoldedName(out []byte, in string) []byte {
	for i := 0; i < len(in); {
		if c := in[i]; c < utf8.RuneSelf {
			if 'a' <= c && c <= 'z' {
				c -= 'a' - 'A'
			}
			out = append(out, c)
			i++
			continue
		}
		r, n := utf8.DecodeRuneInString(in[i:])
		out = utf8.AppendRune(out, foldRune(r))
		i += n
	}
	return out
}

// FoldKey returns the case-folded form of the current key, the form
// encoding/json compares when it matches member names without regard to
// case: ASCII lower case becomes upper case and every other rune maps to
// the smallest rune of its unicode.SimpleFold orbit. The result is valid
// until the next call of FoldKey or Next and does not allocate for keys of
// up to 32 bytes.
func (s *Scanner) FoldKey() string {
	buf := appendFoldedName(s.fold[:0], s.key)
	if len(buf) == 0 {
		return ""
	}
	return unsafe.String(&buf[0], len(buf))
}

// Null consumes the current value and returns true if it is the literal
// null followed by a delimiter; otherwise it returns false and consumes
// nothing.
func (s *Scanner) Null() bool {
	i := skipSpace(s.data, s.pos)
	end := i + len("null")
	if end > len(s.data) || string(s.data[i:end]) != "null" {
		return false
	}
	if end < len(s.data) && !isDelimiter(s.data[end]) {
		return false
	}
	s.pos = end
	return true
}

// Skip consumes and fully validates the current value and ignores it, for
// a member the struct does not know.
func (s *Scanner) Skip() {
	s.scalar()
}

// Value consumes and validates the current JSON value, advances pos,
// and returns its raw byte span.
func (s *Scanner) Value() []byte {
	if s.scalar() == 0 {
		return nil
	}
	return s.data[s.start:s.end]
}

// Fallback consumes the current value, validated, and decodes it with
// encoding/json.Unmarshal into dst, a pointer to the field. An
// *UnmarshalTypeError is saved like the errors of the typed readers: the
// first one wins, decoding continues, Offset becomes relative to the whole
// input, Struct is set when empty and Field is the current key, followed
// by any path the error already carries. Any other error is fatal.
func (s *Scanner) Fallback(dst any) {
	if s.scalar() == 0 {
		return
	}
	s.decodeRaw(s.data[s.start:s.end], dst)
}

// FallbackQ decodes a json.Number field carrying the `string` option: the
// value must be a JSON string whose content is itself JSON, or null.
// dst points at the field and T is its type, which names the destination in
// the saved error. It unwraps the outer string before delegating to
// encoding/json, matching encoding/json's destring branch for Number. A bare
// non-string value saves the invalid use of ,string error; null decodes as
// null (pointer nil, value unchanged) via the same path as Fallback. A field
// whose type declares its own UnmarshalJSON, UnmarshalText or
// UnmarshalJSONFrom method uses the compiler-selected typed QuotedJSON,
// QuotedText or QuotedFrom helpers instead, which keep the profile-specific
// token and error policy without runtime method discovery.
func FallbackQ[T any](s *Scanner, dst *T) {
	item, st := s.readQuoted()
	switch st {
	case quotedBare:
		s.save(tagUnquoted(reflect.TypeFor[T]()))
	case quotedContent:
		s.decodeRaw([]byte(item), dst)
	default:
		if s.err == nil {
			s.decodeRaw(s.data[s.start:s.end], dst)
		}
	}
}

func (s *Scanner) decodeRaw(raw []byte, dst any) {
	err := stdjson.Unmarshal(raw, dst)
	if err == nil {
		return
	}
	if ute, ok := err.(*stdjson.UnmarshalTypeError); ok {
		s.saveFallback(ute)
		return
	}
	s.fatal(err)
}

func (s *Scanner) saveFallback(ute *stdjson.UnmarshalTypeError) {
	if s.saved != nil {
		return
	}
	ute.Offset += int64(s.start)
	ute.Struct = fallbackStructName(ute.Struct, s.structName())
	if ute.Field == "" {
		ute.Field = detach(s.key)
	} else {
		ute.Field = s.key + "." + ute.Field
	}
	s.saved = ute
}

// Err returns the fatal error: the syntax error, a *SyntaxError identical
// in message and offset to what encoding/json.Unmarshal reports for the
// same bytes, or any fatal error of Fallback. It is nil otherwise and
// sticky once set.
func (s *Scanner) Err() error {
	return s.err
}

// Saved returns the first saved, non-fatal error, normally an
// *UnmarshalTypeError, or nil if there was none.
func (s *Scanner) Saved() error {
	return s.saved
}

// End returns Err if it is not nil and Saved otherwise.
func (s *Scanner) End() error {
	if s.err != nil {
		return s.err
	}
	return s.saved
}

const (
	UnionTrialMiss = iota
	UnionTrialSingle
	UnionTrialAmbiguous
	UnionTrialNoMatch
)

type UnionTrialOutcome struct {
	Kind    int
	Winner  int
	Matched int
}

// A UnionTrial memoizes union trial outcomes for one root value: the raw
// bytes of the outermost union value in flight. It holds those bytes, so
// their address range stays allocated and fixed until the outermost trial
// returns, and a nested trial joins the root whose range contains its bytes.
// Decodes of separate buffers therefore share no outcomes.
//
// Outcomes are keyed by content: first by typeID, then by the raw bytes
// of the union value, so equal spans inside one root value share an
// outcome regardless of address. Store snapshots the bytes, so a later
// mutation of the caller's buffer can neither corrupt nor re-key an
// already stored entry. entries counts stored outcomes across all
// typeIDs and bytes counts their retained key payload; exceeding either
// budget drops the whole cache.
type UnionTrial struct {
	root     []byte
	depth    int
	mu       sync.Mutex
	outcomes map[string]map[string]UnionTrialOutcome
	entries  int
	bytes    int
}

var unionTrials struct {
	mu     sync.Mutex
	active []*UnionTrial
}

func addressRange(raw []byte) (lo, hi uintptr) {
	lo = uintptr(unsafe.Pointer(unsafe.SliceData(raw)))
	return lo, lo + uintptr(len(raw))
}

func EnterUnionTrial(raw []byte) *UnionTrial {
	lo, hi := addressRange(raw)
	unionTrials.mu.Lock()
	defer unionTrials.mu.Unlock()
	active := unionTrials.active
	at, _ := slices.BinarySearchFunc(active, lo, func(t *UnionTrial, lo uintptr) int {
		if rootLo, _ := addressRange(t.root); rootLo > lo {
			return 1
		}
		return -1
	})
	if at > 0 {
		if _, rootHi := addressRange(active[at-1].root); hi <= rootHi {
			active[at-1].depth++
			return active[at-1]
		}
	}
	t := &UnionTrial{root: raw, depth: 1}
	unionTrials.active = slices.Insert(active, at, t)
	return t
}

func (t *UnionTrial) Exit() {
	unionTrials.mu.Lock()
	defer unionTrials.mu.Unlock()
	t.depth--
	if t.depth == 0 {
		unionTrials.active = slices.DeleteFunc(unionTrials.active, func(o *UnionTrial) bool { return o == t })
	}
}

func (t *UnionTrial) Lookup(typeID string, raw []byte) (bool, UnionTrialOutcome) {
	t.mu.Lock()
	defer t.mu.Unlock()
	out, ok := t.outcomes[typeID][string(raw)]
	if !ok {
		return false, UnionTrialOutcome{Kind: UnionTrialMiss, Winner: -1}
	}
	return true, out
}

func (t *UnionTrial) Store(typeID string, raw []byte, out UnionTrialOutcome) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.outcomes == nil || t.entries > 1024 || t.bytes > 1<<20 {
		t.outcomes = make(map[string]map[string]UnionTrialOutcome)
		t.entries = 0
		t.bytes = 0
	}
	inner := t.outcomes[typeID]
	if inner == nil {
		inner = make(map[string]UnionTrialOutcome)
		id := strings.Clone(typeID)
		t.outcomes[id] = inner
		t.bytes += len(id)
	}
	old, dup := inner[string(raw)]
	if dup {
		if old != out {
			inner[string(raw)] = out
		}
		return
	}
	t.entries++
	t.bytes += len(raw)
	inner[string(raw)] = out
}
