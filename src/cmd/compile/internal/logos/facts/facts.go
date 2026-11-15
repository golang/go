// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"bytes"
	"cmp"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math/bits"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode"

	"cmd/internal/goobj"
)

const (
	Version    = 3
	MemberName = "__.LOGOS"

	arHeaderSize = 60
	arMagic      = "!<arch>\n"

	maxMemberBytes = 64 << 20
)

var versionLine = "logos-facts " + strconv.Itoa(Version)

type PathFact struct {
	Object string
	Path   string
}

type ReadonlyFact struct {
	Object string
	Path   string
}

type FnFact struct {
	Object          string
	AlwaysNilError  bool
	ReceiverMutated bool
	Readonly        bool
	AssertTP        string
	AssertRTP       string
	HasAssert       bool
	ParamMutated    string
	MayYield        bool
	HasMayYield     bool
}

func (fn FnFact) MayYieldOnly() bool {
	return fn.HasMayYield && !fn.ReceiverMutated && !fn.Readonly && fn.ParamMutated == ""
}

type PropFact struct {
	Interface string
	Prefix    string
}

type ConformsFact struct {
	Type      string
	IfacePath string
	Iface     string
}

type ComptimeFact struct {
	Object string
	Source []byte
}

type ConstFuncFact struct {
	Object string
	Source []byte
}

type ConstMapFact struct {
	Object  string
	Path    string
	Bearing bool
	Order   []int
	Hashes  []uint64
	Canons  []string
	Keys    []string
	Vals    []string
}

type LockFact struct {
	Object string
	Effect string
	Path   string
}

type AllocFreeFact struct {
	Object string
}

type Facts struct {
	Fingerprint  goobj.FingerprintType
	Opts         []PathFact
	Consts       []PathFact
	Fns          []FnFact
	Readonlys    []ReadonlyFact
	Props        []PropFact
	Conforms     []ConformsFact
	Comptimes    []ComptimeFact
	ConstFuncs   []ConstFuncFact
	ConstMaps    []ConstMapFact
	ConstComps   []string
	Defaults     []DefaultFact
	Locks        []LockFact
	AllocFrees   []AllocFreeFact
	JsonMetas    []JsonMetaFact
	JsonKeys     []JsonKeyFact
	UnionMembers []UnionMemberFact
	BareUnions   []string
	GoPackage    bool
}

func (f *Facts) empty() bool {
	return len(f.Opts) == 0 && len(f.Consts) == 0 && len(f.Fns) == 0 &&
		len(f.Readonlys) == 0 && len(f.Props) == 0 && len(f.Conforms) == 0 &&
		len(f.Comptimes) == 0 && len(f.ConstFuncs) == 0 && len(f.Defaults) == 0 &&
		len(f.Locks) == 0 && len(f.AllocFrees) == 0 && len(f.JsonMetas) == 0 &&
		len(f.JsonKeys) == 0 && len(f.UnionMembers) == 0 && len(f.BareUnions) == 0 && len(f.ConstMaps) == 0 && len(f.ConstComps) == 0 && !f.GoPackage
}

type DefaultFact struct {
	Type  string
	Field string
	Text  string
}

type JsonMetaFact struct {
	Type       string
	HasDecoder bool
	Eligible   bool
}

type JsonKeyFact struct {
	Type  string
	Index int
	Key   string
}

type UnionMemberFact struct {
	Type   string
	Index  int
	Member string
}

type factRecord struct {
	line string
	body []byte
}

func (f *Facts) jsonRecords() []factRecord {
	if f == nil || f.GoPackage {
		return nil
	}
	var out []factRecord
	for _, m := range f.JsonMetas {
		if m.Type == "" || !isIdent(m.Type) {
			continue
		}
		flags := ""
		if m.HasDecoder {
			flags = "decoder"
		}
		if m.Eligible {
			if flags != "" {
				flags += ","
			}
			flags += "eligible"
		}
		if flags == "" {
			continue
		}
		out = append(out, factRecord{line: "jsonmeta " + m.Type + " " + flags})
	}
	for _, k := range f.JsonKeys {
		if k.Type == "" || !isIdent(k.Type) || k.Index < 0 || k.Index > 9999 {
			continue
		}
		out = append(out, factRecord{line: "jsonkey " + k.Type + " " + paddedIndex(k.Index) + " " + strconv.Itoa(len(k.Key)), body: []byte(k.Key)})
	}
	for _, u := range f.UnionMembers {
		if u.Type == "" || !isIdent(u.Type) || u.Index < 0 || u.Index > 9999 || u.Member == "" {
			continue
		}
		out = append(out, factRecord{line: "unionmember " + u.Type + " " + paddedIndex(u.Index) + " " + u.Member})
	}
	return out
}

func paddedIndex(i int) string {
	s := strconv.Itoa(i)
	for len(s) < 4 {
		s = "0" + s
	}
	return s
}

func Encode(fp goobj.FingerprintType, f *Facts) []byte {
	var recs []factRecord
	if f != nil {
		for _, o := range f.Opts {
			if o.Object != "" && o.Path != "" {
				recs = append(recs, factRecord{line: "opt " + o.Object + " " + o.Path})
			}
		}
		for _, c := range f.Consts {
			if c.Object != "" && c.Path != "" {
				recs = append(recs, factRecord{line: "const " + c.Object + " " + c.Path})
			}
		}
		for _, fn := range f.Fns {
			if fl := fn.flags(); fn.Object != "" && fl != "" {
				recs = append(recs, factRecord{line: "fn " + fn.Object + " " + fl})
			}
		}
		for _, r := range f.Readonlys {
			if r.Object != "" && r.Path != "" {
				recs = append(recs, factRecord{line: "readonly " + r.Object + " " + r.Path})
			}
		}
		for _, p := range f.Props {
			if p.Interface != "" && p.Prefix != "" {
				recs = append(recs, factRecord{line: "prop " + p.Interface + " " + p.Prefix})
			}
		}
		for _, c := range f.Conforms {
			if c.Type != "" && c.IfacePath != "" && c.Iface != "" {
				recs = append(recs, factRecord{line: "conforms " + c.Type + " " + c.IfacePath + " " + c.Iface})
			}
		}
		for _, c := range f.Comptimes {
			if c.Object != "" && len(c.Source) > 0 {
				recs = append(recs, factRecord{line: "comptime " + c.Object + " " + strconv.Itoa(len(c.Source)), body: c.Source})
			}
		}
		for _, c := range f.ConstFuncs {
			if c.Object != "" && len(c.Source) > 0 {
				recs = append(recs, factRecord{line: "constfn " + c.Object + " " + strconv.Itoa(len(c.Source)), body: c.Source})
			}
		}
		for _, c := range f.ConstMaps {
			if body, ok := encodeConstMap(c); ok {
				recs = append(recs, factRecord{line: strings.TrimSpace("constmap " + c.Object + " " + strconv.Itoa(len(body)) + " " + c.Path), body: body})
			}
		}
		for _, o := range f.ConstComps {
			if ValidObject(o) {
				recs = append(recs, factRecord{line: "constcomp " + o})
			}
		}
		if f.GoPackage {
			recs = append(recs, factRecord{line: "lang go"})
		}
		for _, d := range f.Defaults {
			if d.Type != "" && d.Field != "" && d.Text != "" {
				recs = append(recs, factRecord{line: "default " + d.Type + " " + d.Field + " " + strconv.Itoa(len(d.Text)), body: []byte(d.Text)})
			}
		}
		for _, l := range f.Locks {
			if l.Object != "" && l.Effect != "" && l.Path != "" {
				recs = append(recs, factRecord{line: "locks " + l.Object + " " + l.Effect + " " + l.Path})
			}
		}
		for _, a := range f.AllocFrees {
			if a.Object != "" {
				recs = append(recs, factRecord{line: "allocfree " + a.Object})
			}
		}
		for _, u := range f.BareUnions {
			if isIdent(u) {
				recs = append(recs, factRecord{line: "unionbare " + u})
			}
		}
		recs = append(recs, f.jsonRecords()...)
	}
	slices.SortFunc(recs, func(a, b factRecord) int { return cmp.Compare(a.line, b.line) })
	var b bytes.Buffer
	b.WriteString(versionLine + "\n")
	fmt.Fprintf(&b, "fingerprint %x\n", fp[:])
	for _, r := range recs {
		b.WriteString(r.line + "\n")
		if r.body != nil {
			b.Write(r.body)
			b.WriteByte('\n')
		}
	}
	return b.Bytes()
}

func (fn FnFact) flags() string {
	var b strings.Builder
	add := func(s string) {
		if b.Len() > 0 {
			b.WriteByte(',')
		}
		b.WriteString(s)
	}
	if fn.AlwaysNilError {
		add("AlwaysNilError")
	}
	if fn.ReceiverMutated {
		add("ReceiverMutated")
	}
	if fn.Readonly {
		add("Readonly")
	}
	if fn.HasAssert && fn.AssertTP != "" && fn.AssertRTP != "" {
		add("AssertTP=" + normHex(fn.AssertTP))
		add("AssertRTP=" + normHex(fn.AssertRTP))
	}
	if fn.ParamMutated != "" {
		add("ParamMutated=" + normHex(fn.ParamMutated))
	}
	if fn.HasMayYield {
		if fn.MayYield {
			add("MayYield=1")
		} else {
			add("MayYield=0")
		}
	}
	return b.String()
}

func (fn *FnFact) parseFlag(flag string) error {
	switch flag {
	case "AlwaysNilError":
		fn.AlwaysNilError = true
	case "ReceiverMutated":
		fn.ReceiverMutated = true
	case "Readonly":
		fn.Readonly = true
	default:
		if rest, ok := strings.CutPrefix(flag, "ParamMutated="); ok {
			if !isHexMask(rest) {
				return fmt.Errorf("bad ParamMutated flag %q (want hex mask)", flag)
			}
			fn.ParamMutated = normHex(rest)
			return nil
		}
		if rest, ok := strings.CutPrefix(flag, "AssertTP="); ok {
			if !isHexMask(rest) {
				return fmt.Errorf("bad AssertTP flag %q (want hex mask)", flag)
			}
			fn.AssertTP = normHex(rest)
			fn.HasAssert = true
			return nil
		}
		if rest, ok := strings.CutPrefix(flag, "AssertRTP="); ok {
			if !isHexMask(rest) {
				return fmt.Errorf("bad AssertRTP flag %q (want hex mask)", flag)
			}
			fn.AssertRTP = normHex(rest)
			fn.HasAssert = true
			return nil
		}
		if rest, ok := strings.CutPrefix(flag, "MayYield="); ok {
			if rest != "0" && rest != "1" {
				return fmt.Errorf("bad MayYield flag %q (want 0 or 1)", flag)
			}
			fn.HasMayYield = true
			fn.MayYield = rest == "1"
			return nil
		}
		return fmt.Errorf("unknown fn flag %q", flag)
	}
	return nil
}

func HighestAssertBit(s string) (int, bool) {
	if s == "" {
		return 0, false
	}
	if strings.Trim(s, "0123456789abcdef") != "" {
		return 0, false
	}
	trimmed := strings.TrimLeft(s, "0")
	if trimmed == "" {
		return -1, true
	}
	top := 4*(len(trimmed)-1) + bits.Len8(uint8(nibbleValue(trimmed[0]))) - 1
	return top, true
}

func nibbleValue(c byte) int {
	if c > '9' {
		return int(c - 'a' + 10)
	}
	return int(c - '0')
}

func isHexMask(s string) bool {
	return s != "" && strings.Trim(s, "0123456789abcdefABCDEF") == ""
}

func normHex(s string) string {
	s = strings.ToLower(s)
	s = strings.TrimLeft(s, "0")
	if s == "" {
		s = "0"
	}
	return s
}

type decoder struct {
	data []byte
	off  int
	line int
	seen map[string]int
}

func (d *decoder) dup(ln int, kind, key, display string) error {
	if d.seen == nil {
		d.seen = make(map[string]int)
	}
	k := kind + "\x00" + key
	if first, ok := d.seen[k]; ok {
		return fmt.Errorf("logos facts line %d: duplicate %s record for %q (first on line %d)", ln, kind, display, first)
	}
	d.seen[k] = ln
	return nil
}

func (d *decoder) header() (ln int, s string, err error) {
	ln = d.line
	i := bytes.IndexByte(d.data[d.off:], '\n')
	if i < 0 {
		return ln, "", fmt.Errorf("logos facts line %d: truncated record %q", ln, d.data[d.off:])
	}
	s = string(d.data[d.off : d.off+i])
	d.off += i + 1
	d.line++
	return ln, s, nil
}

func (d *decoder) takeBody(ln int, kind, object string, n int) ([]byte, error) {
	return d.body(ln, kind, object, n, false)
}

func (d *decoder) requireBody(ln int, kind, object string, n int) ([]byte, error) {
	return d.body(ln, kind, object, n, true)
}

func (d *decoder) body(ln int, kind, object string, n int, strictNL bool) ([]byte, error) {
	if len(d.data)-d.off < n {
		return nil, fmt.Errorf("logos facts line %d: truncated %s body for %q (want %d bytes, have %d)", ln, kind, object, n, len(d.data)-d.off)
	}
	src := bytes.Clone(d.data[d.off : d.off+n])
	d.off += n
	d.line += bytes.Count(src, []byte{'\n'})
	atEOF := d.off >= len(d.data)
	if strictNL && atEOF {
		return nil, fmt.Errorf("logos facts line %d: want newline after %s body", ln, kind)
	}
	if !atEOF {
		if d.data[d.off] != '\n' {
			return nil, fmt.Errorf("logos facts line %d: want newline after %s body", ln, kind)
		}
		d.off++
		d.line++
	}
	return src, nil
}
func decodeJSONFact(f *Facts, d *decoder, ln int, line string) (bool, error) {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return false, nil
	}
	switch fields[0] {
	case "jsonmeta":
		if len(fields) != 3 {
			return true, fmt.Errorf(`logos facts line %d: bad jsonmeta record %q (want "jsonmeta <type> <flag>[,<flag>]")`, ln, line)
		}
		if !isIdent(fields[1]) {
			return true, fmt.Errorf("logos facts line %d: bad type %q", ln, fields[1])
		}
		meta := JsonMetaFact{Type: fields[1]}
		for flag := range strings.SplitSeq(fields[2], ",") {
			if flag == "decoder" {
				meta.HasDecoder = true
			} else if flag == "eligible" {
				meta.Eligible = true
			} else {
				return true, fmt.Errorf("logos facts line %d: bad jsonmeta flag %q", ln, flag)
			}
		}
		if !meta.HasDecoder && !meta.Eligible {
			return true, fmt.Errorf("logos facts line %d: empty jsonmeta flags %q", ln, line)
		}
		if err := d.dup(ln, "jsonmeta", fields[1], fields[1]); err != nil {
			return true, err
		}
		f.JsonMetas = append(f.JsonMetas, meta)
		return true, nil
	case "jsonkey":
		if len(fields) != 4 {
			return true, fmt.Errorf(`logos facts line %d: bad jsonkey record %q (want "jsonkey <type> <index> <length>")`, ln, line)
		}
		if !isIdent(fields[1]) {
			return true, fmt.Errorf("logos facts line %d: bad type %q", ln, fields[1])
		}
		if !isDigits(fields[2]) {
			return true, fmt.Errorf("logos facts line %d: bad jsonkey index %q", ln, fields[2])
		}
		n, aerr := strconv.Atoi(fields[3])
		if aerr != nil || n < 0 {
			return true, fmt.Errorf("logos facts line %d: bad jsonkey length %q", ln, fields[3])
		}
		index, _ := strconv.Atoi(fields[2])
		if err := d.dup(ln, "jsonkey", fields[1]+"\x00"+strconv.Itoa(index), fields[1]+" "+fields[2]); err != nil {
			return true, err
		}
		key, err := d.takeBody(ln, "jsonkey", fields[1], n)
		if err != nil {
			return true, err
		}
		f.JsonKeys = append(f.JsonKeys, JsonKeyFact{Type: fields[1], Index: index, Key: string(key)})
		return true, nil
	case "unionmember":
		if len(fields) != 4 {
			return true, fmt.Errorf(`logos facts line %d: bad unionmember record %q (want "unionmember <type> <index> <member>")`, ln, line)
		}
		if !isIdent(fields[1]) {
			return true, fmt.Errorf("logos facts line %d: bad type %q", ln, fields[1])
		}
		if !isDigits(fields[2]) {
			return true, fmt.Errorf("logos facts line %d: bad unionmember index %q", ln, fields[2])
		}
		if fields[3] == "" {
			return true, fmt.Errorf("logos facts line %d: bad unionmember %q", ln, fields[3])
		}
		index, _ := strconv.Atoi(fields[2])
		if err := d.dup(ln, "unionmember", fields[1]+"\x00"+strconv.Itoa(index), fields[1]+" "+fields[2]); err != nil {
			return true, err
		}
		f.UnionMembers = append(f.UnionMembers, UnionMemberFact{Type: fields[1], Index: index, Member: fields[3]})
		return true, nil
	}
	return false, nil
}

func (d *decoder) preamble(f *Facts) error {
	_, magic, err := d.header()
	if err != nil {
		if len(d.data) == 0 {
			return fmt.Errorf("logos facts line 1: empty facts member, want %q", versionLine)
		}
		return err
	}
	if magic != versionLine {
		return fmt.Errorf("logos facts line 1: unsupported version %q (want %q)", magic, versionLine)
	}

	_, fpLine, err := d.header()
	if err != nil {
		return err
	}
	fpFields := strings.Fields(fpLine)
	if len(fpFields) != 2 || fpFields[0] != "fingerprint" {
		return fmt.Errorf(`logos facts line 2: bad fingerprint line %q (want "fingerprint <hex>")`, fpLine)
	}
	raw, err := hex.DecodeString(fpFields[1])
	if err != nil || len(raw) != len(f.Fingerprint) {
		return fmt.Errorf("logos facts line 2: bad fingerprint %q (want %d hex chars)", fpFields[1], 2*len(f.Fingerprint))
	}
	copy(f.Fingerprint[:], raw)
	return nil
}

func Decode(data []byte) (*Facts, error) {
	d := &decoder{data: data, line: 1}
	f := &Facts{}
	if err := d.preamble(f); err != nil {
		return nil, err
	}

	prev := ""
	for d.off < len(d.data) {
		ln, line, err := d.header()
		if err != nil {
			return nil, err
		}
		if line == "" {
			return nil, fmt.Errorf("logos facts line %d: blank line", ln)
		}
		if line <= prev {
			return nil, fmt.Errorf("logos facts line %d: unsorted records (%q after %q)", ln, line, prev)
		}
		prev = line
		if handled, herr := decodeJSONFact(f, d, ln, line); handled {
			if herr != nil {
				return nil, herr
			}
			continue
		}
		fields := strings.Fields(line)
		switch {
		case strings.HasPrefix(line, "opt ") || strings.HasPrefix(line, "const ") || strings.HasPrefix(line, "readonly "):
			if len(fields) != 3 {
				return nil, fmt.Errorf("logos facts line %d: bad %s record %q (want %q)", ln, fields[0], line, fields[0]+" <object> <path>")
			}
			if !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad object %q", ln, fields[1])
			}
			if !validPath(fields[2]) {
				return nil, fmt.Errorf("logos facts line %d: bad path %q", ln, fields[2])
			}
			if err := d.dup(ln, fields[0], fields[1]+"\x00"+fields[2], fields[1]+" "+fields[2]); err != nil {
				return nil, err
			}
			switch fields[0] {
			case "opt":
				f.Opts = append(f.Opts, PathFact{Object: fields[1], Path: fields[2]})
			case "const":
				f.Consts = append(f.Consts, PathFact{Object: fields[1], Path: fields[2]})
			default:
				f.Readonlys = append(f.Readonlys, ReadonlyFact{Object: fields[1], Path: fields[2]})
			}
		case line == "lang go":
			f.GoPackage = true
		case strings.HasPrefix(line, "locks "):
			if len(fields) != 4 {
				return nil, fmt.Errorf(`logos facts line %d: bad locks record %q (want "locks <object> <requires|acquires|releases> <path>")`, ln, line)
			}
			if !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad object %q", ln, fields[1])
			}
			switch fields[2] {
			case "requires", "acquires", "releases":
			default:
				return nil, fmt.Errorf("logos facts line %d: bad lock effect %q (want requires, acquires, or releases)", ln, fields[2])
			}
			if !validLockPath(fields[3]) {
				return nil, fmt.Errorf("logos facts line %d: bad lock path %q", ln, fields[3])
			}
			if err := d.dup(ln, "locks", fields[1]+"\x00"+fields[2]+"\x00"+fields[3], fields[1]+" "+fields[2]+" "+fields[3]); err != nil {
				return nil, err
			}
			f.Locks = append(f.Locks, LockFact{Object: fields[1], Effect: fields[2], Path: fields[3]})
		case strings.HasPrefix(line, "allocfree "):
			if len(fields) != 2 {
				return nil, fmt.Errorf(`logos facts line %d: bad allocfree record %q (want "allocfree <object>")`, ln, line)
			}
			if !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad object %q", ln, fields[1])
			}
			if err := d.dup(ln, "allocfree", fields[1], fields[1]); err != nil {
				return nil, err
			}
			f.AllocFrees = append(f.AllocFrees, AllocFreeFact{Object: fields[1]})
		case strings.HasPrefix(line, "unionbare "):
			if len(fields) != 2 {
				return nil, fmt.Errorf(`logos facts line %d: bad unionbare record %q (want "unionbare <type>")`, ln, line)
			}
			if !isIdent(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad type %q", ln, fields[1])
			}
			if err := d.dup(ln, "unionbare", fields[1], fields[1]); err != nil {
				return nil, err
			}
			f.BareUnions = append(f.BareUnions, fields[1])
		case strings.HasPrefix(line, "fn "):
			if len(fields) != 3 {
				return nil, fmt.Errorf("logos facts line %d: bad fn record %q (want %q)", ln, line, "fn <object> <flag>[,<flag>...]")
			}
			var fn FnFact
			fn.Object = fields[1]
			if !ValidObject(fn.Object) {
				return nil, fmt.Errorf("logos facts line %d: bad object %q", ln, fn.Object)
			}
			for _, flag := range strings.Split(fields[2], ",") {
				if ferr := fn.parseFlag(flag); ferr != nil {
					return nil, fmt.Errorf("logos facts line %d: %v", ln, ferr)
				}
			}
			if (fn.AssertTP == "") != (fn.AssertRTP == "") {
				return nil, fmt.Errorf("logos facts line %d: incomplete assertion-effect pair on %q (want both AssertTP and AssertRTP or neither)", ln, fn.Object)
			}
			if err := d.dup(ln, "fn", fn.Object, fn.Object); err != nil {
				return nil, err
			}
			f.Fns = append(f.Fns, fn)
		case strings.HasPrefix(line, "prop "):
			if len(fields) != 3 {
				return nil, fmt.Errorf(`logos facts line %d: bad prop record %q (want "prop <interface> <prefix>")`, ln, line)
			}
			if !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad interface %q", ln, fields[1])
			}
			if !validPrefix(fields[2]) {
				return nil, fmt.Errorf("logos facts line %d: bad accessor prefix %q", ln, fields[2])
			}
			if err := d.dup(ln, "prop", fields[1], fields[1]); err != nil {
				return nil, err
			}
			f.Props = append(f.Props, PropFact{Interface: fields[1], Prefix: fields[2]})
		case strings.HasPrefix(line, "conforms "):
			if len(fields) != 4 {
				return nil, fmt.Errorf(`logos facts line %d: bad conforms record %q (want "conforms <struct> <package> <interface>")`, ln, line)
			}
			if !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad struct %q", ln, fields[1])
			}
			if fields[2] == "" {
				return nil, fmt.Errorf("logos facts line %d: bad package %q", ln, fields[2])
			}
			if !isIdent(fields[3]) {
				return nil, fmt.Errorf("logos facts line %d: bad interface %q", ln, fields[3])
			}
			if err := d.dup(ln, "conforms", fields[1]+"\x00"+fields[2]+"\x00"+fields[3], fields[1]+" "+fields[2]+" "+fields[3]); err != nil {
				return nil, err
			}
			f.Conforms = append(f.Conforms, ConformsFact{Type: fields[1], IfacePath: fields[2], Iface: fields[3]})
		case strings.HasPrefix(line, "default "):
			if len(fields) != 4 {
				return nil, fmt.Errorf(`logos facts line %d: bad default record %q (want "default <type> <field> <length>")`, ln, line)
			}
			if !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad type %q", ln, fields[1])
			}
			if !isIdent(fields[2]) {
				return nil, fmt.Errorf("logos facts line %d: bad field %q", ln, fields[2])
			}
			n, aerr := strconv.Atoi(fields[3])
			if aerr != nil || n < 0 {
				return nil, fmt.Errorf("logos facts line %d: bad default length %q", ln, fields[3])
			}
			if err := d.dup(ln, "default", fields[1]+"\x00"+fields[2], fields[1]+" "+fields[2]); err != nil {
				return nil, err
			}
			text, err := d.takeBody(ln, "default", fields[1], n)
			if err != nil {
				return nil, err
			}
			f.Defaults = append(f.Defaults, DefaultFact{Type: fields[1], Field: fields[2], Text: string(text)})
		case strings.HasPrefix(line, "comptime "):
			if len(fields) != 3 {
				return nil, fmt.Errorf(`logos facts line %d: bad comptime record %q (want "comptime <object> <length>")`, ln, line)
			}
			if !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad object %q", ln, fields[1])
			}
			n, aerr := strconv.Atoi(fields[2])
			if aerr != nil || n < 0 {
				return nil, fmt.Errorf("logos facts line %d: bad comptime length %q", ln, fields[2])
			}
			if err := d.dup(ln, "comptime", fields[1], fields[1]); err != nil {
				return nil, err
			}
			src, err := d.takeBody(ln, "comptime", fields[1], n)
			if err != nil {
				return nil, err
			}
			f.Comptimes = append(f.Comptimes, ComptimeFact{Object: fields[1], Source: src})
		case strings.HasPrefix(line, "constfn "):
			if len(fields) != 3 {
				return nil, fmt.Errorf(`logos facts line %d: bad constfn record %q (want "constfn <object> <length>")`, ln, line)
			}
			if !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad object %q", ln, fields[1])
			}
			n, aerr := strconv.Atoi(fields[2])
			if aerr != nil || n <= 0 {
				return nil, fmt.Errorf("logos facts line %d: bad constfn length %q", ln, fields[2])
			}
			if err := d.dup(ln, "constfn", fields[1], fields[1]); err != nil {
				return nil, err
			}
			src, err := d.requireBody(ln, "constfn", fields[1], n)
			if err != nil {
				return nil, err
			}
			f.ConstFuncs = append(f.ConstFuncs, ConstFuncFact{Object: fields[1], Source: src})
		case strings.HasPrefix(line, "constmap "):
			if len(fields) != 3 && len(fields) != 4 {
				return nil, fmt.Errorf(`logos facts line %d: bad constmap record %q (want "constmap <object> <length> [<path>]")`, ln, line)
			}
			if !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad object %q", ln, fields[1])
			}
			n, aerr := strconv.Atoi(fields[2])
			if aerr != nil || n <= 0 {
				return nil, fmt.Errorf("logos facts line %d: bad constmap length %q", ln, fields[2])
			}
			path := strings.Join(fields[3:], "")
			if path != "" && !validCompositePath(path) {
				return nil, fmt.Errorf("logos facts line %d: bad constmap path %q", ln, path)
			}
			if err := d.dup(ln, "constmap", fields[1]+"\x00"+path, fields[1]+path); err != nil {
				return nil, err
			}
			body, err := d.requireBody(ln, "constmap", fields[1], n)
			if err != nil {
				return nil, err
			}
			cm, err := decodeConstMap(fields[1], body)
			if err != nil {
				return nil, fmt.Errorf("logos facts line %d: %v", ln, err)
			}
			cm.Path = path
			f.ConstMaps = append(f.ConstMaps, cm)
		case strings.HasPrefix(line, "constcomp "):
			if len(fields) != 2 || !ValidObject(fields[1]) {
				return nil, fmt.Errorf("logos facts line %d: bad constcomp record %q", ln, line)
			}
			if err := d.dup(ln, "constcomp", fields[1], fields[1]); err != nil {
				return nil, err
			}
			f.ConstComps = append(f.ConstComps, fields[1])
		default:
			kind := line
			if len(fields) > 0 {
				kind = fields[0]
			}
			return nil, fmt.Errorf("logos facts line %d: unknown record %q", ln, kind)
		}
	}
	return f, nil
}

func isIdent(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if r == '_' || unicode.IsLetter(r) {
			continue
		}
		if i > 0 && unicode.IsDigit(r) {
			continue
		}
		return false
	}
	return true
}

func ValidObject(s string) bool {
	recv, tag, hasTag := strings.Cut(s, "@")
	if hasTag && !isIdent(tag) {
		return false
	}
	if before, after, ok := strings.CutLast(recv, "."); ok {
		if !isIdent(after) {
			return false
		}
		recv = before
	}
	if recv == "" {
		return false
	}
	if isIdent(recv) {
		return true
	}
	if len(recv) >= 2 && recv[0] == '(' && recv[len(recv)-1] == ')' {
		inner := strings.TrimPrefix(recv[1:len(recv)-1], "*")
		return isIdent(inner)
	}
	return false
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := range s {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

func validStep(s string) bool {
	if s == "" {
		return false
	}
	switch s[0] {
	case 'R', 'P', 'T', 'G', 'H':
		return isDigits(s[1:])
	case 'F', 'M':
		return isIdent(s[1:])
	case 'D', 'E', 'K':
		return len(s) == 1
	}
	return false
}

func validPath(s string) bool {
	if s == "" {
		return false
	}
	for step := range strings.SplitSeq(s, ".") {
		if !validStep(step) {
			return false
		}
	}
	return true
}

func validCompositePath(s string) bool {
	if s == "" {
		return false
	}
	for s != "" {
		switch s[0] {
		case '.':
			end := strings.IndexAny(s[1:], ".[")
			if end < 0 {
				end = len(s) - 1
			}
			if !isIdent(s[1 : 1+end]) {
				return false
			}
			s = s[1+end:]
		case '[':
			end := strings.IndexByte(s, ']')
			if end < 2 || !isDigits(s[1:end]) || (end > 2 && s[1] == '0') {
				return false
			}
			s = s[end+1:]
		default:
			return false
		}
	}
	return true
}

func validLockPath(s string) bool {
	if s == "" {
		return false
	}
	for step := range strings.SplitSeq(s, ".") {
		if !isIdent(step) {
			return false
		}
	}
	return true
}
func validPrefix(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r != '.' && r != '_' && !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

var (
	currentMu    sync.RWMutex
	currentFacts *Facts

	importedMu sync.RWMutex
	imported   = make(map[string]*Facts)
)

func Current() *Facts {
	currentMu.RLock()
	defer currentMu.RUnlock()
	return currentFacts
}

func SetCurrent(f *Facts) {
	currentMu.Lock()
	defer currentMu.Unlock()
	currentFacts = f
}

func Import(path, exportData string, member []byte) (*Facts, error) {
	f, err := Decode(member)
	if err != nil {
		return nil, fmt.Errorf("logos: bad facts for %s: %v", path, err)
	}
	var want goobj.FingerprintType
	if len(exportData) < len(want) {
		return nil, fmt.Errorf("logos: stale facts for %s (export data too short for fingerprint)", path)
	}
	copy(want[:], exportData[len(exportData)-len(want):])
	if f.Fingerprint != want {
		return nil, fmt.Errorf("logos: stale facts for %s (fingerprint mismatch)", path)
	}
	importedMu.Lock()
	defer importedMu.Unlock()
	imported[path] = f
	return f, nil
}

func Lookup(path string) (*Facts, bool) {
	importedMu.RLock()
	defer importedMu.RUnlock()
	f, ok := imported[path]
	return f, ok
}

func FindMember(f *os.File) ([]byte, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("logos: cannot seek facts archive: %v", err)
	}
	var sig [len(arMagic)]byte
	if _, err := io.ReadFull(f, sig[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, nil
		}
		return nil, fmt.Errorf("logos: cannot read facts archive: %v", err)
	}
	if string(sig[:]) != arMagic {
		return nil, nil
	}
	for {
		var hdr [arHeaderSize]byte
		_, err := io.ReadFull(f, hdr[:])
		if errors.Is(err, io.EOF) {
			return nil, nil
		}
		if err != nil {
			return nil, fmt.Errorf("logos: cannot read facts archive header: %v", err)
		}
		if hdr[58] != '`' || hdr[59] != '\n' {
			return nil, fmt.Errorf("logos: corrupt facts archive header")
		}
		name := strings.TrimRight(string(hdr[:16]), " ")
		size, err := strconv.ParseUint(strings.TrimRight(string(hdr[48:58]), " "), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("logos: corrupt facts archive size for %q", name)
		}
		if name == MemberName {
			if size > maxMemberBytes {
				return nil, fmt.Errorf("logos: facts member too large (%d bytes)", size)
			}
			data := make([]byte, size)
			if _, err := io.ReadFull(f, data); err != nil {
				return nil, fmt.Errorf("logos: truncated facts member: %v", err)
			}
			return data, nil
		}
		if _, err := f.Seek(int64(size+(size&1)), io.SeekCurrent); err != nil {
			return nil, fmt.Errorf("logos: cannot skip facts archive member %q: %v", name, err)
		}
	}
}
