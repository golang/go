// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts_test

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/internal/goobj"
)

func TestEncodeDecodeRoundTrip(t *testing.T) {
	fp := goobj.FingerprintType{0, 1, 2, 3, 4, 5, 6, 7}
	in := &facts.Facts{
		Fingerprint: fp,
		Opts: []facts.PathFact{
			{Object: "S.F", Path: "R0.FName"},
			{Object: "H", Path: "P1"},
		},
		Consts: []facts.PathFact{
			{Object: "C", Path: "K"},
		},
		Fns: []facts.FnFact{
			{Object: "S.Set", ReceiverMutated: true, ParamMutated: "1f", MayYield: true, HasMayYield: true},
			{Object: "(*S).Get", ReturnsNonNil: true, PureConst: true},
		},
		Props: []facts.PropFact{
			{Interface: "I", Prefix: "Get"},
		},
		Comptimes: []facts.ComptimeFact{
			{Object: "New", Source: []byte("line1\nline2\n")},
		},
	}
	want := "logos-facts 1\n" +
		"fingerprint 0001020304050607\n" +
		"comptime New 12\n" +
		"line1\nline2\n" +
		"\n" +
		"const C K\n" +
		"fn (*S).Get ReturnsNonNil,PureConst\n" +
		"fn S.Set ReceiverMutated,ParamMutated=1f,MayYield=1\n" +
		"opt H P1\n" +
		"opt S.F R0.FName\n" +
		"prop I Get\n"
	got := facts.Encode(fp, in)
	if string(got) != want {
		t.Fatalf("Encode mismatch:\n got %q\nwant %q", got, want)
	}
	back, err := facts.Decode(got)
	if err != nil {
		t.Fatalf("Decode(Encode) failed: %v", err)
	}
	wantBack := &facts.Facts{
		Fingerprint: fp,
		Opts: []facts.PathFact{
			{Object: "H", Path: "P1"},
			{Object: "S.F", Path: "R0.FName"},
		},
		Consts: []facts.PathFact{
			{Object: "C", Path: "K"},
		},
		Fns: []facts.FnFact{
			{Object: "(*S).Get", ReturnsNonNil: true, PureConst: true},
			{Object: "S.Set", ReceiverMutated: true, ParamMutated: "1f", MayYield: true, HasMayYield: true},
		},
		Props: []facts.PropFact{
			{Interface: "I", Prefix: "Get"},
		},
		Comptimes: []facts.ComptimeFact{
			{Object: "New", Source: []byte("line1\nline2\n")},
		},
	}
	if !reflect.DeepEqual(back, wantBack) {
		t.Fatalf("round trip mismatch:\n got %+v\nwant %+v", back, wantBack)
	}
}

func TestEncodeNilHeaderOnly(t *testing.T) {
	fp := goobj.FingerprintType{7, 6, 5, 4, 3, 2, 1, 0}
	got := facts.Encode(fp, nil)
	if string(got) != "logos-facts 1\nfingerprint 0706050403020100\n" {
		t.Fatalf("header-only Encode mismatch: %q", got)
	}
	back, err := facts.Decode(got)
	if err != nil {
		t.Fatalf("Decode(header-only) failed: %v", err)
	}
	if !reflect.DeepEqual(back, &facts.Facts{Fingerprint: fp}) {
		t.Fatalf("header-only round trip mismatch: %+v", back)
	}
}

func TestEncodeSkipsVacuousRecords(t *testing.T) {
	fp := goobj.FingerprintType{1}
	in := &facts.Facts{
		Fingerprint: fp,
		Opts:        []facts.PathFact{{Object: "", Path: "P0"}, {Object: "H"}},
		Consts:      []facts.PathFact{{Object: "C", Path: ""}},
		Fns:         []facts.FnFact{{Object: "N"}, {Object: "", ReturnsNonNil: true}},
		Props:       []facts.PropFact{{Interface: "I"}, {Prefix: "Get"}},
		Comptimes:   []facts.ComptimeFact{{Object: "New"}, {Source: []byte("x")}},
	}
	got := facts.Encode(fp, in)
	if string(got) != "logos-facts 1\nfingerprint 0100000000000000\n" {
		t.Fatalf("vacuous records not skipped: %q", got)
	}
	if _, err := facts.Decode(got); err != nil {
		t.Fatalf("Decode(skipped) failed: %v", err)
	}
}

func TestDecodeRejects(t *testing.T) {
	head := "logos-facts 1\nfingerprint 0001020304050607\n"
	cases := []struct{ name, input, line string }{
		{"empty", "", "line 1"},
		{"wrong version", "logos-facts 2\nfingerprint 0001020304050607\n", "line 1"},
		{"bad magic", "hello\n", "line 1"},
		{"missing fingerprint", "logos-facts 1\n", "line 2"},
		{"bad fingerprint", head[:14] + "fingerprint zz\n", "line 2"},
		{"short fingerprint", head[:14] + "fingerprint 00\n", "line 2"},
		{"fingerprint missing hex", head[:14] + "fingerprint\n", "line 2"},
		{"unknown kind", head + "union U A|B\n", "line 3"},
		{"opt missing path", head + "opt H\n", "line 3"},
		{"bad object numeric", head + "opt 9lives P0\n", "line 3"},
		{"bad object qualified", head + "opt A.B.C P0\n", "line 3"},
		{"bad object empty member", head + "opt T. P0\n", "line 3"},
		{"bad object empty receiver", head + "opt .M P0\n", "line 3"},
		{"bad object star receiver", head + "opt (*).M P0\n", "line 3"},
		{"bad path bare R", head + "opt H R\n", "line 3"},
		{"bad path bare F", head + "opt H F\n", "line 3"},
		{"bad path unknown step", head + "opt H Q1\n", "line 3"},
		{"bad path trailing dot", head + "opt H P0.\n", "line 3"},
		{"bad path leading dot", head + "opt H .P0\n", "line 3"},
		{"bad path indexed E", head + "opt H E2\n", "line 3"},
		{"bad path negative", head + "opt H R-1\n", "line 3"},
		{"bad flag unknown", head + "fn S Bogus\n", "line 3"},
		{"bad flag MayYield", head + "fn S MayYield=2\n", "line 3"},
		{"bad flag ParamMutated", head + "fn S ParamMutated=zz\n", "line 3"},
		{"bad flag ParamMutated empty", head + "fn S ParamMutated=\n", "line 3"},
		{"fn missing flags", head + "fn S\n", "line 3"},
		{"unsorted", head + "opt S.F R0\nfn S ReturnsNonNil\n", "line 4"},
		{"duplicate", head + "opt H P1\nopt H P1\n", "line 4"},
		{"blank line", head + "opt H P1\n\nopt S.F R0\n", "line 4"},
		{"unterminated", head + "opt H P1", "line 3"},
		{"bad comptime length", head + "comptime New xyz\n", "line 3"},
		{"negative comptime length", head + "comptime New -1\n", "line 3"},
		{"truncated comptime", head + "comptime New 10\na\nb\n", "line 3"},
		{"bad prop prefix", head + "prop I get-x\n", "line 3"},
		{"conforms missing iface", head + "conforms Orc example.com/l/p\n", "line 3"},
		{"conforms extra field", head + "conforms Orc example.com/l/p Creature x\n", "line 3"},
		{"conforms bad struct", head + "conforms T. example.com/l/p Creature\n", "line 3"},
		{"conforms bad interface", head + "conforms Orc example.com/l/p 9lives\n", "line 3"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := facts.Decode([]byte(c.input))
			if err == nil {
				t.Fatalf("Decode(%q) succeeded, want error naming %s", c.input, c.line)
			}
			if !strings.Contains(err.Error(), c.line) {
				t.Fatalf("Decode(%q) error %q does not name %s", c.input, err, c.line)
			}
		})
	}
}

func TestDecodeLineNumbersAfterComptimeBody(t *testing.T) {
	in := "logos-facts 1\nfingerprint 0001020304050607\ncomptime New 5\nab\ncd\nopt H Q1\n"
	_, err := facts.Decode([]byte(in))
	if err == nil || !strings.Contains(err.Error(), "line 6") {
		t.Fatalf("Decode error = %v, want error naming line 6", err)
	}
}

func TestGoPackageRecord(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	in := &facts.Facts{
		Fingerprint: fp,
		GoPackage:   true,
		Fns:         []facts.FnFact{{Object: "F", MayYield: true, HasMayYield: true}},
	}
	got := string(facts.Encode(fp, in))
	want := "logos-facts 1\nfingerprint 0102030405060708\nfn F MayYield=1\nlang go\n"
	if got != want {
		t.Fatalf("Encode mismatch:\n got %q\nwant %q", got, want)
	}
	back, err := facts.Decode([]byte(got))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if !back.GoPackage || len(back.Fns) != 1 || !back.Fns[0].MayYield {
		t.Fatalf("round trip lost a field: %+v", back)
	}
	loGos, err := facts.Decode(facts.Encode(fp, &facts.Facts{Fns: in.Fns}))
	if err != nil || loGos.GoPackage {
		t.Fatalf("a package without the record must not decode as Go: %+v, %v", loGos, err)
	}
}

func TestMayYieldOnly(t *testing.T) {
	for name, c := range map[string]struct {
		fn   facts.FnFact
		want bool
	}{
		"yield only":      {facts.FnFact{Object: "F", MayYield: true, HasMayYield: true}, true},
		"pure only":       {facts.FnFact{Object: "F", HasMayYield: true}, true},
		"with nonnil":     {facts.FnFact{Object: "F", HasMayYield: true, ReturnsNonNil: true}, false},
		"with receiver":   {facts.FnFact{Object: "F", HasMayYield: true, ReceiverMutated: true}, false},
		"with param mask": {facts.FnFact{Object: "F", HasMayYield: true, ParamMutated: "1"}, false},
		"no yield fact":   {facts.FnFact{Object: "F", PureConst: true}, false},
	} {
		if got := c.fn.MayYieldOnly(); got != c.want {
			t.Errorf("%s: MayYieldOnly = %v, want %v", name, got, c.want)
		}
	}
}

func TestFnFlagNormalization(t *testing.T) {
	back, err := facts.Decode([]byte("logos-facts 1\nfingerprint 0001020304050607\nfn S MayYield=0,ParamMutated=00AB\n"))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if len(back.Fns) != 1 {
		t.Fatalf("got %d fns, want 1", len(back.Fns))
	}
	fn := back.Fns[0]
	if fn.ParamMutated != "ab" || !fn.HasMayYield || fn.MayYield {
		t.Fatalf("bad normalized flags: %+v", fn)
	}
	if out := string(facts.Encode(back.Fingerprint, back)); !strings.Contains(out, "fn S ParamMutated=ab,MayYield=0\n") {
		t.Fatalf("re-Encode not canonical: %q", out)
	}
}

func TestOddLengthParamMaskRoundTrips(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	in := &facts.Facts{Fns: []facts.FnFact{{Object: "F", ParamMutated: "1"}}}
	back, err := facts.Decode(facts.Encode(fp, in))
	if err != nil {
		t.Fatalf("Decode of encoded mask 1 failed: %v", err)
	}
	if len(back.Fns) != 1 || back.Fns[0].ParamMutated != "1" {
		t.Fatalf("got %+v, want one fn with ParamMutated 1", back.Fns)
	}
}

func TestImportVerifiesFingerprint(t *testing.T) {
	fpA := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	fpB := goobj.FingerprintType{8, 7, 6, 5, 4, 3, 2, 1}
	member := facts.Encode(fpA, nil)

	_, err := facts.Import("example.com/m/a", "export-bytes"+string(fpB[:]), member)
	if err == nil || err.Error() != "logos: stale facts for example.com/m/a (fingerprint mismatch)" {
		t.Fatalf("stale Import error = %v, want pinned mismatch message", err)
	}

	_, err = facts.Import("example.com/m/a", "tiny", member)
	if err == nil || !strings.Contains(err.Error(), "stale facts for example.com/m/a") {
		t.Fatalf("short-data Import error = %v, want stale facts message", err)
	}

	got, err := facts.Import("example.com/m/a", "export-bytes"+string(fpA[:]), member)
	if err != nil {
		t.Fatalf("fresh Import failed: %v", err)
	}
	if got.Fingerprint != fpA {
		t.Fatalf("imported fingerprint = %x, want %x", got.Fingerprint, fpA)
	}
	back, ok := facts.Lookup("example.com/m/a")
	if !ok || !reflect.DeepEqual(back, got) {
		t.Fatalf("Lookup = %+v, %v; want adopted facts", back, ok)
	}
	if _, ok := facts.Lookup("example.com/m/missing"); ok {
		t.Fatalf("Lookup of unknown path succeeded")
	}
}

func TestCurrentProvider(t *testing.T) {
	facts.SetCurrent(nil)
	if facts.Current() != nil {
		t.Fatalf("Current with no producer is not nil")
	}
	f := &facts.Facts{}
	facts.SetCurrent(f)
	if facts.Current() != f {
		t.Fatalf("Current did not return the installed facts")
	}
	facts.SetCurrent(nil)
}

func archiveMember(name string, data []byte) []byte {
	hdr := fmt.Sprintf("%-16s%-12d%-6d%-6d%-8o%-10d`\n", name, 0, 0, 0, 0o644, len(data))
	b := append([]byte(hdr), data...)
	if len(data)&1 != 0 {
		b = append(b, 0)
	}
	return b
}

func writeArchive(t *testing.T, data []byte) *os.File {
	t.Helper()
	path := filepath.Join(t.TempDir(), "a.a")
	if err := os.WriteFile(path, data, 0o666); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

// TestConformsRoundTrip covers the M4-S2 conforms records: each type's
// declared conformances encode canonically, decode back, and sort with
// the other records; vacuous records are skipped like every other kind.
func TestConformsRoundTrip(t *testing.T) {
	fp := goobj.FingerprintType{2}
	in := &facts.Facts{
		Fingerprint: fp,
		Props: []facts.PropFact{
			{Interface: "Creature", Prefix: "Logos_"},
		},
		Conforms: []facts.ConformsFact{
			{Type: "Orc", IfacePath: "example.com/l/p", Iface: "Creature"},
			{Type: "Monster", IfacePath: "example.com/l/p", Iface: "Creature"},
			{Type: "", IfacePath: "example.com/l/p", Iface: "Creature"},
			{Type: "Wisp", IfacePath: "", Iface: "Creature"},
			{Type: "Wisp", IfacePath: "example.com/l/p", Iface: ""},
		},
	}
	want := "logos-facts 1\n" +
		"fingerprint 0200000000000000\n" +
		"conforms Monster example.com/l/p Creature\n" +
		"conforms Orc example.com/l/p Creature\n" +
		"prop Creature Logos_\n"
	if got := string(facts.Encode(fp, in)); got != want {
		t.Fatalf("Encode mismatch:\n got %q\nwant %q", got, want)
	}
	back, err := facts.Decode([]byte(want))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	wantBack := &facts.Facts{
		Fingerprint: fp,
		Props: []facts.PropFact{
			{Interface: "Creature", Prefix: "Logos_"},
		},
		Conforms: []facts.ConformsFact{
			{Type: "Monster", IfacePath: "example.com/l/p", Iface: "Creature"},
			{Type: "Orc", IfacePath: "example.com/l/p", Iface: "Creature"},
		},
	}
	if !reflect.DeepEqual(back, wantBack) {
		t.Fatalf("round trip mismatch:\n got %+v\nwant %+v", back, wantBack)
	}
}

func TestFindMember(t *testing.T) {
	payload := facts.Encode(goobj.FingerprintType{9}, nil)
	var ar []byte
	ar = append(ar, []byte("!<arch>\n")...)
	ar = append(ar, archiveMember("__.PKGDEF", []byte("abc"))...)
	ar = append(ar, archiveMember(facts.MemberName, payload)...)
	ar = append(ar, archiveMember("_go_.o", []byte("01"))...)

	got, err := facts.FindMember(writeArchive(t, ar))
	if err != nil {
		t.Fatalf("FindMember failed: %v", err)
	}
	if !reflect.DeepEqual(got, payload) {
		t.Fatalf("FindMember = %q, want %q", got, payload)
	}

	preSidecar := append([]byte("!<arch>\n"), archiveMember("__.PKGDEF", []byte("abc"))...)
	preSidecar = append(preSidecar, archiveMember("_go_.o", []byte("01"))...)
	got, err = facts.FindMember(writeArchive(t, preSidecar))
	if err != nil || got != nil {
		t.Fatalf("FindMember(pre-sidecar) = %q, %v; want nil, nil", got, err)
	}

	got, err = facts.FindMember(writeArchive(t, []byte("hello")))
	if err != nil || got != nil {
		t.Fatalf("FindMember(non-archive) = %q, %v; want nil, nil", got, err)
	}

	hdr := fmt.Sprintf("%-16s%-12d%-6d%-6d%-8o%-10d`\n", facts.MemberName, 0, 0, 0, 0o644, 100)
	truncated := append([]byte("!<arch>\n"+hdr), 'x', 'y', 'z')
	if _, err := facts.FindMember(writeArchive(t, truncated)); err == nil {
		t.Fatalf("FindMember(truncated) succeeded, want error")
	}
}
