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
			{Object: "(*S).Get", Readonly: true},
			{Object: "R.M", Readonly: true},
		},
		Props: []facts.PropFact{
			{Interface: "I", Prefix: "Get"},
		},
		Comptimes: []facts.ComptimeFact{
			{Object: "New", Source: []byte("line1\nline2\n")},
		},
	}
	want := "logos-facts 3\n" +
		"fingerprint 0001020304050607\n" +
		"comptime New 12\n" +
		"line1\nline2\n" +
		"\n" +
		"const C K\n" +
		"fn (*S).Get Readonly\n" +
		"fn R.M Readonly\n" +
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
			{Object: "(*S).Get", Readonly: true},
			{Object: "R.M", Readonly: true},
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

func TestValidObjectTag(t *testing.T) {
	cases := []struct {
		object string
		want   bool
	}{
		{"Float32sFromArch@simd0", true},
		{"Foo@junk!", false},
		{"Foo@", false},
		{"Foo@1x", false},
	}
	for _, c := range cases {
		t.Run(c.object, func(t *testing.T) {
			if got := facts.ValidObject(c.object); got != c.want {
				t.Fatalf("ValidObject(%q) = %v, want %v", c.object, got, c.want)
			}
		})
	}
}

func TestTaggedObjectRoundTrip(t *testing.T) {
	fp := goobj.FingerprintType{0, 1, 2, 3, 4, 5, 6, 7}
	in := &facts.Facts{
		Fingerprint: fp,
		Fns:         []facts.FnFact{{Object: "Float32sFromArch@simd0", Readonly: true}},
	}
	back, err := facts.Decode(facts.Encode(fp, in))
	if err != nil {
		t.Fatalf("Decode(Encode) failed: %v", err)
	}
	if !reflect.DeepEqual(back, in) {
		t.Fatalf("round trip mismatch:\n got %+v\nwant %+v", back, in)
	}
	bad := "logos-facts 3\nfingerprint 0001020304050607\nfn Foo@junk! Readonly\n"
	if _, err := facts.Decode([]byte(bad)); err == nil || !strings.Contains(err.Error(), "bad object") {
		t.Fatalf("Decode accepted a malformed tag: %v", err)
	}
}

func TestEncodeNilHeaderOnly(t *testing.T) {
	fp := goobj.FingerprintType{7, 6, 5, 4, 3, 2, 1, 0}
	got := facts.Encode(fp, nil)
	if string(got) != "logos-facts 3\nfingerprint 0706050403020100\n" {
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
		Fns:         []facts.FnFact{{Object: "N"}, {Object: "", Readonly: true}},
		Props:       []facts.PropFact{{Interface: "I"}, {Prefix: "Get"}},
		Comptimes:   []facts.ComptimeFact{{Object: "New"}, {Source: []byte("x")}},
	}
	got := facts.Encode(fp, in)
	if string(got) != "logos-facts 3\nfingerprint 0100000000000000\n" {
		t.Fatalf("vacuous records not skipped: %q", got)
	}
	if _, err := facts.Decode(got); err != nil {
		t.Fatalf("Decode(skipped) failed: %v", err)
	}
}

func TestDecodeRejects(t *testing.T) {
	head := "logos-facts 3\nfingerprint 0001020304050607\n"
	cases := []struct{ name, input, line string }{
		{"empty", "", "line 1"},
		{"wrong version", "logos-facts 2\nfingerprint 0001020304050607\n", "line 1"},
		{"bad magic", "hello\n", "line 1"},
		{"missing fingerprint", "logos-facts 3\n", "line 2"},
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
		{"bad path indexed D", head + "opt H D2\n", "line 3"},
		{"bad path D with name", head + "opt H DName\n", "line 3"},
		{"bad path negative", head + "opt H R-1\n", "line 3"},
		{"bad flag unknown", head + "fn S Bogus\n", "line 3"},
		{"bad flag MayYield", head + "fn S MayYield=2\n", "line 3"},
		{"bad flag ParamMutated", head + "fn S ParamMutated=zz\n", "line 3"},
		{"bad flag ParamMutated empty", head + "fn S ParamMutated=\n", "line 3"},
		{"fn missing flags", head + "fn S\n", "line 3"},
		{"unsorted", head + "opt S.F R0\nfn S Readonly\n", "line 4"},
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

func TestDPathRoundTrip(t *testing.T) {
	fp := goobj.FingerprintType{3}
	in := &facts.Facts{
		Fingerprint: fp,
		Opts: []facts.PathFact{
			{Object: "(*S).Get", Path: "R0.D.FName"},
			{Object: "H", Path: "P0.D.D"},
			{Object: "H", Path: "P0.D.E"},
		},
		Consts: []facts.PathFact{
			{Object: "C", Path: "R0.D.K"},
		},
	}
	want := "logos-facts 3\n" +
		"fingerprint 0300000000000000\n" +
		"const C R0.D.K\n" +
		"opt (*S).Get R0.D.FName\n" +
		"opt H P0.D.D\n" +
		"opt H P0.D.E\n"
	if got := string(facts.Encode(fp, in)); got != want {
		t.Fatalf("Encode mismatch:\n got %q\nwant %q", got, want)
	}
	back, err := facts.Decode([]byte(want))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	wantBack := &facts.Facts{
		Fingerprint: fp,
		Opts: []facts.PathFact{
			{Object: "(*S).Get", Path: "R0.D.FName"},
			{Object: "H", Path: "P0.D.D"},
			{Object: "H", Path: "P0.D.E"},
		},
		Consts: []facts.PathFact{
			{Object: "C", Path: "R0.D.K"},
		},
	}
	if !reflect.DeepEqual(back, wantBack) {
		t.Fatalf("round trip mismatch:\n got %+v\nwant %+v", back, wantBack)
	}
}

func TestDPathQualifierVsContainer(t *testing.T) {
	in := "logos-facts 3\nfingerprint 0001020304050607\n" +
		"opt H R0.D\n" +
		"opt H R0.E\n" +
		"opt S.F R0.FD\n"
	back, err := facts.Decode([]byte(in))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	want := []facts.PathFact{
		{Object: "H", Path: "R0.D"},
		{Object: "H", Path: "R0.E"},
		{Object: "S.F", Path: "R0.FD"},
	}
	if !reflect.DeepEqual(back.Opts, want) {
		t.Fatalf("Decode paths:\n got %+v\nwant %+v", back.Opts, want)
	}
	if out := string(facts.Encode(back.Fingerprint, back)); out != in {
		t.Fatalf("re-Encode not canonical:\n got %q\nwant %q", out, in)
	}
}

func TestDecodeLineNumbersAfterComptimeBody(t *testing.T) {
	head := "logos-facts 3\nfingerprint 0001020304050607\n"
	for _, pad := range []int{0, 25} {
		body := "ab\n" + strings.Repeat("x\n", pad) + "cd"
		in := head + "comptime New " + itoa(len(body)) + "\n" + body + "\nopt H Q1\n"
		want := 3 + strings.Count(body, "\n") + 2
		_, err := facts.Decode([]byte(in))
		if err == nil || !strings.Contains(err.Error(), "line "+itoa(want)) {
			t.Errorf("pad %d: Decode error = %v, want error naming line %d", pad, err, want)
		}
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
	want := "logos-facts 3\nfingerprint 0102030405060708\nfn F MayYield=1\nlang go\n"
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

func TestGoPackageRecordSingular(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	enc := string(facts.Encode(fp, &facts.Facts{GoPackage: true}))
	n := 0
	for line := range strings.SplitSeq(enc, "\n") {
		if line == "lang go" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("Encode holds %d lang go records, want 1:\n%s", n, enc)
	}
	dup := "logos-facts 3\nfingerprint 0102030405060708\nlang go\nlang go\n"
	if _, err := facts.Decode([]byte(dup)); err == nil {
		t.Fatalf("Decode accepted two lang go records, want rejection")
	}
}

func TestMayYieldOnly(t *testing.T) {
	for name, c := range map[string]struct {
		fn   facts.FnFact
		want bool
	}{
		"yield only":      {facts.FnFact{Object: "F", MayYield: true, HasMayYield: true}, true},
		"pure only":       {facts.FnFact{Object: "F", HasMayYield: true}, true},
		"with receiver":   {facts.FnFact{Object: "F", HasMayYield: true, ReceiverMutated: true}, false},
		"with param mask": {facts.FnFact{Object: "F", HasMayYield: true, ParamMutated: "1"}, false},
	} {
		if got := c.fn.MayYieldOnly(); got != c.want {
			t.Errorf("%s: MayYieldOnly = %v, want %v", name, got, c.want)
		}
	}
}

func TestFnFlagNormalization(t *testing.T) {
	back, err := facts.Decode([]byte("logos-facts 3\nfingerprint 0001020304050607\nfn S MayYield=0,ParamMutated=00AB\n"))
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
	want := "logos-facts 3\n" +
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

func TestAssertEffectsWire(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	in := &facts.Facts{Fns: []facts.FnFact{{
		Object:    "F",
		Readonly:  true,
		HasAssert: true,
		AssertTP:  "0",
		AssertRTP: "800000000000000000",
	}}}
	out := string(facts.Encode(fp, in))
	if !strings.Contains(out, "fn F Readonly,AssertTP=0,AssertRTP=800000000000000000\n") {
		t.Fatalf("assert flags not canonical (after Readonly, arbitrary width): %q", out)
	}
	back, err := facts.Decode([]byte(out))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	fn := back.Fns[0]
	if !fn.HasAssert || fn.AssertTP != "0" || fn.AssertRTP != "800000000000000000" {
		t.Fatalf("assert fields lost: %+v", fn)
	}
	if !fn.MayYieldOnly() && fn.HasMayYield {
		t.Fatal("unexpected MayYield")
	}
	ay := facts.FnFact{Object: "G", MayYield: true, HasMayYield: true, HasAssert: true, AssertTP: "0", AssertRTP: "0"}
	if !ay.MayYieldOnly() {
		t.Fatal("assert-only record must stay MayYieldOnly (mutation unknown)")
	}
}

func TestAssertKnownZeroVsUnknown(t *testing.T) {
	knownZero := "logos-facts 3\nfingerprint 0001020304050607\nfn F AssertTP=0,AssertRTP=0\n"
	back, err := facts.Decode([]byte(knownZero))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if len(back.Fns) != 1 || !back.Fns[0].HasAssert || back.Fns[0].AssertTP != "0" || back.Fns[0].AssertRTP != "0" {
		t.Fatalf("proven zero must survive as Known: %+v", back.Fns)
	}
	legacy := "logos-facts 3\nfingerprint 0001020304050607\nfn F Readonly\n"
	back, err = facts.Decode([]byte(legacy))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if len(back.Fns) != 1 || back.Fns[0].HasAssert {
		t.Fatalf("record without assert flags must decode unknown: %+v", back.Fns)
	}
}

func TestFactsVersion3Cutover(t *testing.T) {
	for _, v := range []string{"logos-facts 1", "logos-facts 2", "logos-facts 4"} {
		_, err := facts.Decode([]byte(v + "\nfingerprint 0001020304050607\n"))
		if err == nil || !strings.Contains(err.Error(), "unsupported version") {
			t.Errorf("Decode(%q) err = %v, want unsupported version", v, err)
		}
	}
	fp := goobj.FingerprintType{7, 6, 5, 4, 3, 2, 1, 0}
	in := &facts.Facts{
		Fingerprint: fp,
		Fns:         []facts.FnFact{{Object: "F", Readonly: true}},
	}
	enc := facts.Encode(fp, in)
	if !strings.HasPrefix(string(enc), "logos-facts 3\n") {
		t.Fatalf("Encode header = %q, want logos-facts 3", enc)
	}
	back, err := facts.Decode(enc)
	if err != nil {
		t.Fatalf("Decode(Encode v3) failed: %v", err)
	}
	if len(back.Fns) != 1 || back.Fns[0].Object != "F" || !back.Fns[0].Readonly {
		t.Fatalf("v3 round trip lost the fn fact: %+v", back.Fns)
	}
}

func TestAssertPairCompleteness(t *testing.T) {
	head := "logos-facts 3\nfingerprint 0001020304050607\n"
	cases := []struct{ name, line string }{
		{"tp only", head + "fn F AssertTP=0\n"},
		{"rtp only", head + "fn F AssertRTP=1\n"},
		{"tp only nonzero", head + "fn F AssertTP=1\n"},
	}
	for _, c := range cases {
		if _, err := facts.Decode([]byte(c.line)); err == nil || !strings.Contains(err.Error(), "incomplete assertion-effect pair") {
			t.Errorf("%s: Decode err = %v, want incomplete-pair rejection", c.name, err)
		}
	}
	for _, ok := range []string{
		head + "fn F AssertTP=0,AssertRTP=0\n",
		head + "fn F AssertTP=1,AssertRTP=0\n",
		head + "fn F Readonly\n",
	} {
		if _, err := facts.Decode([]byte(ok)); err != nil {
			t.Errorf("valid member rejected: %v (%q)", err, ok)
		}
	}
}

func TestHighestAssertBit(t *testing.T) {
	for _, c := range []struct {
		in   string
		want int
		ok   bool
	}{
		{"0", -1, true},
		{"00", -1, true},
		{"1", 0, true},
		{"2", 1, true},
		{"3", 1, true},
		{"ab", 7, true},
		{"800000000000000000", 71, true},
		{"", 0, false},
		{"xyz", 0, false},
		{"10Z", 0, false},
		{"0010", 4, true},
	} {
		got, ok := facts.HighestAssertBit(c.in)
		if got != c.want || ok != c.ok {
			t.Errorf("HighestAssertBit(%q) = %d, %v; want %d, %v", c.in, got, ok, c.want, c.ok)
		}
	}
}

func TestLocksFactRoundTrip(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	in := &facts.Facts{
		Fingerprint: fp,
		Locks: []facts.LockFact{
			{Object: "(*Tank).Fill", Effect: "requires", Path: "t.mu"},
			{Object: "(*Tank).Seal", Effect: "acquires", Path: "t.mu"},
			{Object: "(*Tank).Open", Effect: "releases", Path: "t.mu"},
		},
	}
	want := "logos-facts 3\n" +
		"fingerprint 0102030405060708\n" +
		"locks (*Tank).Fill requires t.mu\n" +
		"locks (*Tank).Open releases t.mu\n" +
		"locks (*Tank).Seal acquires t.mu\n"
	got := facts.Encode(fp, in)
	if string(got) != want {
		t.Fatalf("Encode mismatch:\n got %q\nwant %q", got, want)
	}
	back, err := facts.Decode(got)
	if err != nil {
		t.Fatalf("Decode(Encode) failed: %v", err)
	}
	wantBack := []facts.LockFact{
		{Object: "(*Tank).Fill", Effect: "requires", Path: "t.mu"},
		{Object: "(*Tank).Open", Effect: "releases", Path: "t.mu"},
		{Object: "(*Tank).Seal", Effect: "acquires", Path: "t.mu"},
	}
	if !reflect.DeepEqual(back.Locks, wantBack) {
		t.Errorf("round trip = %+v, want %+v", back.Locks, wantBack)
	}
	for _, bad := range []string{
		"locks F wrong mu\n",
		"locks F requires\n",
		"locks F requires t..mu\n",
		"locks no!such requires mu\n",
		"locks (*Tank).Fill requires t.mu\nallocfree Sum\n",
	} {
		full := "logos-facts 3\nfingerprint 0102030405060708\n" + bad
		if _, err := facts.Decode([]byte(full)); err == nil {
			t.Errorf("Decode accepted %q", bad)
		}
	}
}

func TestAllocFreeFactRoundTrip(t *testing.T) {
	fp := goobj.FingerprintType{9, 8, 7, 6, 5, 4, 3, 2}
	in := &facts.Facts{
		Fingerprint: fp,
		AllocFrees:  []facts.AllocFreeFact{{Object: "Sum"}, {Object: "(*Tank).Load"}},
	}
	want := "logos-facts 3\n" +
		"fingerprint 0908070605040302\n" +
		"allocfree (*Tank).Load\n" +
		"allocfree Sum\n"
	got := facts.Encode(fp, in)
	if string(got) != want {
		t.Fatalf("Encode mismatch:\n got %q\nwant %q", got, want)
	}
	back, err := facts.Decode(got)
	if err != nil {
		t.Fatalf("Decode(Encode) failed: %v", err)
	}
	wantBack := []facts.AllocFreeFact{{Object: "(*Tank).Load"}, {Object: "Sum"}}
	if !reflect.DeepEqual(back.AllocFrees, wantBack) {
		t.Errorf("round trip = %+v, want %+v", back.AllocFrees, wantBack)
	}
	full := "logos-facts 3\nfingerprint 0908070605040302\nallocfree no!such\n"
	if _, err := facts.Decode([]byte(full)); err == nil {
		t.Error("Decode accepted a bad allocfree object")
	}
}

func TestBareUnionFactRoundTrip(t *testing.T) {
	fp := goobj.FingerprintType{9, 8, 7, 6, 5, 4, 3, 2}
	in := &facts.Facts{
		Fingerprint: fp,
		BareUnions:  []string{"Scalar", "Choice"},
	}
	want := "logos-facts 3\n" +
		"fingerprint 0908070605040302\n" +
		"unionbare Choice\n" +
		"unionbare Scalar\n"
	got := facts.Encode(fp, in)
	if string(got) != want {
		t.Fatalf("Encode mismatch:\n got %q\nwant %q", got, want)
	}
	back, err := facts.Decode(got)
	if err != nil {
		t.Fatalf("Decode(Encode) failed: %v", err)
	}
	if wantBack := []string{"Choice", "Scalar"}; !reflect.DeepEqual(back.BareUnions, wantBack) {
		t.Errorf("round trip = %v, want %v", back.BareUnions, wantBack)
	}
	for _, bad := range []string{"unionbare no!such\n", "unionbare A B\n", "unionbare\n"} {
		full := "logos-facts 3\nfingerprint 0908070605040302\n" + bad
		if _, err := facts.Decode([]byte(full)); err == nil {
			t.Errorf("Decode accepted %q", bad)
		}
	}
}

func TestJSONFactsRoundTrip(t *testing.T) {
	fp := goobj.FingerprintType{9, 8, 7, 6, 5, 4, 3, 2}
	in := &facts.Facts{
		JsonMetas: []facts.JsonMetaFact{
			{Type: "Square", HasDecoder: true, Eligible: true},
			{Type: "Circle", HasDecoder: true, Eligible: true},
			{Type: "Custom", HasDecoder: true},
		},
		JsonKeys: []facts.JsonKeyFact{
			{Type: "Square", Index: 1, Key: "s"},
			{Type: "Circle", Index: 0, Key: "r"},
			{Type: "Circle", Index: 1, Key: "extra"},
		},
		UnionMembers: []facts.UnionMemberFact{
			{Type: "Shape", Index: 1, Member: "example.com/m/a.Square"},
			{Type: "Shape", Index: 0, Member: "example.com/m/a.Circle"},
		},
	}
	got := string(facts.Encode(fp, in))
	wantOrder := []string{
		"jsonkey Circle 0000 1\nr\n",
		"jsonkey Circle 0001 5\nextra\n",
		"jsonkey Square 0001 1\ns\n",
		"jsonmeta Circle decoder,eligible\n",
		"jsonmeta Custom decoder\n",
		"jsonmeta Square decoder,eligible\n",
		"unionmember Shape 0000 example.com/m/a.Circle\n",
		"unionmember Shape 0001 example.com/m/a.Square\n",
	}
	prev := -1
	for _, want := range wantOrder {
		idx := strings.Index(got, want)
		if idx < 0 {
			t.Fatalf("Encode missing %q in:\n%s", want, got)
		}
		if idx <= prev {
			t.Fatalf("Encode not sorted: %q after offset %d", want, prev)
		}
		prev = idx
	}
	back, err := facts.Decode([]byte(got))
	if err != nil {
		t.Fatalf("Decode(Encode) failed: %v", err)
	}
	if len(back.JsonMetas) != 3 || len(back.JsonKeys) != 3 || len(back.UnionMembers) != 2 {
		t.Fatalf("round trip counts = %d/%d/%d, want 3/3/2", len(back.JsonMetas), len(back.JsonKeys), len(back.UnionMembers))
	}
	var circleKeys []string
	for _, k := range back.JsonKeys {
		if k.Type == "Circle" {
			circleKeys = append(circleKeys, k.Key)
		}
	}
	if len(circleKeys) != 2 || circleKeys[0] != "r" || circleKeys[1] != "extra" {
		t.Fatalf("Circle keys = %q, want [r extra]", circleKeys)
	}
	goPkg := &facts.Facts{GoPackage: true, JsonMetas: []facts.JsonMetaFact{{Type: "X", HasDecoder: true, Eligible: true}}, JsonKeys: []facts.JsonKeyFact{{Type: "X", Index: 0, Key: "a"}}, UnionMembers: []facts.UnionMemberFact{{Type: "U", Index: 0, Member: "example.com/m.X"}}}
	out := string(facts.Encode(fp, goPkg))
	if strings.Contains(out, "jsonmeta ") || strings.Contains(out, "jsonkey ") || strings.Contains(out, "unionmember ") {
		t.Fatalf("Go package carries JSON facts:\n%s", out)
	}
	if !strings.Contains(out, "lang go\n") {
		t.Fatalf("Go package missing lang go:\n%s", out)
	}
}
