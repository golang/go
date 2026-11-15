// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts_test

import (
	"reflect"
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/internal/goobj"
)

func dupHead() string {
	return "logos-facts 3\nfingerprint 0001020304050607\n"
}

func dupBase() string {
	return dupHead() +
		"allocfree F\n" +
		"comptime C 4\nabcd\n" +
		"conforms S example.com/x I\n" +
		"const K P0\n" +
		"constcomp M\n" +
		"constfn N 4\nwxyz\n" +
		"constmap M 36\nbearing 0\norder 0\nentry 1a 1 1 1\nckv\n" +
		"constmap M 36 .a[0]\nbearing 0\norder 0\nentry 1a 1 1 1\nckv\n" +
		"default T Fld 3\nabc\n" +
		"fn F MayYield=1\n" +
		"jsonkey J 0000 3\nkey\n" +
		"jsonmeta J decoder\n" +
		"locks F requires P0\n" +
		"opt H P1\n" +
		"prop I Get\n" +
		"readonly R P0\n" +
		"unionmember U 0000 example.com/x.T\n"
}

func TestDecodeRoundTripEveryKind(t *testing.T) {
	fp := goobj.FingerprintType{0, 1, 2, 3, 4, 5, 6, 7}
	in := &facts.Facts{
		Fingerprint: fp,
		Opts:        []facts.PathFact{{Object: "H", Path: "P1"}},
		Consts:      []facts.PathFact{{Object: "K", Path: "P0"}},
		Fns: []facts.FnFact{
			{Object: "(*S).Get", Readonly: true},
			{Object: "A.Id", HasAssert: true, AssertTP: "0", AssertRTP: "1"},
			{Object: "F", AlwaysNilError: true},
			{Object: "G", MayYield: true, HasMayYield: true},
			{Object: "S.Set", ReceiverMutated: true, ParamMutated: "1f", MayYield: true, HasMayYield: true},
		},
		Readonlys:  []facts.ReadonlyFact{{Object: "R", Path: "P0"}},
		Props:      []facts.PropFact{{Interface: "I", Prefix: "Get"}},
		Conforms:   []facts.ConformsFact{{Type: "S", IfacePath: "example.com/x", Iface: "I"}},
		Comptimes:  []facts.ComptimeFact{{Object: "C", Source: []byte("abcd")}},
		ConstFuncs: []facts.ConstFuncFact{{Object: "N", Source: []byte("wxyz")}},
		ConstMaps: []facts.ConstMapFact{
			{
				Object: "M", Order: []int{0}, Hashes: []uint64{0x1a},
				Canons: []string{"c"}, Keys: []string{"k"}, Vals: []string{"v"},
			},
			{
				Object: "M", Path: ".a[0]", Order: []int{0}, Hashes: []uint64{0x1a},
				Canons: []string{"c"}, Keys: []string{"k"}, Vals: []string{"v"},
			},
		},
		ConstComps: []string{"M"},
		Defaults:   []facts.DefaultFact{{Type: "T", Field: "Fld", Text: "abc"}},
		Locks:      []facts.LockFact{{Object: "F", Effect: "requires", Path: "P0"}},
		AllocFrees: []facts.AllocFreeFact{{Object: "F"}},
		JsonMetas:  []facts.JsonMetaFact{{Type: "J", HasDecoder: true}},
		JsonKeys:   []facts.JsonKeyFact{{Type: "J", Index: 0, Key: "key"}},
		UnionMembers: []facts.UnionMemberFact{
			{Type: "U", Index: 0, Member: "example.com/x.T"},
		},
	}
	back, err := facts.Decode(facts.Encode(fp, in))
	if err != nil {
		t.Fatalf("Decode(Encode(x)) failed: %v", err)
	}
	if !reflect.DeepEqual(back, in) {
		t.Fatalf("Decode(Encode(x)) != x:\n got %+v\nwant %+v", back, in)
	}
}

func TestDecodeRejectsDuplicateKey(t *testing.T) {
	cases := []struct{ name, input string }{
		{"fn conflicting flags", dupHead() + "fn F AlwaysNilError\nfn F MayYield=1\n"},
		{"fn readonly vs yield", dupHead() + "fn F MayYield=1\nfn F Readonly\n"},
		{"prop conflicting prefix", dupHead() + "prop I Get\nprop I Put\n"},
		{"jsonmeta conflicting flags", dupHead() + "jsonmeta J decoder\njsonmeta J eligible\n"},
		{"jsonkey conflicting body", dupHead() + "jsonkey J 0000 3\nkey\njsonkey J 0000 4\nkey2\n"},
		{"unionmember conflicting member", dupHead() + "unionmember U 0000 example.com/x.S\nunionmember U 0000 example.com/x.T\n"},
		{"comptime conflicting body", dupHead() + "comptime C 3\nabc\ncomptime C 4\nabcd\n"},
		{"constfn conflicting body", dupHead() + "constfn N 3\nabc\nconstfn N 4\nabcd\n"},
		{"default conflicting text", dupHead() + "default T Fld 2\nab\ndefault T Fld 3\nabc\n"},
		{"constmap conflicting body", dupHead() + "constmap M 16\nbearing 0\norder\n\nconstmap M 36\nbearing 0\norder 0\nentry 1a 1 1 1\nckv\n"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := facts.Decode([]byte(c.input))
			if err == nil {
				t.Fatalf("Decode(forged %s) = %+v, want duplicate error", c.name, got)
			}
			if !strings.Contains(err.Error(), "uplicate") {
				t.Fatalf("Decode(forged %s) error %q does not name duplication", c.name, err)
			}
		})
	}
}

func TestDecodeRejectsExactDuplicate(t *testing.T) {
	cases := []struct{ name, input string }{
		{"fn same payload", dupHead() + "fn F MayYield=1\nfn F MayYield=1\n"},
		{"allocfree same object", dupHead() + "allocfree F\nallocfree F\n"},
		{"lang go twice", dupHead() + "lang go\nlang go\n"},
		{"opt same object same path", dupHead() + "opt H P1\nopt H P1\n"},
		{"const same object same path", dupHead() + "const K P0\nconst K P0\n"},
		{"readonly same object same path", dupHead() + "readonly R P0\nreadonly R P0\n"},
		{"locks same triple", dupHead() + "locks F requires P0\nlocks F requires P0\n"},
		{"conforms same triple", dupHead() + "conforms S example.com/x I\nconforms S example.com/x I\n"},
		{"default same field same text", dupHead() + "default T Fld 3\nabc\ndefault T Fld 3\nabc\n"},
		{"constmap same object same body", dupHead() + "constmap M 16\nbearing 0\norder\n\nconstmap M 16\nbearing 0\norder\n\n"},
		{"constcomp same object", dupHead() + "constcomp M\nconstcomp M\n"},
		{"constmap same object same path", dupHead() + "constmap M 16 .a\nbearing 0\norder\n\nconstmap M 16 .a\nbearing 0\norder\n\n"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got, err := facts.Decode([]byte(c.input)); err == nil {
				t.Fatalf("Decode(dup %s) = %+v, want error", c.name, got)
			}
		})
	}
}

func TestDecodeAcceptsDistinctKeys(t *testing.T) {
	cases := []struct {
		name  string
		input string
		check func(*facts.Facts) bool
	}{
		{"opt same object distinct paths", dupHead() + "opt H P1\nopt H R0\n",
			func(f *facts.Facts) bool { return len(f.Opts) == 2 }},
		{"const same object distinct paths", dupHead() + "const K P0\nconst K R0\n",
			func(f *facts.Facts) bool { return len(f.Consts) == 2 }},
		{"readonly same object distinct paths", dupHead() + "readonly R P0\nreadonly R R0\n",
			func(f *facts.Facts) bool { return len(f.Readonlys) == 2 }},
		{"locks same object distinct effects", dupHead() + "locks F acquires P0\nlocks F requires P0\n",
			func(f *facts.Facts) bool { return len(f.Locks) == 2 }},
		{"conforms same type distinct ifaces", dupHead() + "conforms S example.com/x I\nconforms S example.com/x J\n",
			func(f *facts.Facts) bool { return len(f.Conforms) == 2 }},
		{"default same type distinct fields", dupHead() + "default T A 1\na\ndefault T B 1\nb\n",
			func(f *facts.Facts) bool { return len(f.Defaults) == 2 }},
		{"jsonkey same type distinct index", dupHead() + "jsonkey J 0000 3\nkey\njsonkey J 0001 3\nkey\n",
			func(f *facts.Facts) bool { return len(f.JsonKeys) == 2 }},
		{"unionmember same type distinct index", dupHead() + "unionmember U 0000 example.com/x.S\nunionmember U 0001 example.com/x.T\n",
			func(f *facts.Facts) bool { return len(f.UnionMembers) == 2 }},
		{"constmap same object distinct paths", dupHead() + "constmap M 16 .a\nbearing 0\norder\n\nconstmap M 16 .b\nbearing 0\norder\n\n",
			func(f *facts.Facts) bool {
				return len(f.ConstMaps) == 2 && f.ConstMaps[0].Path == ".a" && f.ConstMaps[1].Path == ".b"
			}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := facts.Decode([]byte(c.input))
			if err != nil {
				t.Fatalf("Decode(distinct %s) failed: %v", c.name, err)
			}
			if !c.check(got) {
				t.Fatalf("Decode(distinct %s) dropped a record: %+v", c.name, got)
			}
		})
	}
}

func TestDecodeTruncationNoPanic(t *testing.T) {
	full := []byte(dupBase())
	if _, err := facts.Decode(full); err != nil {
		t.Fatalf("Decode(valid base) failed: %v", err)
	}
	for off := range len(full) {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("Decode(prefix %d) panicked: %v", off, r)
				}
			}()
			got, err := facts.Decode(full[:off])
			if err != nil {
				return
			}
			re := string(facts.Encode(got.Fingerprint, got))
			if re != string(full[:off]) && re != string(full[:off])+"\n" {
				t.Fatalf("Decode(prefix %d) succeeded but re-encode differs", off)
			}
		}()
	}
}

func TestDecodeRejectsSwappedAdjacent(t *testing.T) {
	blocks := []string{
		"allocfree F\n",
		"comptime C 4\nabcd\n",
		"conforms S example.com/x I\n",
		"const K P0\n",
		"constcomp M\n",
		"constfn N 4\nwxyz\n",
		"constmap M 36\nbearing 0\norder 0\nentry 1a 1 1 1\nckv\n",
		"constmap M 36 .a[0]\nbearing 0\norder 0\nentry 1a 1 1 1\nckv\n",
		"default T Fld 3\nabc\n",
		"fn F MayYield=1\n",
		"jsonkey J 0000 3\nkey\n",
		"jsonmeta J decoder\n",
		"locks F requires P0\n",
		"opt H P1\n",
		"prop I Get\n",
		"readonly R P0\n",
		"unionmember U 0000 example.com/x.T\n",
	}
	for i := range len(blocks) - 1 {
		swapped := slices.Clone(blocks)
		swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
		input := dupHead() + strings.Join(swapped, "")
		if _, err := facts.Decode([]byte(input)); err == nil {
			t.Fatalf("Decode(swapped %d,%d) succeeded, want unsorted error", i, i+1)
		}
	}
}

func TestDecodeRejectsMalformedCompositeRecords(t *testing.T) {
	cases := []struct{ name, input string }{
		{"constcomp bad object", dupHead() + "constcomp 9x\n"},
		{"constcomp extra field", dupHead() + "constcomp M N\n"},
	}
	for _, p := range []string{"a", ".", ".a.", ".a[", "[x]", ".1a", ".a[01]", ".a[0", "..a", ".a]0[", ".a[-1]"} {
		cases = append(cases, struct{ name, input string }{"constmap path " + p, dupHead() + "constmap M 16 " + p + "\nbearing 0\norder\n\n"})
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got, err := facts.Decode([]byte(c.input)); err == nil {
				t.Fatalf("Decode(%s) = %+v, want error", c.name, got)
			}
		})
	}
}

func TestDecodeRejectsDuplicateMapKey(t *testing.T) {
	fp := goobj.FingerprintType{0, 1, 2, 3, 4, 5, 6, 7}
	in := &facts.Facts{Fingerprint: fp, ConstMaps: []facts.ConstMapFact{{
		Object: "M", Order: []int{0, 1}, Hashes: []uint64{1, 1},
		Canons: []string{"c", "c"}, Keys: []string{"k", "k"}, Vals: []string{"v", "w"},
	}}}
	if got, err := facts.Decode(facts.Encode(fp, in)); err == nil {
		t.Fatalf("Decode(map with a repeated key) = %+v, want error", got)
	}
}

func TestDecodeRejectsMalformedShapes(t *testing.T) {
	cases := []struct{ name, input string }{
		{"unknown kind", dupHead() + "union U A|B\n"},
		{"oversized count", dupHead() + "comptime F 99999999999999999999\n"},
		{"huge count", dupHead() + "comptime F 1000000000000\n"},
		{"version skew", "logos-facts 2\nfingerprint 0001020304050607\n"},
		{"version skew high", "logos-facts 99\nfingerprint 0001020304050607\n"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := facts.Decode([]byte(c.input)); err == nil {
				t.Fatalf("Decode(%s) succeeded, want error", c.name)
			}
		})
	}
}

func TestImportRejectsForgedDuplicate(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	forged := "logos-facts 3\n" +
		"fingerprint 0102030405060708\n" +
		"fn F AlwaysNilError\n" +
		"fn F MayYield=1\n"
	var ar []byte
	ar = append(ar, []byte("!<arch>\n")...)
	ar = append(ar, archiveMember("__.PKGDEF", []byte("abc"))...)
	ar = append(ar, archiveMember(facts.MemberName, []byte(forged))...)
	got, err := facts.FindMember(writeArchive(t, ar))
	if err != nil {
		t.Fatalf("FindMember failed: %v", err)
	}
	if _, err := facts.Import("example.com/m/victim", "export-bytes"+string(fp[:]), got); err == nil {
		t.Fatal("Import(forged duplicate) succeeded, want build failure")
	} else if !strings.Contains(err.Error(), "uplicate") {
		t.Fatalf("Import(forged duplicate) error %q does not name duplication", err)
	}
}

func TestImportRejectsForgedCompositePath(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	forged := "logos-facts 3\n" +
		"fingerprint 0102030405060708\n" +
		"constcomp M\n" +
		"constmap M 16 .a[01]\nbearing 0\norder\n\n"
	var ar []byte
	ar = append(ar, []byte("!<arch>\n")...)
	ar = append(ar, archiveMember("__.PKGDEF", []byte("abc"))...)
	ar = append(ar, archiveMember(facts.MemberName, []byte(forged))...)
	got, err := facts.FindMember(writeArchive(t, ar))
	if err != nil {
		t.Fatalf("FindMember failed: %v", err)
	}
	if _, err := facts.Import("example.com/m/victim", "export-bytes"+string(fp[:]), got); err == nil {
		t.Fatal("Import(forged composite path) succeeded, want build failure")
	} else if !strings.Contains(err.Error(), "bad facts") {
		t.Fatalf("Import(forged composite path) error %q does not name the bad facts", err)
	}
}

func FuzzDecode(f *testing.F) {
	seeds := []string{
		dupBase(),
		dupHead() + "fn F AlwaysNilError\nfn F MayYield=1\n",
		dupHead() + "union U A|B\n",
		"",
		"logos-facts 3\n",
		dupHead() + "comptime C 4\nabcd",
	}
	for _, s := range seeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		got, err := facts.Decode(data)
		if err != nil {
			return
		}
		re, err := facts.Decode(facts.Encode(got.Fingerprint, got))
		if err != nil {
			t.Fatalf("Decode(Encode(Decode())) failed: %v", err)
		}
		if string(facts.Encode(re.Fingerprint, re)) != string(facts.Encode(got.Fingerprint, got)) {
			t.Fatalf("Decode(Encode(x)) is unstable for %+v", got)
		}
	})
}
