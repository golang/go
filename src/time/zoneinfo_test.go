// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"errors"
	"fmt"
	"internal/testenv"
	"os"
	"reflect"
	"testing"
	"time"
)

func init() {
	if time.ZoneinfoForTesting() != nil {
		panic(fmt.Errorf("zoneinfo initialized before first LoadLocation"))
	}
}

func TestEnvVarUsage(t *testing.T) {
	time.ResetZoneinfoForTesting()

	const testZoneinfo = "foo.zip"
	const env = "ZONEINFO"

	t.Setenv(env, testZoneinfo)

	// Result isn't important, we're testing the side effect of this command
	time.LoadLocation("Asia/Jerusalem")
	defer time.ResetZoneinfoForTesting()

	if zoneinfo := time.ZoneinfoForTesting(); testZoneinfo != *zoneinfo {
		t.Errorf("zoneinfo does not match env variable: got %q want %q", *zoneinfo, testZoneinfo)
	}
}

func TestBadLocationErrMsg(t *testing.T) {
	time.ResetZoneinfoForTesting()
	loc := "Asia/SomethingNotExist"
	want := errors.New("unknown time zone " + loc)
	_, err := time.LoadLocation(loc)
	if err.Error() != want.Error() {
		t.Errorf("LoadLocation(%q) error = %v; want %v", loc, err, want)
	}
}

func TestLoadLocationValidatesNames(t *testing.T) {
	time.ResetZoneinfoForTesting()
	const env = "ZONEINFO"
	t.Setenv(env, "")

	bad := []string{
		"/usr/foo/Foo",
		"\\UNC\foo",
		"..",
		"a..",
	}
	for _, v := range bad {
		_, err := time.LoadLocation(v)
		if !errors.Is(err, time.ErrLocation) {
			t.Errorf("LoadLocation(%q) error = %v; want ErrLocation", v, err)
		}
	}
}

func TestVersion3(t *testing.T) {
	undo := time.DisablePlatformSources()
	defer undo()
	_, err := time.LoadLocation("Asia/Jerusalem")
	if err != nil {
		t.Fatal(err)
	}
}

// Test that we get the correct results for times before the first
// transition time. To do this we explicitly check early dates in a
// couple of specific timezones.
func TestFirstZone(t *testing.T) {
	undo := time.DisablePlatformSources()
	defer undo()

	const format = "Mon, 02 Jan 2006 15:04:05 -0700 (MST)"
	var tests = []struct {
		zone  string
		unix  int64
		want1 string
		want2 string
	}{
		{
			"PST8PDT",
			-1633269601,
			"Sun, 31 Mar 1918 01:59:59 -0800 (PST)",
			"Sun, 31 Mar 1918 03:00:00 -0700 (PDT)",
		},
		{
			"Pacific/Fakaofo",
			1325242799,
			"Thu, 29 Dec 2011 23:59:59 -1100 (-11)",
			"Sat, 31 Dec 2011 00:00:00 +1300 (+13)",
		},
	}

	for _, test := range tests {
		z, err := time.LoadLocation(test.zone)
		if err != nil {
			t.Fatal(err)
		}
		s := time.Unix(test.unix, 0).In(z).Format(format)
		if s != test.want1 {
			t.Errorf("for %s %d got %q want %q", test.zone, test.unix, s, test.want1)
		}
		s = time.Unix(test.unix+1, 0).In(z).Format(format)
		if s != test.want2 {
			t.Errorf("for %s %d got %q want %q", test.zone, test.unix, s, test.want2)
		}
	}
}

func TestLocationNames(t *testing.T) {
	if time.Local.String() != "Local" {
		t.Errorf(`invalid Local location name: got %q want "Local"`, time.Local)
	}
	if time.UTC.String() != "UTC" {
		t.Errorf(`invalid UTC location name: got %q want "UTC"`, time.UTC)
	}
}

func TestLoadLocationFromTZData(t *testing.T) {
	undo := time.DisablePlatformSources()
	defer undo()

	const locationName = "Asia/Jerusalem"
	reference, err := time.LoadLocation(locationName)
	if err != nil {
		t.Fatal(err)
	}

	gorootSource, ok := time.GorootZoneSource(testenv.GOROOT(t))
	if !ok {
		t.Fatal("Failed to locate tzinfo source in GOROOT.")
	}
	tzinfo, err := time.LoadTzinfo(locationName, gorootSource)
	if err != nil {
		t.Fatal(err)
	}
	sample, err := time.LoadLocationFromTZData(locationName, tzinfo)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(reference, sample) {
		t.Errorf("return values of LoadLocationFromTZData and LoadLocation don't match")
	}
}

// Issue 30099.
func TestEarlyLocation(t *testing.T) {
	undo := time.DisablePlatformSources()
	defer undo()

	const locName = "America/New_York"
	loc, err := time.LoadLocation(locName)
	if err != nil {
		t.Fatal(err)
	}

	d := time.Date(1900, time.January, 1, 0, 0, 0, 0, loc)
	tzName, tzOffset := d.Zone()
	if want := "EST"; tzName != want {
		t.Errorf("Zone name == %s, want %s", tzName, want)
	}
	if want := -18000; tzOffset != want {
		t.Errorf("Zone offset == %d, want %d", tzOffset, want)
	}
}

func TestMalformedTZData(t *testing.T) {
	// The goal here is just that malformed tzdata results in an error, not a panic.
	issue29437 := "TZif\x00000000000000000\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0000"
	_, err := time.LoadLocationFromTZData("abc", []byte(issue29437))
	if err == nil {
		t.Error("expected error, got none")
	}
}

var slimTests = []struct {
	zoneName   string
	fileName   string
	date       func(*time.Location) time.Time
	wantName   string
	wantOffset int
}{
	{
		// 2020b slim tzdata for Europe/Berlin.
		zoneName:   "Europe/Berlin",
		fileName:   "2020b_Europe_Berlin",
		date:       func(loc *time.Location) time.Time { return time.Date(2020, time.October, 29, 15, 30, 0, 0, loc) },
		wantName:   "CET",
		wantOffset: 3600,
	},
	{
		// 2021a slim tzdata for America/Nuuk.
		zoneName:   "America/Nuuk",
		fileName:   "2021a_America_Nuuk",
		date:       func(loc *time.Location) time.Time { return time.Date(2020, time.October, 29, 15, 30, 0, 0, loc) },
		wantName:   "-03",
		wantOffset: -10800,
	},
	{
		// 2021a slim tzdata for Asia/Gaza.
		zoneName:   "Asia/Gaza",
		fileName:   "2021a_Asia_Gaza",
		date:       func(loc *time.Location) time.Time { return time.Date(2020, time.October, 29, 15, 30, 0, 0, loc) },
		wantName:   "EET",
		wantOffset: 7200,
	},
	{
		// 2021a slim tzdata for Europe/Dublin.
		zoneName:   "Europe/Dublin",
		fileName:   "2021a_Europe_Dublin",
		date:       func(loc *time.Location) time.Time { return time.Date(2021, time.April, 2, 11, 12, 13, 0, loc) },
		wantName:   "IST",
		wantOffset: 3600,
	},
}

func TestLoadLocationFromTZDataSlim(t *testing.T) {
	for _, test := range slimTests {
		tzData, err := os.ReadFile("testdata/" + test.fileName)
		if err != nil {
			t.Error(err)
			continue
		}
		reference, err := time.LoadLocationFromTZData(test.zoneName, tzData)
		if err != nil {
			t.Error(err)
			continue
		}

		d := test.date(reference)
		tzName, tzOffset := d.Zone()
		if tzName != test.wantName {
			t.Errorf("Zone name == %s, want %s", tzName, test.wantName)
		}
		if tzOffset != test.wantOffset {
			t.Errorf("Zone offset == %d, want %d", tzOffset, test.wantOffset)
		}
	}
}

func TestTzset(t *testing.T) {
	for _, test := range []struct {
		inStr string
		inEnd int64
		inSec int64
		name  string
		off   int
		start int64
		end   int64
		isDST bool
		ok    bool
	}{
		{"", 0, 0, "", 0, 0, 0, false, false},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2159200800, "PDT", -7 * 60 * 60, 2152173600, 2172733200, true, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2152173599, "PST", -8 * 60 * 60, 2145916800, 2152173600, false, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2152173600, "PDT", -7 * 60 * 60, 2152173600, 2172733200, true, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2152173601, "PDT", -7 * 60 * 60, 2152173600, 2172733200, true, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2172733199, "PDT", -7 * 60 * 60, 2152173600, 2172733200, true, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2172733200, "PST", -8 * 60 * 60, 2172733200, 2177452800, false, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2172733201, "PST", -8 * 60 * 60, 2172733200, 2177452800, false, true},
		{"KST-9", 592333200, 1677246697, "KST", 9 * 60 * 60, 592333200, 1<<63 - 1, false, true},
	} {
		name, off, start, end, isDST, ok := time.Tzset(test.inStr, test.inEnd, test.inSec)
		if name != test.name || off != test.off || start != test.start || end != test.end || isDST != test.isDST || ok != test.ok {
			t.Errorf("tzset(%q, %d, %d) = %q, %d, %d, %d, %t, %t, want %q, %d, %d, %d, %t, %t", test.inStr, test.inEnd, test.inSec, name, off, start, end, isDST, ok, test.name, test.off, test.start, test.end, test.isDST, test.ok)
		}
	}
}

func TestTzsetName(t *testing.T) {
	for _, test := range []struct {
		in   string
		name string
		out  string
		ok   bool
	}{
		{"", "", "", false},
		{"X", "", "", false},
		{"PST", "PST", "", true},
		{"PST8PDT", "PST", "8PDT", true},
		{"PST-08", "PST", "-08", true},
		{"<A+B>+08", "A+B", "+08", true},
	} {
		name, out, ok := time.TzsetName(test.in)
		if name != test.name || out != test.out || ok != test.ok {
			t.Errorf("tzsetName(%q) = %q, %q, %t, want %q, %q, %t", test.in, name, out, ok, test.name, test.out, test.ok)
		}
	}
}

func TestTzsetOffset(t *testing.T) {
	for _, test := range []struct {
		in  string
		off int
		out string
		ok  bool
	}{
		{"", 0, "", false},
		{"X", 0, "", false},
		{"+", 0, "", false},
		{"+08", 8 * 60 * 60, "", true},
		{"-01:02:03", -1*60*60 - 2*60 - 3, "", true},
		{"01", 1 * 60 * 60, "", true},
		{"100", 100 * 60 * 60, "", true},
		{"1000", 0, "", false},
		{"8PDT", 8 * 60 * 60, "PDT", true},
	} {
		off, out, ok := time.TzsetOffset(test.in)
		if off != test.off || out != test.out || ok != test.ok {
			t.Errorf("tzsetName(%q) = %d, %q, %t, want %d, %q, %t", test.in, off, out, ok, test.off, test.out, test.ok)
		}
	}
}

func TestTzsetRule(t *testing.T) {
	for _, test := range []struct {
		in  string
		r   time.Rule
		out string
		ok  bool
	}{
		{"", time.Rule{}, "", false},
		{"X", time.Rule{}, "", false},
		{"J10", time.Rule{Kind: time.RuleJulian, Day: 10, Time: 2 * 60 * 60}, "", true},
		{"20", time.Rule{Kind: time.RuleDOY, Day: 20, Time: 2 * 60 * 60}, "", true},
		{"M1.2.3", time.Rule{Kind: time.RuleMonthWeekDay, Mon: 1, Week: 2, Day: 3, Time: 2 * 60 * 60}, "", true},
		{"30/03:00:00", time.Rule{Kind: time.RuleDOY, Day: 30, Time: 3 * 60 * 60}, "", true},
		{"M4.5.6/03:00:00", time.Rule{Kind: time.RuleMonthWeekDay, Mon: 4, Week: 5, Day: 6, Time: 3 * 60 * 60}, "", true},
		{"M4.5.7/03:00:00", time.Rule{}, "", false},
		{"M4.5.6/-04", time.Rule{Kind: time.RuleMonthWeekDay, Mon: 4, Week: 5, Day: 6, Time: -4 * 60 * 60}, "", true},
	} {
		r, out, ok := time.TzsetRule(test.in)
		if r != test.r || out != test.out || ok != test.ok {
			t.Errorf("tzsetName(%q) = %#v, %q, %t, want %#v, %q, %t", test.in, r, out, ok, test.r, test.out, test.ok)
		}
	}
}
