// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"errors"
	"fmt"
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

	defer os.Setenv(env, os.Getenv(env))
	os.Setenv(env, testZoneinfo)

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
	defer os.Setenv(env, os.Getenv(env))
	os.Setenv(env, "")

	bad := []string{
		"/usr/foo/Foo",
		"\\UNC\foo",
		"..",
		"a..",
	}
	for _, v := range bad {
		_, err := time.LoadLocation(v)
		if err != time.ErrLocation {
			t.Errorf("LoadLocation(%q) error = %v; want ErrLocation", v, err)
		}
	}
}

func TestVersion3(t *testing.T) {
	time.ForceZipFileForTesting(true)
	defer time.ForceZipFileForTesting(false)
	_, err := time.LoadLocation("Asia/Jerusalem")
	if err != nil {
		t.Fatal(err)
	}
}

// Test that we get the correct results for times before the first
// transition time. To do this we explicitly check early dates in a
// couple of specific timezones.
func TestFirstZone(t *testing.T) {
	time.ForceZipFileForTesting(true)
	defer time.ForceZipFileForTesting(false)

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
	time.ForceZipFileForTesting(true)
	defer time.ForceZipFileForTesting(false)

	const locationName = "Asia/Jerusalem"
	reference, err := time.LoadLocation(locationName)
	if err != nil {
		t.Fatal(err)
	}

	tzinfo, err := time.LoadTzinfo(locationName, time.OrigZoneSources[len(time.OrigZoneSources)-1])
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
	time.ForceZipFileForTesting(true)
	defer time.ForceZipFileForTesting(false)

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

func TestLoadLocationFromTZDataSlim(t *testing.T) {
	// A 2020b slim tzdata for Europe/Berlin
	tzData := "TZif2\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00TZif2\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00<\x00\x00\x00\x04\x00\x00\x00\x12\xff\xff\xff\xffo\xa2a\xf8\xff\xff\xff\xff\x9b\f\x17`\xff\xff\xff\xff\x9b\xd5\xda\xf0\xff\xff\xff\xff\x9cٮ\x90\xff\xff\xff\xff\x9d\xa4\xb5\x90\xff\xff\xff\xff\x9e\xb9\x90\x90\xff\xff\xff\xff\x9f\x84\x97\x90\xff\xff\xff\xff\xc8\tq\x90\xff\xff\xff\xff\xcc\xe7K\x10\xff\xff\xff\xffͩ\x17\x90\xff\xff\xff\xff\u03a2C\x10\xff\xff\xff\xffϒ4\x10\xff\xff\xff\xffЂ%\x10\xff\xff\xff\xff\xd1r\x16\x10\xff\xff\xff\xffѶ\x96\x00\xff\xff\xff\xff\xd2X\xbe\x80\xff\xff\xff\xffҡO\x10\xff\xff\xff\xff\xd3c\x1b\x90\xff\xff\xff\xff\xd4K#\x90\xff\xff\xff\xff\xd59\xd1 \xff\xff\xff\xff\xd5g\xe7\x90\xff\xff\xff\xffըs\x00\xff\xff\xff\xff\xd6)\xb4\x10\xff\xff\xff\xff\xd7,\x1a\x10\xff\xff\xff\xff\xd8\t\x96\x10\xff\xff\xff\xff\xd9\x02\xc1\x90\xff\xff\xff\xff\xd9\xe9x\x10\x00\x00\x00\x00\x13MD\x10\x00\x00\x00\x00\x143\xfa\x90\x00\x00\x00\x00\x15#\xeb\x90\x00\x00\x00\x00\x16\x13ܐ\x00\x00\x00\x00\x17\x03͐\x00\x00\x00\x00\x17\xf3\xbe\x90\x00\x00\x00\x00\x18㯐\x00\x00\x00\x00\x19Ӡ\x90\x00\x00\x00\x00\x1aÑ\x90\x00\x00\x00\x00\x1b\xbc\xbd\x10\x00\x00\x00\x00\x1c\xac\xae\x10\x00\x00\x00\x00\x1d\x9c\x9f\x10\x00\x00\x00\x00\x1e\x8c\x90\x10\x00\x00\x00\x00\x1f|\x81\x10\x00\x00\x00\x00 lr\x10\x00\x00\x00\x00!\\c\x10\x00\x00\x00\x00\"LT\x10\x00\x00\x00\x00#<E\x10\x00\x00\x00\x00$,6\x10\x00\x00\x00\x00%\x1c'\x10\x00\x00\x00\x00&\f\x18\x10\x00\x00\x00\x00'\x05C\x90\x00\x00\x00\x00'\xf54\x90\x00\x00\x00\x00(\xe5%\x90\x00\x00\x00\x00)\xd5\x16\x90\x00\x00\x00\x00*\xc5\a\x90\x00\x00\x00\x00+\xb4\xf8\x90\x00\x00\x00\x00,\xa4\xe9\x90\x00\x00\x00\x00-\x94ڐ\x00\x00\x00\x00.\x84ː\x00\x00\x00\x00/t\xbc\x90\x00\x00\x00\x000d\xad\x90\x00\x00\x00\x001]\xd9\x10\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x03\x01\x02\x01\x02\x01\x03\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x02\x01\x00\x00\f\x88\x00\x00\x00\x00\x1c \x01\x04\x00\x00\x0e\x10\x00\t\x00\x00*0\x01\rLMT\x00CEST\x00CET\x00CEMT\x00\nCET-1CEST,M3.5.0,M10.5.0/3\n"

	reference, err := time.LoadLocationFromTZData("Europe/Berlin", []byte(tzData))
	if err != nil {
		t.Fatal(err)
	}

	d := time.Date(2020, time.October, 29, 15, 30, 0, 0, reference)
	tzName, tzOffset := d.Zone()
	if want := "CET"; tzName != want {
		t.Errorf("Zone name == %s, want %s", tzName, want)
	}
	if want := 3600; tzOffset != want {
		t.Errorf("Zone offset == %d, want %d", tzOffset, want)
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
		ok    bool
	}{
		{"", 0, 0, "", 0, 0, 0, false},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2159200800, "PDT", -7 * 60 * 60, 2152173600, 2172733200, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2152173599, "PST", -8 * 60 * 60, 2145916800, 2152173600, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2152173600, "PDT", -7 * 60 * 60, 2152173600, 2172733200, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2152173601, "PDT", -7 * 60 * 60, 2152173600, 2172733200, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2172733199, "PDT", -7 * 60 * 60, 2152173600, 2172733200, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2172733200, "PST", -8 * 60 * 60, 2172733200, 2177452800, true},
		{"PST8PDT,M3.2.0,M11.1.0", 0, 2172733201, "PST", -8 * 60 * 60, 2172733200, 2177452800, true},
	} {
		name, off, start, end, ok := time.Tzset(test.inStr, test.inEnd, test.inSec)
		if name != test.name || off != test.off || start != test.start || end != test.end || ok != test.ok {
			t.Errorf("tzset(%q, %d, %d) = %q, %d, %d, %d, %t, want %q, %d, %d, %d, %t", test.inStr, test.inEnd, test.inSec, name, off, start, end, ok, test.name, test.off, test.start, test.end, test.ok)
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
		{"100", 0, "", false},
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
	} {
		r, out, ok := time.TzsetRule(test.in)
		if r != test.r || out != test.out || ok != test.ok {
			t.Errorf("tzsetName(%q) = %#v, %q, %t, want %#v, %q, %t", test.in, r, out, ok, test.r, test.out, test.ok)
		}
	}
}
