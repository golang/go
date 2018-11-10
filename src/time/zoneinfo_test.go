// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"fmt"
	"os"
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
		t.Errorf("zoneinfo does not match env variable: got %q want %q", zoneinfo, testZoneinfo)
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
