// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import "testing"

func TestParseVersionRow(t *testing.T) {
	tests := []struct {
		row  string
		want versionedRow
	}{
		{
			row: "# comment",
		},
		{
			row: "",
		},
		{
			row: "pkg archive/tar, type Writer struct",
			want: versionedRow{
				pkg:  "archive/tar",
				kind: "type",
				name: "Writer",
			},
		},
		{
			row: "pkg archive/tar, type Header struct, AccessTime time.Time",
			want: versionedRow{
				pkg:        "archive/tar",
				kind:       "field",
				structName: "Header",
				name:       "AccessTime",
			},
		},
		{
			row: "pkg archive/tar, method (*Reader) Read([]uint8) (int, error)",
			want: versionedRow{
				pkg:  "archive/tar",
				kind: "method",
				name: "Read",
				recv: "*Reader",
			},
		},
		{
			row: "pkg archive/zip, func FileInfoHeader(os.FileInfo) (*FileHeader, error)",
			want: versionedRow{
				pkg:  "archive/zip",
				kind: "func",
				name: "FileInfoHeader",
			},
		},
	}

	for i, tt := range tests {
		got, ok := parseRow(tt.row)
		if !ok {
			got = versionedRow{}
		}
		if got != tt.want {
			t.Errorf("%d. parseRow(%q) = %+v; want %+v", i, tt.row, got, tt.want)
		}
	}
}

func TestAPIVersion(t *testing.T) {
	av, err := parsePackageAPIInfo()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		kind     string
		pkg      string
		name     string
		receiver string
		want     string
	}{
		// Things that were added post-1.0 should appear
		{"func", "archive/tar", "FileInfoHeader", "", "1.1"},
		{"type", "bufio", "Scanner", "", "1.1"},
		{"method", "bufio", "WriteTo", "*Reader", "1.1"},

		{"func", "bytes", "LastIndexByte", "", "1.5"},
		{"type", "crypto", "Decrypter", "", "1.5"},
		{"method", "crypto/rsa", "Decrypt", "*PrivateKey", "1.5"},
		{"method", "debug/dwarf", "GoString", "Class", "1.5"},

		{"func", "os", "IsTimeout", "", "1.10"},
		{"type", "strings", "Builder", "", "1.10"},
		{"method", "strings", "WriteString", "*Builder", "1.10"},

		// Things from package syscall should never appear
		{"func", "syscall", "FchFlags", "", ""},
		{"type", "syscall", "Inet4Pktinfo", "", ""},

		// Things added in Go 1 should never appear
		{"func", "archive/tar", "NewReader", "", ""},
		{"type", "archive/tar", "Header", "", ""},
		{"method", "archive/tar", "Next", "*Reader", ""},
	} {
		if got := av.sinceVersionFunc(tc.kind, tc.receiver, tc.name, tc.pkg); got != tc.want {
			t.Errorf(`sinceFunc("%s", "%s", "%s", "%s") = "%s"; want "%s"`, tc.kind, tc.receiver, tc.name, tc.pkg, got, tc.want)
		}
	}
}
