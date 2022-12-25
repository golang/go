// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package user

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
	"unicode/utf8"
)

func TestQueryNoUserdb(t *testing.T) {
	cl := &userdbClient{dir: "/non/existent"}
	if _, ok, err := cl.lookupGroup(context.Background(), "stdlibcontrib"); ok {
		t.Fatalf("should fail but lookup has been handled or error is nil: %v", err)
	}
}

type userdbTestData map[string]udbResponse

type udbResponse struct {
	data  []byte
	delay time.Duration
}

func userdbServer(t *testing.T, sockFn string, data userdbTestData) {
	ready := make(chan struct{})
	go func() {
		if err := serveUserdb(ready, sockFn, data); err != nil {
			t.Error(err)
		}
	}()
	<-ready
}

func (u userdbTestData) String() string {
	var s strings.Builder
	for k, v := range u {
		s.WriteString("Request:\n")
		s.WriteString(k)
		s.WriteString("\nResponse:\n")
		if v.delay > 0 {
			s.WriteString("Delay: ")
			s.WriteString(v.delay.String())
			s.WriteString("\n")
		}
		s.WriteString("Data:\n")
		s.Write(v.data)
		s.WriteString("\n")
	}
	return s.String()
}

// serverUserdb is a simple userdb server that replies to VARLINK method calls.
// A message is sent on the ready channel when the server is ready to accept calls.
// The server will reply to each request in the data map. If a request is not
// found in the map, the server will return an error.
func serveUserdb(ready chan<- struct{}, sockFn string, data userdbTestData) error {
	sockFd, err := syscall.Socket(syscall.AF_UNIX, syscall.SOCK_STREAM, 0)
	if err != nil {
		return err
	}
	defer syscall.Close(sockFd)
	if err := syscall.Bind(sockFd, &syscall.SockaddrUnix{Name: sockFn}); err != nil {
		return err
	}
	if err := syscall.Listen(sockFd, 1); err != nil {
		return err
	}

	// Send ready signal.
	ready <- struct{}{}

	var srvGroup sync.WaitGroup

	srvErrs := make(chan error, len(data))
	for len(data) != 0 {
		nfd, _, err := syscall.Accept(sockFd)
		if err != nil {
			syscall.Close(nfd)
			return err
		}

		// Read request.
		buf := make([]byte, 4096)
		n, err := syscall.Read(nfd, buf)
		if err != nil {
			syscall.Close(nfd)
			return err
		}
		if n == 0 {
			// Client went away.
			continue
		}
		if buf[n-1] != 0 {
			syscall.Close(nfd)
			return errors.New("request not null terminated")
		}
		// Remove null terminator.
		buf = buf[:n-1]
		got := string(buf)

		// Fetch response for request.
		response, ok := data[got]
		if !ok {
			syscall.Close(nfd)
			msg := "unexpected request:\n" + got + "\n\ndata:\n" + data.String()
			return errors.New(msg)
		}
		delete(data, got)

		srvGroup.Add(1)
		go func() {
			defer srvGroup.Done()
			if err := serveClient(nfd, response); err != nil {
				srvErrs <- err
			}
		}()
	}

	srvGroup.Wait()
	// Combine serve errors if any.
	if len(srvErrs) > 0 {
		var errs []error
		for err := range srvErrs {
			errs = append(errs, err)
		}
		return errors.Join(errs...)
	}

	return nil
}

func serveClient(fd int, response udbResponse) error {
	defer syscall.Close(fd)
	time.Sleep(response.delay)
	data := response.data
	if len(data) != 0 && data[len(data)-1] != 0 {
		data = append(data, 0)
	}
	written := 0
	for written < len(data) {
		if n, err := syscall.Write(fd, data[written:]); err != nil {
			return err
		} else {
			written += n
		}
	}
	return nil
}

func TestSlowUserdbLookup(t *testing.T) {
	tmpdir := t.TempDir()
	data := userdbTestData{
		`{"method":"io.systemd.UserDatabase.GetGroupRecord","parameters":{"service":"io.systemd.Multiplexer","groupName":"stdlibcontrib"}}`: udbResponse{
			delay: time.Hour,
		},
	}
	userdbServer(t, tmpdir+"/"+svcMultiplexer, data)
	cl := &userdbClient{dir: tmpdir}
	// Lookup should timeout.
	ctx, cancel := context.WithTimeout(context.Background(), time.Microsecond)
	defer cancel()
	if _, ok, _ := cl.lookupGroup(ctx, "stdlibcontrib"); ok {
		t.Fatalf("lookup should not be handled but was")
	}
}

func TestFastestUserdbLookup(t *testing.T) {
	tmpdir := t.TempDir()
	fastData := userdbTestData{
		`{"method":"io.systemd.UserDatabase.GetGroupRecord","parameters":{"service":"fast","groupName":"stdlibcontrib"}}`: udbResponse{
			data: []byte(
				`{"parameters":{"record":{"groupName":"stdlibcontrib","gid":181,"members":["stdlibcontrib"],"status":{"ecb5a44f1a5846ad871566e113bf8937":{"service":"io.systemd.NameServiceSwitch"}}},"incomplete":false}}`,
			),
		},
	}
	slowData := userdbTestData{
		`{"method":"io.systemd.UserDatabase.GetGroupRecord","parameters":{"service":"slow","groupName":"stdlibcontrib"}}`: udbResponse{
			delay: 50 * time.Millisecond,
			data: []byte(
				`{"parameters":{"record":{"groupName":"stdlibcontrib","gid":182,"members":["stdlibcontrib"],"status":{"ecb5a44f1a5846ad871566e113bf8937":{"service":"io.systemd.NameServiceSwitch"}}},"incomplete":false}}`,
			),
		},
	}
	userdbServer(t, tmpdir+"/"+"fast", fastData)
	userdbServer(t, tmpdir+"/"+"slow", slowData)
	cl := &userdbClient{dir: tmpdir}
	group, ok, err := cl.lookupGroup(context.Background(), "stdlibcontrib")
	if !ok {
		t.Fatalf("lookup should be handled but was not")
	}
	if err != nil {
		t.Fatalf("lookup should not fail but did: %v", err)
	}
	if group.Gid != "181" {
		t.Fatalf("lookup should return group 181 but returned %s", group.Gid)
	}
}

func TestUserdbLookupGroup(t *testing.T) {
	tmpdir := t.TempDir()
	data := userdbTestData{
		`{"method":"io.systemd.UserDatabase.GetGroupRecord","parameters":{"service":"io.systemd.Multiplexer","groupName":"stdlibcontrib"}}`: udbResponse{
			data: []byte(
				`{"parameters":{"record":{"groupName":"stdlibcontrib","gid":181,"members":["stdlibcontrib"],"status":{"ecb5a44f1a5846ad871566e113bf8937":{"service":"io.systemd.NameServiceSwitch"}}},"incomplete":false}}`,
			),
		},
	}
	userdbServer(t, tmpdir+"/"+svcMultiplexer, data)

	groupname := "stdlibcontrib"
	want := &Group{
		Name: "stdlibcontrib",
		Gid:  "181",
	}
	cl := &userdbClient{dir: tmpdir}
	got, ok, err := cl.lookupGroup(context.Background(), groupname)
	if !ok {
		t.Fatal("lookup should have been handled")
	}
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("lookupGroup(%s) = %v, want %v", groupname, got, want)
	}
}

func TestUserdbLookupUser(t *testing.T) {
	tmpdir := t.TempDir()
	data := userdbTestData{
		`{"method":"io.systemd.UserDatabase.GetUserRecord","parameters":{"service":"io.systemd.Multiplexer","userName":"stdlibcontrib"}}`: udbResponse{
			data: []byte(
				`{"parameters":{"record":{"userName":"stdlibcontrib","uid":181,"gid":181,"realName":"Stdlib Contrib","homeDirectory":"/home/stdlibcontrib","status":{"ecb5a44f1a5846ad871566e113bf8937":{"service":"io.systemd.NameServiceSwitch"}}},"incomplete":false}}`,
			),
		},
	}
	userdbServer(t, tmpdir+"/"+svcMultiplexer, data)

	username := "stdlibcontrib"
	want := &User{
		Uid:      "181",
		Gid:      "181",
		Username: "stdlibcontrib",
		Name:     "Stdlib Contrib",
		HomeDir:  "/home/stdlibcontrib",
	}
	cl := &userdbClient{dir: tmpdir}
	got, ok, err := cl.lookupUser(context.Background(), username)
	if !ok {
		t.Fatal("lookup should have been handled")
	}
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("lookupUser(%s) = %v, want %v", username, got, want)
	}
}

func TestUserdbLookupGroupIds(t *testing.T) {
	tmpdir := t.TempDir()
	data := userdbTestData{
		`{"method":"io.systemd.UserDatabase.GetMemberships","parameters":{"service":"io.systemd.Multiplexer","userName":"stdlibcontrib"},"more":true}`: udbResponse{
			data: []byte(
				`{"parameters":{"userName":"stdlibcontrib","groupName":"stdlib"},"continues":true}` + "\x00" + `{"parameters":{"userName":"stdlibcontrib","groupName":"contrib"}}`,
			),
		},
		// group records
		`{"method":"io.systemd.UserDatabase.GetGroupRecord","parameters":{"service":"io.systemd.Multiplexer","groupName":"stdlibcontrib"}}`: udbResponse{
			data: []byte(
				`{"parameters":{"record":{"groupName":"stdlibcontrib","members":["stdlibcontrib"],"gid":181,"status":{"ecb5a44f1a5846ad871566e113bf8937":{"service":"io.systemd.NameServiceSwitch"}}},"incomplete":false}}`,
			),
		},
		`{"method":"io.systemd.UserDatabase.GetGroupRecord","parameters":{"service":"io.systemd.Multiplexer","groupName":"stdlib"}}`: udbResponse{
			data: []byte(
				`{"parameters":{"record":{"groupName":"stdlib","members":["stdlibcontrib"],"gid":182,"status":{"ecb5a44f1a5846ad871566e113bf8937":{"service":"io.systemd.NameServiceSwitch"}}},"incomplete":false}}`,
			),
		},
		`{"method":"io.systemd.UserDatabase.GetGroupRecord","parameters":{"service":"io.systemd.Multiplexer","groupName":"contrib"}}`: udbResponse{
			data: []byte(
				`{"parameters":{"record":{"groupName":"contrib","members":["stdlibcontrib"],"gid":183,"status":{"ecb5a44f1a5846ad871566e113bf8937":{"service":"io.systemd.NameServiceSwitch"}}},"incomplete":false}}`,
			),
		},
	}
	userdbServer(t, tmpdir+"/"+svcMultiplexer, data)

	username := "stdlibcontrib"
	want := []string{"181", "182", "183"}
	cl := &userdbClient{dir: tmpdir}
	got, ok, err := cl.lookupGroupIds(context.Background(), username)
	if !ok {
		t.Fatal("lookup should have been handled")
	}
	if err != nil {
		t.Fatal(err)
	}
	// Result order is not specified so sort it.
	sort.Strings(got)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("lookupGroupIds(%s) = %v, want %v", username, got, want)
	}
}

var findElementStartTestCases = []struct {
	in   []byte
	want []byte
	err  bool
}{
	{in: []byte(`:`), want: []byte(``)},
	{in: []byte(`: `), want: []byte(``)},
	{in: []byte(`:"foo"`), want: []byte(`"foo"`)},
	{in: []byte(`  :"foo"`), want: []byte(`"foo"`)},
	{in: []byte(` 1231 :"foo"`), err: true},
	{in: []byte(``), err: true},
	{in: []byte(`"foo"`), err: true},
	{in: []byte(`foo`), err: true},
}

func TestFindElementStart(t *testing.T) {
	for i, tc := range findElementStartTestCases {
		t.Run("#"+strconv.Itoa(i), func(t *testing.T) {
			got, err := findElementStart(tc.in)
			if tc.err && err == nil {
				t.Errorf("want err for findElementStart(%s), got nil", tc.in)
			}
			if !tc.err {
				if err != nil {
					t.Errorf("findElementStart(%s) unexpected error: %s", tc.in, err.Error())
				}
				if !bytes.Contains(tc.in, got) {
					t.Errorf("%s should contain %s but does not", tc.in, got)
				}
			}
		})
	}
}

func FuzzFindElementStart(f *testing.F) {
	for _, tc := range findElementStartTestCases {
		if !tc.err {
			f.Add(tc.in)
		}
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		if out, err := findElementStart(b); err == nil && !bytes.Contains(b, out) {
			t.Errorf("%s, %v", out, err)
		}
	})
}

var parseJSONStringTestCases = []struct {
	in   []byte
	want string
	err  bool
}{
	{in: []byte(`:""`)},
	{in: []byte(`:"\n"`), want: "\n"},
	{in: []byte(`: "\""`), want: "\""},
	{in: []byte(`:"\t \\"`), want: "\t \\"},
	{in: []byte(`:"\\\\"`), want: `\\`},
	{in: []byte(`::`), err: true},
	{in: []byte(`""`), err: true},
	{in: []byte(`"`), err: true},
	{in: []byte(":\"0\xE5"), err: true},
	{in: []byte{':', '"', 0xFE, 0xFE, 0xFF, 0xFF, '"'}, want: "\uFFFD\uFFFD\uFFFD\uFFFD"},
	{in: []byte(`:"\u0061a"`), want: "aa"},
	{in: []byte(`:"\u0159\u0170"`), want: "řŰ"},
	{in: []byte(`:"\uD800\uDC00"`), want: "\U00010000"},
	{in: []byte(`:"\uD800"`), want: "\uFFFD"},
	{in: []byte(`:"\u000"`), err: true},
	{in: []byte(`:"\u00MF"`), err: true},
	{in: []byte(`:"\uD800\uDC0"`), err: true},
}

func TestParseJSONString(t *testing.T) {
	for i, tc := range parseJSONStringTestCases {
		t.Run("#"+strconv.Itoa(i), func(t *testing.T) {
			got, err := parseJSONString(tc.in)
			if tc.err && err == nil {
				t.Errorf("want err for parseJSONString(%s), got nil", tc.in)
			}
			if !tc.err {
				if err != nil {
					t.Errorf("parseJSONString(%s) unexpected error: %s", tc.in, err.Error())
				}
				if tc.want != got {
					t.Errorf("parseJSONString(%s) = %s, want %s", tc.in, got, tc.want)
				}
			}
		})
	}
}

func FuzzParseJSONString(f *testing.F) {
	for _, tc := range parseJSONStringTestCases {
		f.Add(tc.in)
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		if out, err := parseJSONString(b); err == nil && !utf8.ValidString(out) {
			t.Errorf("parseJSONString(%s) = %s, invalid string", b, out)
		}
	})
}

var parseJSONInt64TestCases = []struct {
	in   []byte
	want int64
	err  bool
}{
	{in: []byte(":1235"), want: 1235},
	{in: []byte(": 123"), want: 123},
	{in: []byte(":0")},
	{in: []byte(":5012313123131231"), want: 5012313123131231},
	{in: []byte("1231"), err: true},
}

func TestParseJSONInt64(t *testing.T) {
	for i, tc := range parseJSONInt64TestCases {
		t.Run("#"+strconv.Itoa(i), func(t *testing.T) {
			got, err := parseJSONInt64(tc.in)
			if tc.err && err == nil {
				t.Errorf("want err for parseJSONInt64(%s), got nil", tc.in)
			}
			if !tc.err {
				if err != nil {
					t.Errorf("parseJSONInt64(%s) unexpected error: %s", tc.in, err.Error())
				}
				if tc.want != got {
					t.Errorf("parseJSONInt64(%s) = %d, want %d", tc.in, got, tc.want)
				}
			}
		})
	}
}

func FuzzParseJSONInt64(f *testing.F) {
	for _, tc := range parseJSONInt64TestCases {
		f.Add(tc.in)
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		if out, err := parseJSONInt64(b); err == nil &&
			!bytes.Contains(b, []byte(strconv.FormatInt(out, 10))) {
			t.Errorf("parseJSONInt64(%s) = %d, %v", b, out, err)
		}
	})
}

var parseJSONBooleanTestCases = []struct {
	in   []byte
	want bool
	err  bool
}{
	{in: []byte(": true "), want: true},
	{in: []byte(":true  "), want: true},
	{in: []byte(": false  "), want: false},
	{in: []byte(":false  "), want: false},
	{in: []byte("true"), err: true},
	{in: []byte("false"), err: true},
	{in: []byte("foo"), err: true},
}

func TestParseJSONBoolean(t *testing.T) {
	for i, tc := range parseJSONBooleanTestCases {
		t.Run("#"+strconv.Itoa(i), func(t *testing.T) {
			got, err := parseJSONBoolean(tc.in)
			if tc.err && err == nil {
				t.Errorf("want err for parseJSONBoolean(%s), got nil", tc.in)
			}
			if !tc.err {
				if err != nil {
					t.Errorf("parseJSONBoolean(%s) unexpected error: %s", tc.in, err.Error())
				}
				if tc.want != got {
					t.Errorf("parseJSONBoolean(%s) = %t, want %t", tc.in, got, tc.want)
				}
			}
		})
	}
}

func FuzzParseJSONBoolean(f *testing.F) {
	for _, tc := range parseJSONBooleanTestCases {
		f.Add(tc.in)
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		if out, err := parseJSONBoolean(b); err == nil && !bytes.Contains(b, []byte(strconv.FormatBool(out))) {
			t.Errorf("parseJSONBoolean(%s) = %t, %v", b, out, err)
		}
	})
}
