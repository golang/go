// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package registry_test

import (
	"bytes"
	"crypto/rand"
	"os"
	"syscall"
	"testing"
	"unsafe"

	"internal/syscall/windows/registry"
)

func randKeyName(prefix string) string {
	const numbers = "0123456789"
	buf := make([]byte, 10)
	rand.Read(buf)
	for i, b := range buf {
		buf[i] = numbers[b%byte(len(numbers))]
	}
	return prefix + string(buf)
}

func TestReadSubKeyNames(t *testing.T) {
	k, err := registry.OpenKey(registry.CLASSES_ROOT, "TypeLib", registry.ENUMERATE_SUB_KEYS)
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()

	names, err := k.ReadSubKeyNames()
	if err != nil {
		t.Fatal(err)
	}
	var foundStdOle bool
	for _, name := range names {
		// Every PC has "stdole 2.0 OLE Automation" library installed.
		if name == "{00020430-0000-0000-C000-000000000046}" {
			foundStdOle = true
		}
	}
	if !foundStdOle {
		t.Fatal("could not find stdole 2.0 OLE Automation")
	}
}

func TestCreateOpenDeleteKey(t *testing.T) {
	k, err := registry.OpenKey(registry.CURRENT_USER, "Software", registry.QUERY_VALUE)
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()

	testKName := randKeyName("TestCreateOpenDeleteKey_")

	testK, exist, err := registry.CreateKey(k, testKName, registry.CREATE_SUB_KEY)
	if err != nil {
		t.Fatal(err)
	}
	defer testK.Close()

	if exist {
		t.Fatalf("key %q already exists", testKName)
	}

	testKAgain, exist, err := registry.CreateKey(k, testKName, registry.CREATE_SUB_KEY)
	if err != nil {
		t.Fatal(err)
	}
	defer testKAgain.Close()

	if !exist {
		t.Fatalf("key %q should already exist", testKName)
	}

	testKOpened, err := registry.OpenKey(k, testKName, registry.ENUMERATE_SUB_KEYS)
	if err != nil {
		t.Fatal(err)
	}
	defer testKOpened.Close()

	err = registry.DeleteKey(k, testKName)
	if err != nil {
		t.Fatal(err)
	}

	testKOpenedAgain, err := registry.OpenKey(k, testKName, registry.ENUMERATE_SUB_KEYS)
	if err == nil {
		defer testKOpenedAgain.Close()
		t.Fatalf("key %q should already been deleted", testKName)
	}
	if err != registry.ErrNotExist {
		t.Fatalf(`unexpected error ("not exist" expected): %v`, err)
	}
}

func equalStringSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	if a == nil {
		return true
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type ValueTest struct {
	Type     uint32
	Name     string
	Value    interface{}
	WillFail bool
}

var ValueTests = []ValueTest{
	{Type: registry.SZ, Name: "String1", Value: ""},
	{Type: registry.SZ, Name: "String2", Value: "\000", WillFail: true},
	{Type: registry.SZ, Name: "String3", Value: "Hello World"},
	{Type: registry.SZ, Name: "String4", Value: "Hello World\000", WillFail: true},
	{Type: registry.EXPAND_SZ, Name: "ExpString1", Value: ""},
	{Type: registry.EXPAND_SZ, Name: "ExpString2", Value: "\000", WillFail: true},
	{Type: registry.EXPAND_SZ, Name: "ExpString3", Value: "Hello World"},
	{Type: registry.EXPAND_SZ, Name: "ExpString4", Value: "Hello\000World", WillFail: true},
	{Type: registry.EXPAND_SZ, Name: "ExpString5", Value: "%PATH%"},
	{Type: registry.EXPAND_SZ, Name: "ExpString6", Value: "%NO_SUCH_VARIABLE%"},
	{Type: registry.EXPAND_SZ, Name: "ExpString7", Value: "%PATH%;."},
	{Type: registry.BINARY, Name: "Binary1", Value: []byte{}},
	{Type: registry.BINARY, Name: "Binary2", Value: []byte{1, 2, 3}},
	{Type: registry.BINARY, Name: "Binary3", Value: []byte{3, 2, 1, 0, 1, 2, 3}},
	{Type: registry.DWORD, Name: "Dword1", Value: uint64(0)},
	{Type: registry.DWORD, Name: "Dword2", Value: uint64(1)},
	{Type: registry.DWORD, Name: "Dword3", Value: uint64(0xff)},
	{Type: registry.DWORD, Name: "Dword4", Value: uint64(0xffff)},
	{Type: registry.QWORD, Name: "Qword1", Value: uint64(0)},
	{Type: registry.QWORD, Name: "Qword2", Value: uint64(1)},
	{Type: registry.QWORD, Name: "Qword3", Value: uint64(0xff)},
	{Type: registry.QWORD, Name: "Qword4", Value: uint64(0xffff)},
	{Type: registry.QWORD, Name: "Qword5", Value: uint64(0xffffff)},
	{Type: registry.QWORD, Name: "Qword6", Value: uint64(0xffffffff)},
	{Type: registry.MULTI_SZ, Name: "MultiString1", Value: []string{"a", "b", "c"}},
	{Type: registry.MULTI_SZ, Name: "MultiString2", Value: []string{"abc", "", "cba"}},
	{Type: registry.MULTI_SZ, Name: "MultiString3", Value: []string{""}},
	{Type: registry.MULTI_SZ, Name: "MultiString4", Value: []string{"abcdef"}},
	{Type: registry.MULTI_SZ, Name: "MultiString5", Value: []string{"\000"}, WillFail: true},
	{Type: registry.MULTI_SZ, Name: "MultiString6", Value: []string{"a\000b"}, WillFail: true},
	{Type: registry.MULTI_SZ, Name: "MultiString7", Value: []string{"ab", "\000", "cd"}, WillFail: true},
	{Type: registry.MULTI_SZ, Name: "MultiString8", Value: []string{"\000", "cd"}, WillFail: true},
	{Type: registry.MULTI_SZ, Name: "MultiString9", Value: []string{"ab", "\000"}, WillFail: true},
}

func setValues(t *testing.T, k registry.Key) {
	for _, test := range ValueTests {
		var err error
		switch test.Type {
		case registry.SZ:
			err = k.SetStringValue(test.Name, test.Value.(string))
		case registry.EXPAND_SZ:
			err = k.SetExpandStringValue(test.Name, test.Value.(string))
		case registry.MULTI_SZ:
			err = k.SetStringsValue(test.Name, test.Value.([]string))
		case registry.BINARY:
			err = k.SetBinaryValue(test.Name, test.Value.([]byte))
		case registry.DWORD:
			err = k.SetDWordValue(test.Name, uint32(test.Value.(uint64)))
		case registry.QWORD:
			err = k.SetQWordValue(test.Name, test.Value.(uint64))
		default:
			t.Fatalf("unsupported type %d for %s value", test.Type, test.Name)
		}
		if test.WillFail {
			if err == nil {
				t.Fatalf("setting %s value %q should fail, but succeeded", test.Name, test.Value)
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}

func enumerateValues(t *testing.T, k registry.Key) {
	names, err := k.ReadValueNames()
	if err != nil {
		t.Error(err)
		return
	}
	haveNames := make(map[string]bool)
	for _, n := range names {
		haveNames[n] = false
	}
	for _, test := range ValueTests {
		wantFound := !test.WillFail
		_, haveFound := haveNames[test.Name]
		if wantFound && !haveFound {
			t.Errorf("value %s is not found while enumerating", test.Name)
		}
		if haveFound && !wantFound {
			t.Errorf("value %s is found while enumerating, but expected to fail", test.Name)
		}
		if haveFound {
			delete(haveNames, test.Name)
		}
	}
	for n, v := range haveNames {
		t.Errorf("value %s (%v) is found while enumerating, but has not been created", n, v)
	}
}

func testErrNotExist(t *testing.T, name string, err error) {
	if err == nil {
		t.Errorf("%s value should not exist", name)
		return
	}
	if err != registry.ErrNotExist {
		t.Errorf("reading %s value should return 'not exist' error, but got: %s", name, err)
		return
	}
}

func testErrUnexpectedType(t *testing.T, test ValueTest, gottype uint32, err error) {
	if err == nil {
		t.Errorf("GetXValue(%q) should not succeed", test.Name)
		return
	}
	if err != registry.ErrUnexpectedType {
		t.Errorf("reading %s value should return 'unexpected key value type' error, but got: %s", test.Name, err)
		return
	}
	if gottype != test.Type {
		t.Errorf("want %s value type %v, got %v", test.Name, test.Type, gottype)
		return
	}
}

func testGetStringValue(t *testing.T, k registry.Key, test ValueTest) {
	got, gottype, err := k.GetStringValue(test.Name)
	if err != nil {
		t.Errorf("GetStringValue(%s) failed: %v", test.Name, err)
		return
	}
	if got != test.Value {
		t.Errorf("want %s value %q, got %q", test.Name, test.Value, got)
		return
	}
	if gottype != test.Type {
		t.Errorf("want %s value type %v, got %v", test.Name, test.Type, gottype)
		return
	}
	if gottype == registry.EXPAND_SZ {
		_, err = registry.ExpandString(got)
		if err != nil {
			t.Errorf("ExpandString(%s) failed: %v", got, err)
			return
		}
	}
}

func testGetIntegerValue(t *testing.T, k registry.Key, test ValueTest) {
	got, gottype, err := k.GetIntegerValue(test.Name)
	if err != nil {
		t.Errorf("GetIntegerValue(%s) failed: %v", test.Name, err)
		return
	}
	if got != test.Value.(uint64) {
		t.Errorf("want %s value %v, got %v", test.Name, test.Value, got)
		return
	}
	if gottype != test.Type {
		t.Errorf("want %s value type %v, got %v", test.Name, test.Type, gottype)
		return
	}
}

func testGetBinaryValue(t *testing.T, k registry.Key, test ValueTest) {
	got, gottype, err := k.GetBinaryValue(test.Name)
	if err != nil {
		t.Errorf("GetBinaryValue(%s) failed: %v", test.Name, err)
		return
	}
	if !bytes.Equal(got, test.Value.([]byte)) {
		t.Errorf("want %s value %v, got %v", test.Name, test.Value, got)
		return
	}
	if gottype != test.Type {
		t.Errorf("want %s value type %v, got %v", test.Name, test.Type, gottype)
		return
	}
}

func testGetStringsValue(t *testing.T, k registry.Key, test ValueTest) {
	got, gottype, err := k.GetStringsValue(test.Name)
	if err != nil {
		t.Errorf("GetStringsValue(%s) failed: %v", test.Name, err)
		return
	}
	if !equalStringSlice(got, test.Value.([]string)) {
		t.Errorf("want %s value %#v, got %#v", test.Name, test.Value, got)
		return
	}
	if gottype != test.Type {
		t.Errorf("want %s value type %v, got %v", test.Name, test.Type, gottype)
		return
	}
}

func testGetValue(t *testing.T, k registry.Key, test ValueTest, size int) {
	if size <= 0 {
		return
	}
	// read data with no buffer
	gotsize, gottype, err := k.GetValue(test.Name, nil)
	if err != nil {
		t.Errorf("GetValue(%s, [%d]byte) failed: %v", test.Name, size, err)
		return
	}
	if gotsize != size {
		t.Errorf("want %s value size of %d, got %v", test.Name, size, gotsize)
		return
	}
	if gottype != test.Type {
		t.Errorf("want %s value type %v, got %v", test.Name, test.Type, gottype)
		return
	}
	// read data with short buffer
	gotsize, gottype, err = k.GetValue(test.Name, make([]byte, size-1))
	if err == nil {
		t.Errorf("GetValue(%s, [%d]byte) should fail, but succeeded", test.Name, size-1)
		return
	}
	if err != registry.ErrShortBuffer {
		t.Errorf("reading %s value should return 'short buffer' error, but got: %s", test.Name, err)
		return
	}
	if gotsize != size {
		t.Errorf("want %s value size of %d, got %v", test.Name, size, gotsize)
		return
	}
	if gottype != test.Type {
		t.Errorf("want %s value type %v, got %v", test.Name, test.Type, gottype)
		return
	}
	// read full data
	gotsize, gottype, err = k.GetValue(test.Name, make([]byte, size))
	if err != nil {
		t.Errorf("GetValue(%s, [%d]byte) failed: %v", test.Name, size, err)
		return
	}
	if gotsize != size {
		t.Errorf("want %s value size of %d, got %v", test.Name, size, gotsize)
		return
	}
	if gottype != test.Type {
		t.Errorf("want %s value type %v, got %v", test.Name, test.Type, gottype)
		return
	}
	// check GetValue returns ErrNotExist as required
	_, _, err = k.GetValue(test.Name+"_not_there", make([]byte, size))
	if err == nil {
		t.Errorf("GetValue(%q) should not succeed", test.Name)
		return
	}
	if err != registry.ErrNotExist {
		t.Errorf("GetValue(%q) should return 'not exist' error, but got: %s", test.Name, err)
		return
	}
}

func testValues(t *testing.T, k registry.Key) {
	for _, test := range ValueTests {
		switch test.Type {
		case registry.SZ, registry.EXPAND_SZ:
			if test.WillFail {
				_, _, err := k.GetStringValue(test.Name)
				testErrNotExist(t, test.Name, err)
			} else {
				testGetStringValue(t, k, test)
				_, gottype, err := k.GetIntegerValue(test.Name)
				testErrUnexpectedType(t, test, gottype, err)
				// Size of utf16 string in bytes is not perfect,
				// but correct for current test values.
				// Size also includes terminating 0.
				testGetValue(t, k, test, (len(test.Value.(string))+1)*2)
			}
			_, _, err := k.GetStringValue(test.Name + "_string_not_created")
			testErrNotExist(t, test.Name+"_string_not_created", err)
		case registry.DWORD, registry.QWORD:
			testGetIntegerValue(t, k, test)
			_, gottype, err := k.GetBinaryValue(test.Name)
			testErrUnexpectedType(t, test, gottype, err)
			_, _, err = k.GetIntegerValue(test.Name + "_int_not_created")
			testErrNotExist(t, test.Name+"_int_not_created", err)
			size := 8
			if test.Type == registry.DWORD {
				size = 4
			}
			testGetValue(t, k, test, size)
		case registry.BINARY:
			testGetBinaryValue(t, k, test)
			_, gottype, err := k.GetStringsValue(test.Name)
			testErrUnexpectedType(t, test, gottype, err)
			_, _, err = k.GetBinaryValue(test.Name + "_byte_not_created")
			testErrNotExist(t, test.Name+"_byte_not_created", err)
			testGetValue(t, k, test, len(test.Value.([]byte)))
		case registry.MULTI_SZ:
			if test.WillFail {
				_, _, err := k.GetStringsValue(test.Name)
				testErrNotExist(t, test.Name, err)
			} else {
				testGetStringsValue(t, k, test)
				_, gottype, err := k.GetStringValue(test.Name)
				testErrUnexpectedType(t, test, gottype, err)
				size := 0
				for _, s := range test.Value.([]string) {
					size += len(s) + 1 // nil terminated
				}
				size += 1 // extra nil at the end
				size *= 2 // count bytes, not uint16
				testGetValue(t, k, test, size)
			}
			_, _, err := k.GetStringsValue(test.Name + "_strings_not_created")
			testErrNotExist(t, test.Name+"_strings_not_created", err)
		default:
			t.Errorf("unsupported type %d for %s value", test.Type, test.Name)
			continue
		}
	}
}

func testStat(t *testing.T, k registry.Key) {
	subk, _, err := registry.CreateKey(k, "subkey", registry.CREATE_SUB_KEY)
	if err != nil {
		t.Error(err)
		return
	}
	defer subk.Close()

	defer registry.DeleteKey(k, "subkey")

	ki, err := k.Stat()
	if err != nil {
		t.Error(err)
		return
	}
	if ki.SubKeyCount != 1 {
		t.Error("key must have 1 subkey")
	}
	if ki.MaxSubKeyLen != 6 {
		t.Error("key max subkey name length must be 6")
	}
	if ki.ValueCount != 24 {
		t.Errorf("key must have 24 values, but is %d", ki.ValueCount)
	}
	if ki.MaxValueNameLen != 12 {
		t.Errorf("key max value name length must be 10, but is %d", ki.MaxValueNameLen)
	}
	if ki.MaxValueLen != 38 {
		t.Errorf("key max value length must be 38, but is %d", ki.MaxValueLen)
	}
}

func deleteValues(t *testing.T, k registry.Key) {
	for _, test := range ValueTests {
		if test.WillFail {
			continue
		}
		err := k.DeleteValue(test.Name)
		if err != nil {
			t.Error(err)
			continue
		}
	}
	names, err := k.ReadValueNames()
	if err != nil {
		t.Error(err)
		return
	}
	if len(names) != 0 {
		t.Errorf("some values remain after deletion: %v", names)
	}
}

func TestValues(t *testing.T) {
	softwareK, err := registry.OpenKey(registry.CURRENT_USER, "Software", registry.QUERY_VALUE)
	if err != nil {
		t.Fatal(err)
	}
	defer softwareK.Close()

	testKName := randKeyName("TestValues_")

	k, exist, err := registry.CreateKey(softwareK, testKName, registry.CREATE_SUB_KEY|registry.QUERY_VALUE|registry.SET_VALUE)
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()

	if exist {
		t.Fatalf("key %q already exists", testKName)
	}

	defer registry.DeleteKey(softwareK, testKName)

	setValues(t, k)

	enumerateValues(t, k)

	testValues(t, k)

	testStat(t, k)

	deleteValues(t, k)
}

func TestExpandString(t *testing.T) {
	got, err := registry.ExpandString("%PATH%")
	if err != nil {
		t.Fatal(err)
	}
	want := os.Getenv("PATH")
	if got != want {
		t.Errorf("want %q string expanded, got %q", want, got)
	}
}

func TestInvalidValues(t *testing.T) {
	softwareK, err := registry.OpenKey(registry.CURRENT_USER, "Software", registry.QUERY_VALUE)
	if err != nil {
		t.Fatal(err)
	}
	defer softwareK.Close()

	testKName := randKeyName("TestInvalidValues_")

	k, exist, err := registry.CreateKey(softwareK, testKName, registry.CREATE_SUB_KEY|registry.QUERY_VALUE|registry.SET_VALUE)
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()

	if exist {
		t.Fatalf("key %q already exists", testKName)
	}

	defer registry.DeleteKey(softwareK, testKName)

	var tests = []struct {
		Type uint32
		Name string
		Data []byte
	}{
		{registry.DWORD, "Dword1", nil},
		{registry.DWORD, "Dword2", []byte{1, 2, 3}},
		{registry.QWORD, "Qword1", nil},
		{registry.QWORD, "Qword2", []byte{1, 2, 3}},
		{registry.QWORD, "Qword3", []byte{1, 2, 3, 4, 5, 6, 7}},
		{registry.MULTI_SZ, "MultiString1", nil},
		{registry.MULTI_SZ, "MultiString2", []byte{0}},
		{registry.MULTI_SZ, "MultiString3", []byte{'a', 'b', 0}},
		{registry.MULTI_SZ, "MultiString4", []byte{'a', 0, 0, 'b', 0}},
		{registry.MULTI_SZ, "MultiString5", []byte{'a', 0, 0}},
	}

	for _, test := range tests {
		err := k.SetValue(test.Name, test.Type, test.Data)
		if err != nil {
			t.Fatalf("SetValue for %q failed: %v", test.Name, err)
		}
	}

	for _, test := range tests {
		switch test.Type {
		case registry.DWORD, registry.QWORD:
			value, valType, err := k.GetIntegerValue(test.Name)
			if err == nil {
				t.Errorf("GetIntegerValue(%q) succeeded. Returns type=%d value=%v", test.Name, valType, value)
			}
		case registry.MULTI_SZ:
			value, valType, err := k.GetStringsValue(test.Name)
			if err == nil {
				if len(value) != 0 {
					t.Errorf("GetStringsValue(%q) succeeded. Returns type=%d value=%v", test.Name, valType, value)
				}
			}
		default:
			t.Errorf("unsupported type %d for %s value", test.Type, test.Name)
		}
	}
}

func TestGetMUIStringValue(t *testing.T) {
	if err := registry.LoadRegLoadMUIString(); err != nil {
		t.Skip("regLoadMUIString not supported; skipping")
	}
	if err := procGetDynamicTimeZoneInformation.Find(); err != nil {
		t.Skipf("%s not supported; skipping", procGetDynamicTimeZoneInformation.Name)
	}
	var dtzi DynamicTimezoneinformation
	if _, err := GetDynamicTimeZoneInformation(&dtzi); err != nil {
		t.Fatal(err)
	}
	tzKeyName := syscall.UTF16ToString(dtzi.TimeZoneKeyName[:])
	timezoneK, err := registry.OpenKey(registry.LOCAL_MACHINE,
		`SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones\`+tzKeyName, registry.READ)
	if err != nil {
		t.Fatal(err)
	}
	defer timezoneK.Close()

	type testType struct {
		name string
		want string
	}
	var tests = []testType{
		{"MUI_Std", syscall.UTF16ToString(dtzi.StandardName[:])},
	}
	if dtzi.DynamicDaylightTimeDisabled == 0 {
		tests = append(tests, testType{"MUI_Dlt", syscall.UTF16ToString(dtzi.DaylightName[:])})
	}

	for _, test := range tests {
		got, err := timezoneK.GetMUIStringValue(test.name)
		if err != nil {
			t.Error("GetMUIStringValue:", err)
		}

		if got != test.want {
			t.Errorf("GetMUIStringValue: %s: Got %q, want %q", test.name, got, test.want)
		}
	}
}

type DynamicTimezoneinformation struct {
	Bias                        int32
	StandardName                [32]uint16
	StandardDate                syscall.Systemtime
	StandardBias                int32
	DaylightName                [32]uint16
	DaylightDate                syscall.Systemtime
	DaylightBias                int32
	TimeZoneKeyName             [128]uint16
	DynamicDaylightTimeDisabled uint8
}

var (
	kernel32DLL = syscall.NewLazyDLL("kernel32")

	procGetDynamicTimeZoneInformation = kernel32DLL.NewProc("GetDynamicTimeZoneInformation")
)

func GetDynamicTimeZoneInformation(dtzi *DynamicTimezoneinformation) (rc uint32, err error) {
	r0, _, e1 := syscall.Syscall(procGetDynamicTimeZoneInformation.Addr(), 1, uintptr(unsafe.Pointer(dtzi)), 0, 0)
	rc = uint32(r0)
	if rc == 0xffffffff {
		if e1 != 0 {
			err = error(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}
