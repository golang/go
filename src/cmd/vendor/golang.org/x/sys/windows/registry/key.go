// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

// Package registry provides access to the Windows registry.
//
// Here is a simple example, opening a registry key and reading a string value from it.
//
//	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\Windows NT\CurrentVersion`, registry.QUERY_VALUE)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer k.Close()
//
//	s, _, err := k.GetStringValue("SystemRoot")
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Printf("Windows system root is %q\n", s)
//
package registry

import (
	"io"
	"syscall"
	"time"
)

const (
	// Registry key security and access rights.
	// See https://msdn.microsoft.com/en-us/library/windows/desktop/ms724878.aspx
	// for details.
	ALL_ACCESS         = 0xf003f
	CREATE_LINK        = 0x00020
	CREATE_SUB_KEY     = 0x00004
	ENUMERATE_SUB_KEYS = 0x00008
	EXECUTE            = 0x20019
	NOTIFY             = 0x00010
	QUERY_VALUE        = 0x00001
	READ               = 0x20019
	SET_VALUE          = 0x00002
	WOW64_32KEY        = 0x00200
	WOW64_64KEY        = 0x00100
	WRITE              = 0x20006
)

// Key is a handle to an open Windows registry key.
// Keys can be obtained by calling OpenKey; there are
// also some predefined root keys such as CURRENT_USER.
// Keys can be used directly in the Windows API.
type Key syscall.Handle

const (
	// Windows defines some predefined root keys that are always open.
	// An application can use these keys as entry points to the registry.
	// Normally these keys are used in OpenKey to open new keys,
	// but they can also be used anywhere a Key is required.
	CLASSES_ROOT     = Key(syscall.HKEY_CLASSES_ROOT)
	CURRENT_USER     = Key(syscall.HKEY_CURRENT_USER)
	LOCAL_MACHINE    = Key(syscall.HKEY_LOCAL_MACHINE)
	USERS            = Key(syscall.HKEY_USERS)
	CURRENT_CONFIG   = Key(syscall.HKEY_CURRENT_CONFIG)
	PERFORMANCE_DATA = Key(syscall.HKEY_PERFORMANCE_DATA)
)

// Close closes open key k.
func (k Key) Close() error {
	return syscall.RegCloseKey(syscall.Handle(k))
}

// OpenKey opens a new key with path name relative to key k.
// It accepts any open key, including CURRENT_USER and others,
// and returns the new key and an error.
// The access parameter specifies desired access rights to the
// key to be opened.
func OpenKey(k Key, path string, access uint32) (Key, error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	var subkey syscall.Handle
	err = syscall.RegOpenKeyEx(syscall.Handle(k), p, 0, access, &subkey)
	if err != nil {
		return 0, err
	}
	return Key(subkey), nil
}

// OpenRemoteKey opens a predefined registry key on another
// computer pcname. The key to be opened is specified by k, but
// can only be one of LOCAL_MACHINE, PERFORMANCE_DATA or USERS.
// If pcname is "", OpenRemoteKey returns local computer key.
func OpenRemoteKey(pcname string, k Key) (Key, error) {
	var err error
	var p *uint16
	if pcname != "" {
		p, err = syscall.UTF16PtrFromString(`\\` + pcname)
		if err != nil {
			return 0, err
		}
	}
	var remoteKey syscall.Handle
	err = regConnectRegistry(p, syscall.Handle(k), &remoteKey)
	if err != nil {
		return 0, err
	}
	return Key(remoteKey), nil
}

// ReadSubKeyNames returns the names of subkeys of key k.
// The parameter n controls the number of returned names,
// analogous to the way os.File.Readdirnames works.
func (k Key) ReadSubKeyNames(n int) ([]string, error) {
	names := make([]string, 0)
	// Registry key size limit is 255 bytes and described there:
	// https://msdn.microsoft.com/library/windows/desktop/ms724872.aspx
	buf := make([]uint16, 256) //plus extra room for terminating zero byte
loopItems:
	for i := uint32(0); ; i++ {
		if n > 0 {
			if len(names) == n {
				return names, nil
			}
		}
		l := uint32(len(buf))
		for {
			err := syscall.RegEnumKeyEx(syscall.Handle(k), i, &buf[0], &l, nil, nil, nil, nil)
			if err == nil {
				break
			}
			if err == syscall.ERROR_MORE_DATA {
				// Double buffer size and try again.
				l = uint32(2 * len(buf))
				buf = make([]uint16, l)
				continue
			}
			if err == _ERROR_NO_MORE_ITEMS {
				break loopItems
			}
			return names, err
		}
		names = append(names, syscall.UTF16ToString(buf[:l]))
	}
	if n > len(names) {
		return names, io.EOF
	}
	return names, nil
}

// CreateKey creates a key named path under open key k.
// CreateKey returns the new key and a boolean flag that reports
// whether the key already existed.
// The access parameter specifies the access rights for the key
// to be created.
func CreateKey(k Key, path string, access uint32) (newk Key, openedExisting bool, err error) {
	var h syscall.Handle
	var d uint32
	err = regCreateKeyEx(syscall.Handle(k), syscall.StringToUTF16Ptr(path),
		0, nil, _REG_OPTION_NON_VOLATILE, access, nil, &h, &d)
	if err != nil {
		return 0, false, err
	}
	return Key(h), d == _REG_OPENED_EXISTING_KEY, nil
}

// DeleteKey deletes the subkey path of key k and its values.
func DeleteKey(k Key, path string) error {
	return regDeleteKey(syscall.Handle(k), syscall.StringToUTF16Ptr(path))
}

// A KeyInfo describes the statistics of a key. It is returned by Stat.
type KeyInfo struct {
	SubKeyCount     uint32
	MaxSubKeyLen    uint32 // size of the key's subkey with the longest name, in Unicode characters, not including the terminating zero byte
	ValueCount      uint32
	MaxValueNameLen uint32 // size of the key's longest value name, in Unicode characters, not including the terminating zero byte
	MaxValueLen     uint32 // longest data component among the key's values, in bytes
	lastWriteTime   syscall.Filetime
}

// ModTime returns the key's last write time.
func (ki *KeyInfo) ModTime() time.Time {
	return time.Unix(0, ki.lastWriteTime.Nanoseconds())
}

// Stat retrieves information about the open key k.
func (k Key) Stat() (*KeyInfo, error) {
	var ki KeyInfo
	err := syscall.RegQueryInfoKey(syscall.Handle(k), nil, nil, nil,
		&ki.SubKeyCount, &ki.MaxSubKeyLen, nil, &ki.ValueCount,
		&ki.MaxValueNameLen, &ki.MaxValueLen, nil, &ki.lastWriteTime)
	if err != nil {
		return nil, err
	}
	return &ki, nil
}
