// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package eventlog

import (
	"errors"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

const (
	// Log levels.
	Info    = windows.EVENTLOG_INFORMATION_TYPE
	Warning = windows.EVENTLOG_WARNING_TYPE
	Error   = windows.EVENTLOG_ERROR_TYPE
)

const addKeyName = `SYSTEM\CurrentControlSet\Services\EventLog\Application`

// Install modifies PC registry to allow logging with an event source src.
// It adds all required keys and values to the event log registry key.
// Install uses msgFile as the event message file. If useExpandKey is true,
// the event message file is installed as REG_EXPAND_SZ value,
// otherwise as REG_SZ. Use bitwise of log.Error, log.Warning and
// log.Info to specify events supported by the new event source.
func Install(src, msgFile string, useExpandKey bool, eventsSupported uint32) error {
	appkey, err := registry.OpenKey(registry.LOCAL_MACHINE, addKeyName, registry.CREATE_SUB_KEY)
	if err != nil {
		return err
	}
	defer appkey.Close()

	sk, alreadyExist, err := registry.CreateKey(appkey, src, registry.SET_VALUE)
	if err != nil {
		return err
	}
	defer sk.Close()
	if alreadyExist {
		return errors.New(addKeyName + `\` + src + " registry key already exists")
	}

	err = sk.SetDWordValue("CustomSource", 1)
	if err != nil {
		return err
	}
	if useExpandKey {
		err = sk.SetExpandStringValue("EventMessageFile", msgFile)
	} else {
		err = sk.SetStringValue("EventMessageFile", msgFile)
	}
	if err != nil {
		return err
	}
	err = sk.SetDWordValue("TypesSupported", eventsSupported)
	if err != nil {
		return err
	}
	return nil
}

// InstallAsEventCreate is the same as Install, but uses
// %SystemRoot%\System32\EventCreate.exe as the event message file.
func InstallAsEventCreate(src string, eventsSupported uint32) error {
	return Install(src, "%SystemRoot%\\System32\\EventCreate.exe", true, eventsSupported)
}

// Remove deletes all registry elements installed by the correspondent Install.
func Remove(src string) error {
	appkey, err := registry.OpenKey(registry.LOCAL_MACHINE, addKeyName, registry.SET_VALUE)
	if err != nil {
		return err
	}
	defer appkey.Close()
	return registry.DeleteKey(appkey, src)
}
