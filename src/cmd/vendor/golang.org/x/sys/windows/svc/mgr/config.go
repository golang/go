// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package mgr

import (
	"syscall"
	"unicode/utf16"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	// Service start types.
	StartManual    = windows.SERVICE_DEMAND_START // the service must be started manually
	StartAutomatic = windows.SERVICE_AUTO_START   // the service will start by itself whenever the computer reboots
	StartDisabled  = windows.SERVICE_DISABLED     // the service cannot be started

	// The severity of the error, and action taken,
	// if this service fails to start.
	ErrorCritical = windows.SERVICE_ERROR_CRITICAL
	ErrorIgnore   = windows.SERVICE_ERROR_IGNORE
	ErrorNormal   = windows.SERVICE_ERROR_NORMAL
	ErrorSevere   = windows.SERVICE_ERROR_SEVERE
)

// TODO(brainman): Password is not returned by windows.QueryServiceConfig, not sure how to get it.

type Config struct {
	ServiceType      uint32
	StartType        uint32
	ErrorControl     uint32
	BinaryPathName   string // fully qualified path to the service binary file, can also include arguments for an auto-start service
	LoadOrderGroup   string
	TagId            uint32
	Dependencies     []string
	ServiceStartName string // name of the account under which the service should run
	DisplayName      string
	Password         string
	Description      string
}

func toString(p *uint16) string {
	if p == nil {
		return ""
	}
	return syscall.UTF16ToString((*[4096]uint16)(unsafe.Pointer(p))[:])
}

func toStringSlice(ps *uint16) []string {
	if ps == nil {
		return nil
	}
	r := make([]string, 0)
	for from, i, p := 0, 0, (*[1 << 24]uint16)(unsafe.Pointer(ps)); true; i++ {
		if p[i] == 0 {
			// empty string marks the end
			if i <= from {
				break
			}
			r = append(r, string(utf16.Decode(p[from:i])))
			from = i + 1
		}
	}
	return r
}

// Config retrieves service s configuration paramteres.
func (s *Service) Config() (Config, error) {
	var p *windows.QUERY_SERVICE_CONFIG
	n := uint32(1024)
	for {
		b := make([]byte, n)
		p = (*windows.QUERY_SERVICE_CONFIG)(unsafe.Pointer(&b[0]))
		err := windows.QueryServiceConfig(s.Handle, p, n, &n)
		if err == nil {
			break
		}
		if err.(syscall.Errno) != syscall.ERROR_INSUFFICIENT_BUFFER {
			return Config{}, err
		}
		if n <= uint32(len(b)) {
			return Config{}, err
		}
	}

	var p2 *windows.SERVICE_DESCRIPTION
	n = uint32(1024)
	for {
		b := make([]byte, n)
		p2 = (*windows.SERVICE_DESCRIPTION)(unsafe.Pointer(&b[0]))
		err := windows.QueryServiceConfig2(s.Handle,
			windows.SERVICE_CONFIG_DESCRIPTION, &b[0], n, &n)
		if err == nil {
			break
		}
		if err.(syscall.Errno) != syscall.ERROR_INSUFFICIENT_BUFFER {
			return Config{}, err
		}
		if n <= uint32(len(b)) {
			return Config{}, err
		}
	}

	return Config{
		ServiceType:      p.ServiceType,
		StartType:        p.StartType,
		ErrorControl:     p.ErrorControl,
		BinaryPathName:   toString(p.BinaryPathName),
		LoadOrderGroup:   toString(p.LoadOrderGroup),
		TagId:            p.TagId,
		Dependencies:     toStringSlice(p.Dependencies),
		ServiceStartName: toString(p.ServiceStartName),
		DisplayName:      toString(p.DisplayName),
		Description:      toString(p2.Description),
	}, nil
}

func updateDescription(handle windows.Handle, desc string) error {
	d := windows.SERVICE_DESCRIPTION{Description: toPtr(desc)}
	return windows.ChangeServiceConfig2(handle,
		windows.SERVICE_CONFIG_DESCRIPTION, (*byte)(unsafe.Pointer(&d)))
}

// UpdateConfig updates service s configuration parameters.
func (s *Service) UpdateConfig(c Config) error {
	err := windows.ChangeServiceConfig(s.Handle, c.ServiceType, c.StartType,
		c.ErrorControl, toPtr(c.BinaryPathName), toPtr(c.LoadOrderGroup),
		nil, toStringBlock(c.Dependencies), toPtr(c.ServiceStartName),
		toPtr(c.Password), toPtr(c.DisplayName))
	if err != nil {
		return err
	}
	return updateDescription(s.Handle, c.Description)
}
