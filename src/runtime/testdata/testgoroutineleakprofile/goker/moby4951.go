// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/4951
 * Buggy version: 81f148be566ab2b17810ad4be61a5d8beac8330f
 * fix commit-id: 2ffef1b7eb618162673c6ffabccb9ca57c7dfce3
 * Flaky: 100/100
 */
package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Moby4951", Moby4951)
}

type DeviceSet_moby4951 struct {
	sync.Mutex
	infos            map[string]*DevInfo_moby4951
	nrDeletedDevices int
}

func (devices *DeviceSet_moby4951) DeleteDevice(hash string) {
	devices.Lock()
	defer devices.Unlock()

	info := devices.lookupDevice(hash)

	info.lock.Lock()
	runtime.Gosched()
	defer info.lock.Unlock()

	devices.deleteDevice(info)
}

func (devices *DeviceSet_moby4951) lookupDevice(hash string) *DevInfo_moby4951 {
	existing, ok := devices.infos[hash]
	if !ok {
		return nil
	}
	return existing
}

func (devices *DeviceSet_moby4951) deleteDevice(info *DevInfo_moby4951) {
	devices.removeDeviceAndWait(info.Name())
}

func (devices *DeviceSet_moby4951) removeDeviceAndWait(devname string) {
	/// remove devices by devname
	devices.Unlock()
	time.Sleep(300 * time.Nanosecond)
	devices.Lock()
}

type DevInfo_moby4951 struct {
	lock sync.Mutex
	name string
}

func (info *DevInfo_moby4951) Name() string {
	return info.name
}

func NewDeviceSet_moby4951() *DeviceSet_moby4951 {
	devices := &DeviceSet_moby4951{
		infos: make(map[string]*DevInfo_moby4951),
	}
	info1 := &DevInfo_moby4951{
		name: "info1",
	}
	info2 := &DevInfo_moby4951{
		name: "info2",
	}
	devices.infos[info1.name] = info1
	devices.infos[info2.name] = info2
	return devices
}

func Moby4951() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	go func() {
		ds := NewDeviceSet_moby4951()
		/// Delete devices by the same info
		go ds.DeleteDevice("info1")
		go ds.DeleteDevice("info1")
	}()
}
