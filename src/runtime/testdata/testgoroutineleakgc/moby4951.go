/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/4951
 * Buggy version: 81f148be566ab2b17810ad4be61a5d8beac8330f
 * fix commit-id: 2ffef1b7eb618162673c6ffabccb9ca57c7dfce3
 * Flaky: 100/100
 * Description:
 *   The root cause and patch is clearly explained in the commit
 * description. The global lock is devices.Lock(), and the device
 * lock is baseInfo.lock.Lock(). It is very likely that this bug
 * can be reproduced.
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
		// deadlocks: x > 0
		go ds.DeleteDevice("info1")
		// deadlocks: x > 0
		go ds.DeleteDevice("info1")
	}()
}
