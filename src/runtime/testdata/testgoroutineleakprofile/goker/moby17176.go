// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/17176
 * Buggy version: d295dc66521e2734390473ec1f1da8a73ad3288a
 * fix commit-id: 2f16895ee94848e2d8ad72bc01968b4c88d84cb8
 * Flaky: 100/100
 */
package main

import (
	"errors"
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Moby17176", Moby17176)
}

type DeviceSet_moby17176 struct {
	sync.Mutex
	nrDeletedDevices int
}

func (devices *DeviceSet_moby17176) cleanupDeletedDevices() error {
	devices.Lock()
	if devices.nrDeletedDevices == 0 {
		/// Missing devices.Unlock()
		return nil
	}
	devices.Unlock()
	return errors.New("Error")
}

func testDevmapperLockReleasedDeviceDeletion_moby17176() {
	ds := &DeviceSet_moby17176{
		nrDeletedDevices: 0,
	}
	ds.cleanupDeletedDevices()
	doneChan := make(chan bool)
	go func() {
		ds.Lock()
		defer ds.Unlock()
		doneChan <- true
	}()

	select {
	case <-time.After(time.Millisecond):
	case <-doneChan:
	}
}
func Moby17176() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go testDevmapperLockReleasedDeviceDeletion_moby17176()
	}
}
