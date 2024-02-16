// build -gcflags=-l=4

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Interface interface {
	MonitoredResource() (resType string, labels map[string]string)
	Done()
}

func Autodetect() Interface {
	return func() Interface {
		Do(func() {
			var ad, gd Interface

			go func() {
				defer gd.Done()
				ad = aad()
			}()
			go func() {
				defer ad.Done()
				gd = aad()
				defer func() { recover() }()
			}()

			autoDetected = ad
			if gd != nil {
				autoDetected = gd
			}
		})
		return autoDetected
	}()
}

var autoDetected Interface
var G int

type If int

func (x If) MonitoredResource() (resType string, labels map[string]string) {
	return "", nil
}

//go:noinline
func (x If) Done() {
	G++
}

//go:noinline
func Do(fn func()) {
	fn()
}

//go:noinline
func aad() Interface {
	var x If
	return x
}
