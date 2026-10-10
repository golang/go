// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1 || windows

package signal

import (
	"os"
	"reflect"
	"syscall"
)

// Defined by the runtime package.
func signal_disable(uint32)
func signal_enable(uint32)
func signal_ignore(uint32)
func signal_ignored(uint32) bool
func signal_recv() uint32

func loop() {
	for {
		process(syscall.Signal(signal_recv()))
	}
}

func init() {
	watchSignalLoop = loop
}

const (
	numSig = 65 // max across all systems
)

func signum(sig os.Signal) int {
	switch sig := sig.(type) {
	case syscall.Signal:
		i := int(sig)
		if i < 0 || i >= numSig {
			return -1
		}
		return i
	default:
		// Use reflection to determine if sig has an underlying integer type.
		// In some systems like Windows, os.Signal may have implementations
		// with underlying integer types that are not directly accessible,
		// as they might be defined in external packages like golang.org/x/sys.
		// Since we cannot import those platform-specific signal types,
		// reflection allows us to handle them in a generic way.
		v := reflect.ValueOf(sig)
		switch v.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			// Extract the integer value from sig and validate it.
			i := int(v.Int())
			if i < 0 || i >= numSig {
				return -1
			}
			return i
		default:
			return -1
		}
	}
}

func enableSignal(sig int) {
	signal_enable(uint32(sig))
}

func disableSignal(sig int) {
	signal_disable(uint32(sig))
}

func ignoreSignal(sig int) {
	signal_ignore(uint32(sig))
}

func signalIgnored(sig int) bool {
	return signal_ignored(uint32(sig))
}
