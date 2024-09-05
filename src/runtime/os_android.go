// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

//go:linkname androidVersion runtime.androidVersion
func androidVersion() int {
	const PROP_VALUE_MAX = 92
	var value [PROP_VALUE_MAX]byte
	name := []byte("ro.build.version.release\x00")
	length := __system_property_get(&name[0], &value[0])
	for i := int32(0); i < length; i++ {
		if value[i] < '0' || value[i] > '9' {
			length = i
			break
		}
	}
	version, _ := atoi(unsafe.String(&value[0], length))
	return version
}

// Export the main function.
//
// Used by the app package to start all-Go Android apps that are
// loaded via JNI. See golang.org/x/mobile/app.

//go:cgo_export_static main.main
//go:cgo_export_dynamic main.main
