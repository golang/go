// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe" // for go:cgo_export_static and go:cgo_export_dynamic

// Export the main function.
//
// Used by the app package to start all-Go Android apps that are
// loaded via JNI. See golang.org/x/mobile/app.

//go:cgo_export_static main.main
//go:cgo_export_dynamic main.main
