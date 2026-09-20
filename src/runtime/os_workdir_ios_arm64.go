// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	maxPathLen             = 1024
	_kCFStringEncodingUTF8 = 0x08000100
)

// initWorkingDir sets the current working directory to the app root on iOS.
// By default ios/arm64 processes start in "/".
func initWorkingDir() {
	bundle := cfBundleGetMainBundle()
	if bundle == 0 {
		writeErrStr("runtime/cgo: no main bundle\n")
		return
	}
	if !bundleHasInfoPlist(bundle) {
		// Not an app bundle; keep the inherited working directory.
		return
	}
	url := cfBundleCopyBundleURL(bundle)
	if url == 0 {
		// No app bundle URL found.
		return
	}

	var buf [maxPathLen]byte
	path := &buf[0]
	ok := cfURLGetFileSystemRepresentation(url, true, path, uintptr(len(buf)))
	cfRelease(url)
	if !ok {
		writeErrStr("runtime/cgo: cannot get bundle URL path\n")
		return
	}

	if chdir(path) != 0 {
		writeErrStr("runtime/cgo: chdir(")
		writeErrData(path, int32(findnull(path)))
		writeErrStr(") failed\n")
	}

	const goExecWrapperWorkingDirectoryKey = "GoExecWrapperWorkingDirectory\x00"
	key := cfStringCreateWithCString(0, unsafe.StringData(goExecWrapperWorkingDirectoryKey), _kCFStringEncodingUTF8)
	if key == 0 {
		writeErrStr("runtime/cgo: cannot create GoExecWrapperWorkingDirectory string\n")
		return
	}
	wd := cfBundleGetValueForInfoDictionaryKey(bundle, key)
	cfRelease(key)
	if wd == 0 {
		return
	}
	if !cfStringGetCString(wd, path, uintptr(len(buf)), _kCFStringEncodingUTF8) {
		writeErrStr("runtime/cgo: cannot get GoExecWrapperWorkingDirectory string\n")
		return
	}

	if chdir(path) != 0 {
		writeErrStr("runtime/cgo: chdir(")
		writeErrData(path, int32(findnull(path)))
		writeErrStr(") failed\n")
	}
}

// bundleHasInfoPlist reports whether bundle contains an Info.plist. The main
// bundle of an executable that is not inside an app bundle is the directory
// holding the executable, which has no Info.plist. It can also happen on
// Corellium virtual devices.
func bundleHasInfoPlist(bundle uintptr) bool {
	const (
		infoName = "Info\x00"
		infoType = "plist\x00"
	)
	name := cfStringCreateWithCString(0, unsafe.StringData(infoName), _kCFStringEncodingUTF8)
	if name == 0 {
		writeErrStr("runtime/cgo: cannot create Info.plist strings\n")
		return false
	}
	typ := cfStringCreateWithCString(0, unsafe.StringData(infoType), _kCFStringEncodingUTF8)
	if typ == 0 {
		cfRelease(name)
		writeErrStr("runtime/cgo: cannot create Info.plist strings\n")
		return false
	}
	url := cfBundleCopyResourceURL(bundle, name, typ, 0)
	cfRelease(name)
	cfRelease(typ)
	if url == 0 {
		return false
	}
	cfRelease(url)
	return true
}
