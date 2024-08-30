// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"internal/syscall/windows/registry"
)

func init() {
	osInitMime = initMimeWindows
}

func initMimeWindows() {
	names, err := registry.CLASSES_ROOT.ReadSubKeyNames()
	if err != nil {
		return
	}
	for _, name := range names {
		if len(name) < 2 || name[0] != '.' { // looking for extensions only
			continue
		}
		k, err := registry.OpenKey(registry.CLASSES_ROOT, name, registry.READ)
		if err != nil {
			continue
		}
		v, _, err := k.GetStringValue("Content Type")
		k.Close()
		if err != nil {
			continue
		}

		// There is a long-standing problem on Windows: the
		// registry sometimes records that the ".js" extension
		// should be "text/plain". See issue #32350. While
		// normally local configuration should override
		// defaults, this problem is common enough that we
		// handle it here by ignoring that registry setting.
		if name == ".js" && (v == "text/plain" || v == "text/plain; charset=utf-8") {
			continue
		}

		setExtensionType(name, v)
	}
}

func initMimeForTests() map[string]string {
	return map[string]string{
		".PnG": "image/png",
	}
}
