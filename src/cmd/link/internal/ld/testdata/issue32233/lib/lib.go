// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lib

/*
#cgo darwin CFLAGS: -mmacosx-version-min=10.10 -D__MAC_OS_X_VERSION_MAX_ALLOWED=101450 -Wunguarded-availability-new
#cgo darwin LDFLAGS: -framework Foundation -framework UserNotifications
#include "stdlib.h"
int function(void);
*/
import "C"
import "fmt"

func DoC() {
	C.function()
	fmt.Println("called c function")
}
