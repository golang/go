// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package issue24161e1

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreFoundation -framework Security
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
*/
import "C"
import (
	"fmt"
	"testing"
)

func f1() {
	C.SecKeyCreateSignature(0, C.kSecKeyAlgorithmECDSASignatureDigestX962SHA1, 0, nil)
}

func f2(e C.CFErrorRef) {
	if desc := C.CFErrorCopyDescription(e); desc != 0 {
		fmt.Println(desc)
	}
}

func Test(t *testing.T) {}
