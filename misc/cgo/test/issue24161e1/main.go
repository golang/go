// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package issue24161e1

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreFoundation -framework Security
#include <TargetConditionals.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#if TARGET_OS_IPHONE == 0 && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101200
  typedef CFStringRef SecKeyAlgorithm;
  static CFDataRef SecKeyCreateSignature(SecKeyRef key, SecKeyAlgorithm algorithm, CFDataRef dataToSign, CFErrorRef *error){return NULL;}
  #define kSecKeyAlgorithmECDSASignatureDigestX962SHA1 foo()
  static SecKeyAlgorithm foo(void){return NULL;}
#endif
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
