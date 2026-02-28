// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

func ForceAusFromTZIForTesting() {
	ResetLocalOnceForTest()
	localOnce.Do(func() { initLocalFromTZI(&aus) })
}

func ForceUSPacificFromTZIForTesting() {
	ResetLocalOnceForTest()
	localOnce.Do(func() { initLocalFromTZI(&usPacific) })
}

func ToEnglishName(stdname, dstname string) (string, error) {
	return toEnglishName(stdname, dstname)
}
