// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

func Heapsort(data Interface) {
	heapSort(data, 0, data.Len())
}
