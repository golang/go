// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

var NewProcThreadAttributeList = newProcThreadAttributeList
var UpdateProcThreadAttribute = updateProcThreadAttribute
var DeleteProcThreadAttributeList = deleteProcThreadAttributeList

const PROC_THREAD_ATTRIBUTE_HANDLE_LIST = _PROC_THREAD_ATTRIBUTE_HANDLE_LIST
