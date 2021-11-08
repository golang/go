// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Constants that were deprecated or moved to enums in the FreeBSD headers. Keep
// them here for backwards compatibility.

package unix

const (
	DLT_HHDLC            = 0x79
	IPV6_MIN_MEMBERSHIPS = 0x1f
	IP_MAX_SOURCE_FILTER = 0x400
	IP_MIN_MEMBERSHIPS   = 0x1f
	RT_CACHING_CONTEXT   = 0x1
	RT_NORTREF           = 0x2
)
