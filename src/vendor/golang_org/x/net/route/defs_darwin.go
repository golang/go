// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package route

/*
#include <sys/socket.h>
#include <sys/sysctl.h>

#include <net/if.h>
#include <net/if_dl.h>
#include <net/route.h>

#include <netinet/in.h>
*/
import "C"

const (
	sysAF_UNSPEC = C.AF_UNSPEC
	sysAF_INET   = C.AF_INET
	sysAF_ROUTE  = C.AF_ROUTE
	sysAF_LINK   = C.AF_LINK
	sysAF_INET6  = C.AF_INET6

	sysSOCK_RAW = C.SOCK_RAW

	sysNET_RT_DUMP    = C.NET_RT_DUMP
	sysNET_RT_FLAGS   = C.NET_RT_FLAGS
	sysNET_RT_IFLIST  = C.NET_RT_IFLIST
	sysNET_RT_STAT    = C.NET_RT_STAT
	sysNET_RT_TRASH   = C.NET_RT_TRASH
	sysNET_RT_IFLIST2 = C.NET_RT_IFLIST2
	sysNET_RT_DUMP2   = C.NET_RT_DUMP2
	sysNET_RT_MAXID   = C.NET_RT_MAXID
)

const (
	sysCTL_MAXNAME = C.CTL_MAXNAME

	sysCTL_UNSPEC  = C.CTL_UNSPEC
	sysCTL_KERN    = C.CTL_KERN
	sysCTL_VM      = C.CTL_VM
	sysCTL_VFS     = C.CTL_VFS
	sysCTL_NET     = C.CTL_NET
	sysCTL_DEBUG   = C.CTL_DEBUG
	sysCTL_HW      = C.CTL_HW
	sysCTL_MACHDEP = C.CTL_MACHDEP
	sysCTL_USER    = C.CTL_USER
	sysCTL_MAXID   = C.CTL_MAXID
)

const (
	sysRTM_VERSION = C.RTM_VERSION

	sysRTM_ADD       = C.RTM_ADD
	sysRTM_DELETE    = C.RTM_DELETE
	sysRTM_CHANGE    = C.RTM_CHANGE
	sysRTM_GET       = C.RTM_GET
	sysRTM_LOSING    = C.RTM_LOSING
	sysRTM_REDIRECT  = C.RTM_REDIRECT
	sysRTM_MISS      = C.RTM_MISS
	sysRTM_LOCK      = C.RTM_LOCK
	sysRTM_OLDADD    = C.RTM_OLDADD
	sysRTM_OLDDEL    = C.RTM_OLDDEL
	sysRTM_RESOLVE   = C.RTM_RESOLVE
	sysRTM_NEWADDR   = C.RTM_NEWADDR
	sysRTM_DELADDR   = C.RTM_DELADDR
	sysRTM_IFINFO    = C.RTM_IFINFO
	sysRTM_NEWMADDR  = C.RTM_NEWMADDR
	sysRTM_DELMADDR  = C.RTM_DELMADDR
	sysRTM_IFINFO2   = C.RTM_IFINFO2
	sysRTM_NEWMADDR2 = C.RTM_NEWMADDR2
	sysRTM_GET2      = C.RTM_GET2

	sysRTA_DST     = C.RTA_DST
	sysRTA_GATEWAY = C.RTA_GATEWAY
	sysRTA_NETMASK = C.RTA_NETMASK
	sysRTA_GENMASK = C.RTA_GENMASK
	sysRTA_IFP     = C.RTA_IFP
	sysRTA_IFA     = C.RTA_IFA
	sysRTA_AUTHOR  = C.RTA_AUTHOR
	sysRTA_BRD     = C.RTA_BRD

	sysRTAX_DST     = C.RTAX_DST
	sysRTAX_GATEWAY = C.RTAX_GATEWAY
	sysRTAX_NETMASK = C.RTAX_NETMASK
	sysRTAX_GENMASK = C.RTAX_GENMASK
	sysRTAX_IFP     = C.RTAX_IFP
	sysRTAX_IFA     = C.RTAX_IFA
	sysRTAX_AUTHOR  = C.RTAX_AUTHOR
	sysRTAX_BRD     = C.RTAX_BRD
	sysRTAX_MAX     = C.RTAX_MAX
)

const (
	sizeofIfMsghdrDarwin15    = C.sizeof_struct_if_msghdr
	sizeofIfaMsghdrDarwin15   = C.sizeof_struct_ifa_msghdr
	sizeofIfmaMsghdrDarwin15  = C.sizeof_struct_ifma_msghdr
	sizeofIfMsghdr2Darwin15   = C.sizeof_struct_if_msghdr2
	sizeofIfmaMsghdr2Darwin15 = C.sizeof_struct_ifma_msghdr2
	sizeofIfDataDarwin15      = C.sizeof_struct_if_data
	sizeofIfData64Darwin15    = C.sizeof_struct_if_data64

	sizeofRtMsghdrDarwin15  = C.sizeof_struct_rt_msghdr
	sizeofRtMsghdr2Darwin15 = C.sizeof_struct_rt_msghdr2
	sizeofRtMetricsDarwin15 = C.sizeof_struct_rt_metrics

	sizeofSockaddrStorage = C.sizeof_struct_sockaddr_storage
	sizeofSockaddrInet    = C.sizeof_struct_sockaddr_in
	sizeofSockaddrInet6   = C.sizeof_struct_sockaddr_in6
)
