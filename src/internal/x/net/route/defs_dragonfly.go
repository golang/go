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

	sysNET_RT_DUMP   = C.NET_RT_DUMP
	sysNET_RT_FLAGS  = C.NET_RT_FLAGS
	sysNET_RT_IFLIST = C.NET_RT_IFLIST
	sysNET_RT_MAXID  = C.NET_RT_MAXID
)

const (
	sysCTL_MAXNAME = C.CTL_MAXNAME

	sysCTL_UNSPEC   = C.CTL_UNSPEC
	sysCTL_KERN     = C.CTL_KERN
	sysCTL_VM       = C.CTL_VM
	sysCTL_VFS      = C.CTL_VFS
	sysCTL_NET      = C.CTL_NET
	sysCTL_DEBUG    = C.CTL_DEBUG
	sysCTL_HW       = C.CTL_HW
	sysCTL_MACHDEP  = C.CTL_MACHDEP
	sysCTL_USER     = C.CTL_USER
	sysCTL_P1003_1B = C.CTL_P1003_1B
	sysCTL_LWKT     = C.CTL_LWKT
	sysCTL_MAXID    = C.CTL_MAXID
)

const (
	sysRTM_VERSION = C.RTM_VERSION

	sysRTM_ADD        = C.RTM_ADD
	sysRTM_DELETE     = C.RTM_DELETE
	sysRTM_CHANGE     = C.RTM_CHANGE
	sysRTM_GET        = C.RTM_GET
	sysRTM_LOSING     = C.RTM_LOSING
	sysRTM_REDIRECT   = C.RTM_REDIRECT
	sysRTM_MISS       = C.RTM_MISS
	sysRTM_LOCK       = C.RTM_LOCK
	sysRTM_OLDADD     = C.RTM_OLDADD
	sysRTM_OLDDEL     = C.RTM_OLDDEL
	sysRTM_RESOLVE    = C.RTM_RESOLVE
	sysRTM_NEWADDR    = C.RTM_NEWADDR
	sysRTM_DELADDR    = C.RTM_DELADDR
	sysRTM_IFINFO     = C.RTM_IFINFO
	sysRTM_NEWMADDR   = C.RTM_NEWMADDR
	sysRTM_DELMADDR   = C.RTM_DELMADDR
	sysRTM_IFANNOUNCE = C.RTM_IFANNOUNCE
	sysRTM_IEEE80211  = C.RTM_IEEE80211

	sysRTA_DST     = C.RTA_DST
	sysRTA_GATEWAY = C.RTA_GATEWAY
	sysRTA_NETMASK = C.RTA_NETMASK
	sysRTA_GENMASK = C.RTA_GENMASK
	sysRTA_IFP     = C.RTA_IFP
	sysRTA_IFA     = C.RTA_IFA
	sysRTA_AUTHOR  = C.RTA_AUTHOR
	sysRTA_BRD     = C.RTA_BRD
	sysRTA_MPLS1   = C.RTA_MPLS1
	sysRTA_MPLS2   = C.RTA_MPLS2
	sysRTA_MPLS3   = C.RTA_MPLS3

	sysRTAX_DST     = C.RTAX_DST
	sysRTAX_GATEWAY = C.RTAX_GATEWAY
	sysRTAX_NETMASK = C.RTAX_NETMASK
	sysRTAX_GENMASK = C.RTAX_GENMASK
	sysRTAX_IFP     = C.RTAX_IFP
	sysRTAX_IFA     = C.RTAX_IFA
	sysRTAX_AUTHOR  = C.RTAX_AUTHOR
	sysRTAX_BRD     = C.RTAX_BRD
	sysRTAX_MPLS1   = C.RTAX_MPLS1
	sysRTAX_MPLS2   = C.RTAX_MPLS2
	sysRTAX_MPLS3   = C.RTAX_MPLS3
	sysRTAX_MAX     = C.RTAX_MAX
)

const (
	sizeofIfMsghdrDragonFlyBSD4         = C.sizeof_struct_if_msghdr
	sizeofIfaMsghdrDragonFlyBSD4        = C.sizeof_struct_ifa_msghdr
	sizeofIfmaMsghdrDragonFlyBSD4       = C.sizeof_struct_ifma_msghdr
	sizeofIfAnnouncemsghdrDragonFlyBSD4 = C.sizeof_struct_if_announcemsghdr

	sizeofRtMsghdrDragonFlyBSD4  = C.sizeof_struct_rt_msghdr
	sizeofRtMetricsDragonFlyBSD4 = C.sizeof_struct_rt_metrics

	sizeofSockaddrStorage = C.sizeof_struct_sockaddr_storage
	sizeofSockaddrInet    = C.sizeof_struct_sockaddr_in
	sizeofSockaddrInet6   = C.sizeof_struct_sockaddr_in6
)
