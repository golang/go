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
	sysCTL_DDB      = C.CTL_DDB
	sysCTL_PROC     = C.CTL_PROC
	sysCTL_VENDOR   = C.CTL_VENDOR
	sysCTL_EMUL     = C.CTL_EMUL
	sysCTL_SECURITY = C.CTL_SECURITY
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
	sysRTM_IFANNOUNCE = C.RTM_IFANNOUNCE
	sysRTM_IEEE80211  = C.RTM_IEEE80211
	sysRTM_SETGATE    = C.RTM_SETGATE
	sysRTM_LLINFO_UPD = C.RTM_LLINFO_UPD
	sysRTM_IFINFO     = C.RTM_IFINFO
	sysRTM_CHGADDR    = C.RTM_CHGADDR

	sysRTA_DST     = C.RTA_DST
	sysRTA_GATEWAY = C.RTA_GATEWAY
	sysRTA_NETMASK = C.RTA_NETMASK
	sysRTA_GENMASK = C.RTA_GENMASK
	sysRTA_IFP     = C.RTA_IFP
	sysRTA_IFA     = C.RTA_IFA
	sysRTA_AUTHOR  = C.RTA_AUTHOR
	sysRTA_BRD     = C.RTA_BRD
	sysRTA_TAG     = C.RTA_TAG

	sysRTAX_DST     = C.RTAX_DST
	sysRTAX_GATEWAY = C.RTAX_GATEWAY
	sysRTAX_NETMASK = C.RTAX_NETMASK
	sysRTAX_GENMASK = C.RTAX_GENMASK
	sysRTAX_IFP     = C.RTAX_IFP
	sysRTAX_IFA     = C.RTAX_IFA
	sysRTAX_AUTHOR  = C.RTAX_AUTHOR
	sysRTAX_BRD     = C.RTAX_BRD
	sysRTAX_TAG     = C.RTAX_TAG
	sysRTAX_MAX     = C.RTAX_MAX
)

const (
	sizeofIfMsghdrNetBSD7         = C.sizeof_struct_if_msghdr
	sizeofIfaMsghdrNetBSD7        = C.sizeof_struct_ifa_msghdr
	sizeofIfAnnouncemsghdrNetBSD7 = C.sizeof_struct_if_announcemsghdr

	sizeofRtMsghdrNetBSD7  = C.sizeof_struct_rt_msghdr
	sizeofRtMetricsNetBSD7 = C.sizeof_struct_rt_metrics

	sizeofSockaddrStorage = C.sizeof_struct_sockaddr_storage
	sizeofSockaddrInet    = C.sizeof_struct_sockaddr_in
	sizeofSockaddrInet6   = C.sizeof_struct_sockaddr_in6
)
