// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs defs_netbsd.go

package route

const (
	sysAF_UNSPEC = 0x0
	sysAF_INET   = 0x2
	sysAF_ROUTE  = 0x22
	sysAF_LINK   = 0x12
	sysAF_INET6  = 0x18

	sysNET_RT_DUMP   = 0x1
	sysNET_RT_FLAGS  = 0x2
	sysNET_RT_IFLIST = 0x5
	sysNET_RT_MAXID  = 0x6
)

const (
	sysCTL_MAXNAME = 0xc

	sysCTL_UNSPEC   = 0x0
	sysCTL_KERN     = 0x1
	sysCTL_VM       = 0x2
	sysCTL_VFS      = 0x3
	sysCTL_NET      = 0x4
	sysCTL_DEBUG    = 0x5
	sysCTL_HW       = 0x6
	sysCTL_MACHDEP  = 0x7
	sysCTL_USER     = 0x8
	sysCTL_DDB      = 0x9
	sysCTL_PROC     = 0xa
	sysCTL_VENDOR   = 0xb
	sysCTL_EMUL     = 0xc
	sysCTL_SECURITY = 0xd
	sysCTL_MAXID    = 0xe
)

const (
	sysRTM_VERSION = 0x4

	sysRTM_ADD        = 0x1
	sysRTM_DELETE     = 0x2
	sysRTM_CHANGE     = 0x3
	sysRTM_GET        = 0x4
	sysRTM_LOSING     = 0x5
	sysRTM_REDIRECT   = 0x6
	sysRTM_MISS       = 0x7
	sysRTM_LOCK       = 0x8
	sysRTM_OLDADD     = 0x9
	sysRTM_OLDDEL     = 0xa
	sysRTM_RESOLVE    = 0xb
	sysRTM_NEWADDR    = 0xc
	sysRTM_DELADDR    = 0xd
	sysRTM_IFANNOUNCE = 0x10
	sysRTM_IEEE80211  = 0x11
	sysRTM_SETGATE    = 0x12
	sysRTM_LLINFO_UPD = 0x13
	sysRTM_IFINFO     = 0x14
	sysRTM_CHGADDR    = 0x15

	sysRTA_DST     = 0x1
	sysRTA_GATEWAY = 0x2
	sysRTA_NETMASK = 0x4
	sysRTA_GENMASK = 0x8
	sysRTA_IFP     = 0x10
	sysRTA_IFA     = 0x20
	sysRTA_AUTHOR  = 0x40
	sysRTA_BRD     = 0x80
	sysRTA_TAG     = 0x100

	sysRTAX_DST     = 0x0
	sysRTAX_GATEWAY = 0x1
	sysRTAX_NETMASK = 0x2
	sysRTAX_GENMASK = 0x3
	sysRTAX_IFP     = 0x4
	sysRTAX_IFA     = 0x5
	sysRTAX_AUTHOR  = 0x6
	sysRTAX_BRD     = 0x7
	sysRTAX_TAG     = 0x8
	sysRTAX_MAX     = 0x9
)

const (
	sizeofIfMsghdrNetBSD7         = 0x98
	sizeofIfaMsghdrNetBSD7        = 0x18
	sizeofIfAnnouncemsghdrNetBSD7 = 0x18

	sizeofRtMsghdrNetBSD7  = 0x78
	sizeofRtMetricsNetBSD7 = 0x50
)
