// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs defs_freebsd.go

package route

const (
	sysAF_UNSPEC = 0x0
	sysAF_INET   = 0x2
	sysAF_ROUTE  = 0x11
	sysAF_LINK   = 0x12
	sysAF_INET6  = 0x1c

	sysSOCK_RAW = 0x3

	sysNET_RT_DUMP     = 0x1
	sysNET_RT_FLAGS    = 0x2
	sysNET_RT_IFLIST   = 0x3
	sysNET_RT_IFMALIST = 0x4
	sysNET_RT_IFLISTL  = 0x5
)

const (
	sysCTL_MAXNAME = 0x18

	sysCTL_UNSPEC   = 0x0
	sysCTL_KERN     = 0x1
	sysCTL_VM       = 0x2
	sysCTL_VFS      = 0x3
	sysCTL_NET      = 0x4
	sysCTL_DEBUG    = 0x5
	sysCTL_HW       = 0x6
	sysCTL_MACHDEP  = 0x7
	sysCTL_USER     = 0x8
	sysCTL_P1003_1B = 0x9
)

const (
	sysRTM_VERSION = 0x5

	sysRTM_ADD        = 0x1
	sysRTM_DELETE     = 0x2
	sysRTM_CHANGE     = 0x3
	sysRTM_GET        = 0x4
	sysRTM_LOSING     = 0x5
	sysRTM_REDIRECT   = 0x6
	sysRTM_MISS       = 0x7
	sysRTM_LOCK       = 0x8
	sysRTM_RESOLVE    = 0xb
	sysRTM_NEWADDR    = 0xc
	sysRTM_DELADDR    = 0xd
	sysRTM_IFINFO     = 0xe
	sysRTM_NEWMADDR   = 0xf
	sysRTM_DELMADDR   = 0x10
	sysRTM_IFANNOUNCE = 0x11
	sysRTM_IEEE80211  = 0x12

	sysRTA_DST     = 0x1
	sysRTA_GATEWAY = 0x2
	sysRTA_NETMASK = 0x4
	sysRTA_GENMASK = 0x8
	sysRTA_IFP     = 0x10
	sysRTA_IFA     = 0x20
	sysRTA_AUTHOR  = 0x40
	sysRTA_BRD     = 0x80

	sysRTAX_DST     = 0x0
	sysRTAX_GATEWAY = 0x1
	sysRTAX_NETMASK = 0x2
	sysRTAX_GENMASK = 0x3
	sysRTAX_IFP     = 0x4
	sysRTAX_IFA     = 0x5
	sysRTAX_AUTHOR  = 0x6
	sysRTAX_BRD     = 0x7
	sysRTAX_MAX     = 0x8
)

const (
	sizeofIfMsghdrlFreeBSD10        = 0x68
	sizeofIfaMsghdrFreeBSD10        = 0x14
	sizeofIfaMsghdrlFreeBSD10       = 0x6c
	sizeofIfmaMsghdrFreeBSD10       = 0x10
	sizeofIfAnnouncemsghdrFreeBSD10 = 0x18

	sizeofRtMsghdrFreeBSD10  = 0x5c
	sizeofRtMetricsFreeBSD10 = 0x38

	sizeofIfMsghdrFreeBSD7  = 0x70
	sizeofIfMsghdrFreeBSD8  = 0x70
	sizeofIfMsghdrFreeBSD9  = 0x70
	sizeofIfMsghdrFreeBSD10 = 0x70
	sizeofIfMsghdrFreeBSD11 = 0xa8

	sizeofIfDataFreeBSD7  = 0x60
	sizeofIfDataFreeBSD8  = 0x60
	sizeofIfDataFreeBSD9  = 0x60
	sizeofIfDataFreeBSD10 = 0x60
	sizeofIfDataFreeBSD11 = 0x98

	sizeofIfMsghdrlFreeBSD10Emu        = 0x68
	sizeofIfaMsghdrFreeBSD10Emu        = 0x14
	sizeofIfaMsghdrlFreeBSD10Emu       = 0x6c
	sizeofIfmaMsghdrFreeBSD10Emu       = 0x10
	sizeofIfAnnouncemsghdrFreeBSD10Emu = 0x18

	sizeofRtMsghdrFreeBSD10Emu  = 0x5c
	sizeofRtMetricsFreeBSD10Emu = 0x38

	sizeofIfMsghdrFreeBSD7Emu  = 0x70
	sizeofIfMsghdrFreeBSD8Emu  = 0x70
	sizeofIfMsghdrFreeBSD9Emu  = 0x70
	sizeofIfMsghdrFreeBSD10Emu = 0x70
	sizeofIfMsghdrFreeBSD11Emu = 0xa8

	sizeofIfDataFreeBSD7Emu  = 0x60
	sizeofIfDataFreeBSD8Emu  = 0x60
	sizeofIfDataFreeBSD9Emu  = 0x60
	sizeofIfDataFreeBSD10Emu = 0x60
	sizeofIfDataFreeBSD11Emu = 0x98

	sizeofSockaddrStorage = 0x80
	sizeofSockaddrInet    = 0x10
	sizeofSockaddrInet6   = 0x1c
)
