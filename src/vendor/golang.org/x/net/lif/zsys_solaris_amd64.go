// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs defs_solaris.go

package lif

const (
	sysAF_UNSPEC = 0x0
	sysAF_INET   = 0x2
	sysAF_INET6  = 0x1a

	sysSOCK_DGRAM = 0x1
)

type sockaddrStorage struct {
	Family     uint16
	X_ss_pad1  [6]int8
	X_ss_align float64
	X_ss_pad2  [240]int8
}

const (
	sysLIFC_NOXMIT          = 0x1
	sysLIFC_EXTERNAL_SOURCE = 0x2
	sysLIFC_TEMPORARY       = 0x4
	sysLIFC_ALLZONES        = 0x8
	sysLIFC_UNDER_IPMP      = 0x10
	sysLIFC_ENABLED         = 0x20

	sysSIOCGLIFADDR    = -0x3f87968f
	sysSIOCGLIFDSTADDR = -0x3f87968d
	sysSIOCGLIFFLAGS   = -0x3f87968b
	sysSIOCGLIFMTU     = -0x3f879686
	sysSIOCGLIFNETMASK = -0x3f879683
	sysSIOCGLIFMETRIC  = -0x3f879681
	sysSIOCGLIFNUM     = -0x3ff3967e
	sysSIOCGLIFINDEX   = -0x3f87967b
	sysSIOCGLIFSUBNET  = -0x3f879676
	sysSIOCGLIFLNKINFO = -0x3f879674
	sysSIOCGLIFCONF    = -0x3fef965b
	sysSIOCGLIFHWADDR  = -0x3f879640
)

const (
	sysIFF_UP          = 0x1
	sysIFF_BROADCAST   = 0x2
	sysIFF_DEBUG       = 0x4
	sysIFF_LOOPBACK    = 0x8
	sysIFF_POINTOPOINT = 0x10
	sysIFF_NOTRAILERS  = 0x20
	sysIFF_RUNNING     = 0x40
	sysIFF_NOARP       = 0x80
	sysIFF_PROMISC     = 0x100
	sysIFF_ALLMULTI    = 0x200
	sysIFF_INTELLIGENT = 0x400
	sysIFF_MULTICAST   = 0x800
	sysIFF_MULTI_BCAST = 0x1000
	sysIFF_UNNUMBERED  = 0x2000
	sysIFF_PRIVATE     = 0x8000
)

const (
	sizeofLifnum       = 0xc
	sizeofLifreq       = 0x178
	sizeofLifconf      = 0x18
	sizeofLifIfinfoReq = 0x10
)

type lifnum struct {
	Family    uint16
	Pad_cgo_0 [2]byte
	Flags     int32
	Count     int32
}

type lifreq struct {
	Name   [32]int8
	Lifru1 [4]byte
	Type   uint32
	Lifru  [336]byte
}

type lifconf struct {
	Family    uint16
	Pad_cgo_0 [2]byte
	Flags     int32
	Len       int32
	Pad_cgo_1 [4]byte
	Lifcu     [8]byte
}

type lifIfinfoReq struct {
	Maxhops      uint8
	Pad_cgo_0    [3]byte
	Reachtime    uint32
	Reachretrans uint32
	Maxmtu       uint32
}

const (
	sysIFT_IPV4 = 0xc8
	sysIFT_IPV6 = 0xc9
	sysIFT_6TO4 = 0xca
)
