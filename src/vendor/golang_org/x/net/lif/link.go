// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build solaris

package lif

import "unsafe"

// A Link represents logical data link information.
//
// It also represents base information for logical network interface.
// On Solaris, each logical network interface represents network layer
// adjacency information and the interface has a only single network
// address or address pair for tunneling. It's usual that multiple
// logical network interfaces share the same logical data link.
type Link struct {
	Name  string // name, equivalent to IP interface name
	Index int    // index, equivalent to IP interface index
	Type  int    // type
	Flags int    // flags
	MTU   int    // maximum transmission unit, basically link MTU but may differ between IP address families
	Addr  []byte // address
}

func (ll *Link) fetch(s uintptr) {
	var lifr lifreq
	for i := 0; i < len(ll.Name); i++ {
		lifr.Name[i] = int8(ll.Name[i])
	}
	ioc := int64(sysSIOCGLIFINDEX)
	if err := ioctl(s, uintptr(ioc), unsafe.Pointer(&lifr)); err == nil {
		ll.Index = int(nativeEndian.Uint32(lifr.Lifru[:4]))
	}
	ioc = int64(sysSIOCGLIFFLAGS)
	if err := ioctl(s, uintptr(ioc), unsafe.Pointer(&lifr)); err == nil {
		ll.Flags = int(nativeEndian.Uint64(lifr.Lifru[:8]))
	}
	ioc = int64(sysSIOCGLIFMTU)
	if err := ioctl(s, uintptr(ioc), unsafe.Pointer(&lifr)); err == nil {
		ll.MTU = int(nativeEndian.Uint32(lifr.Lifru[:4]))
	}
	switch ll.Type {
	case sysIFT_IPV4, sysIFT_IPV6, sysIFT_6TO4:
	default:
		ioc = int64(sysSIOCGLIFHWADDR)
		if err := ioctl(s, uintptr(ioc), unsafe.Pointer(&lifr)); err == nil {
			ll.Addr, _ = parseLinkAddr(lifr.Lifru[4:])
		}
	}
}

// Links returns a list of logical data links.
//
// The provided af must be an address family and name must be a data
// link name. The zero value of af or name means a wildcard.
func Links(af int, name string) ([]Link, error) {
	eps, err := newEndpoints(af)
	if len(eps) == 0 {
		return nil, err
	}
	defer func() {
		for _, ep := range eps {
			ep.close()
		}
	}()
	return links(eps, name)
}

func links(eps []endpoint, name string) ([]Link, error) {
	var lls []Link
	lifn := sysLifnum{Flags: sysLIFC_NOXMIT | sysLIFC_TEMPORARY | sysLIFC_ALLZONES | sysLIFC_UNDER_IPMP}
	lifc := lifconf{Flags: sysLIFC_NOXMIT | sysLIFC_TEMPORARY | sysLIFC_ALLZONES | sysLIFC_UNDER_IPMP}
	for _, ep := range eps {
		lifn.Family = uint16(ep.af)
		ioc := int64(sysSIOCGLIFNUM)
		if err := ioctl(ep.s, uintptr(ioc), unsafe.Pointer(&lifn)); err != nil {
			continue
		}
		if lifn.Count == 0 {
			continue
		}
		b := make([]byte, lifn.Count*sizeofLifreq)
		lifc.Family = uint16(ep.af)
		lifc.Len = lifn.Count * sizeofLifreq
		if len(lifc.Lifcu) == 8 {
			nativeEndian.PutUint64(lifc.Lifcu[:], uint64(uintptr(unsafe.Pointer(&b[0]))))
		} else {
			nativeEndian.PutUint32(lifc.Lifcu[:], uint32(uintptr(unsafe.Pointer(&b[0]))))
		}
		ioc = int64(sysSIOCGLIFCONF)
		if err := ioctl(ep.s, uintptr(ioc), unsafe.Pointer(&lifc)); err != nil {
			continue
		}
		nb := make([]byte, 32) // see LIFNAMSIZ in net/if.h
		for i := 0; i < int(lifn.Count); i++ {
			lifr := (*lifreq)(unsafe.Pointer(&b[i*sizeofLifreq]))
			for i := 0; i < 32; i++ {
				if lifr.Name[i] == 0 {
					nb = nb[:i]
					break
				}
				nb[i] = byte(lifr.Name[i])
			}
			llname := string(nb)
			nb = nb[:32]
			if isDupLink(lls, llname) || name != "" && name != llname {
				continue
			}
			ll := Link{Name: llname, Type: int(lifr.Type)}
			ll.fetch(ep.s)
			lls = append(lls, ll)
		}
	}
	return lls, nil
}

func isDupLink(lls []Link, name string) bool {
	for _, ll := range lls {
		if ll.Name == name {
			return true
		}
	}
	return false
}
