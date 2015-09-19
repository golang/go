// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

// See https://www.freebsd.org/doc/en_US.ISO8859-1/books/porters-handbook/versions.html#freebsd-versions-table.
var freebsdVersion uint32

func init() {
	freebsdVersion, _ = SysctlUint32("kern.osreldate")
	conf, _ := Sysctl("kern.conftxt")
	for i, j := 0, 0; j < len(conf); j++ {
		if conf[j] != '\n' {
			continue
		}
		s := conf[i:j]
		i = j + 1
		if len(s) > len("machine") && s[:len("machine")] == "machine" {
			s = s[len("machine"):]
			for k := 0; k < len(s); k++ {
				if s[k] == ' ' || s[k] == '\t' {
					s = s[1:]
				}
				break
			}
			freebsdConfArch = s
			break
		}
	}
}

func (any *anyMessage) toRoutingMessage(b []byte) RoutingMessage {
	switch any.Type {
	case RTM_ADD, RTM_DELETE, RTM_CHANGE, RTM_GET, RTM_LOSING, RTM_REDIRECT, RTM_MISS, RTM_LOCK, RTM_RESOLVE:
		return any.parseRouteMessage(b)
	case RTM_IFINFO:
		return any.parseInterfaceMessage(b)
	case RTM_IFANNOUNCE:
		p := (*InterfaceAnnounceMessage)(unsafe.Pointer(any))
		return &InterfaceAnnounceMessage{Header: p.Header}
	case RTM_NEWADDR, RTM_DELADDR:
		return any.parseInterfaceAddrMessage(b)
	case RTM_NEWMADDR, RTM_DELMADDR:
		p := (*InterfaceMulticastAddrMessage)(unsafe.Pointer(any))
		return &InterfaceMulticastAddrMessage{Header: p.Header, Data: b[SizeofIfmaMsghdr:any.Msglen]}
	}
	return nil
}

func (any *anyMessage) parseInterfaceMessage(b []byte) *InterfaceMessage {
	p := (*ifMsghdrFixed)(unsafe.Pointer(any))
	h := IfMsghdr{
		Msglen:   p.Msglen,
		Version:  p.Version,
		Type:     p.Type,
		Addrs:    p.Addrs,
		Flags:    p.Flags,
		Index:    p.Index,
		Len:      p.Len,
		Data_off: p.Data_off,
	}

	switch {
	case freebsdVersion >= 1100011:
		// FreeBSD 11 uses a new struct if_data layout
		// See https://svnweb.freebsd.org/base?view=revision&revision=263102
		data11 := *(*ifData11Raw)(unsafe.Pointer(&b[p.Data_off:p.Len][0]))
		h.Data.copyFromV11Raw(&data11)
	case freebsdVersion >= 1001000:
		// FreeBSD 10.1 and newer
		data10 := *(*ifData10)(unsafe.Pointer(&b[p.Data_off:p.Len][0]))
		h.Data.copyFromV10(&data10)
	case freebsdVersion >= 903000:
		// TODO
	}

	return &InterfaceMessage{Header: h, Data: b[p.Len:any.Msglen]}
}

func (d *IfData) copyFromV11Raw(data11 *ifData11Raw) {
	d.Type = data11.Type
	d.Physical = data11.Physical
	d.Addrlen = data11.Addrlen
	d.Hdrlen = data11.Hdrlen
	d.Link_state = data11.Link_state
	d.Vhid = data11.Vhid
	d.Datalen = data11.Datalen
	d.Mtu = data11.Mtu
	d.Metric = data11.Metric
	d.Baudrate = data11.Baudrate
	d.Ipackets = data11.Ipackets
	d.Ierrors = data11.Ierrors
	d.Opackets = data11.Opackets
	d.Oerrors = data11.Oerrors
	d.Collisions = data11.Collisions
	d.Ibytes = data11.Ibytes
	d.Obytes = data11.Obytes
	d.Imcasts = data11.Imcasts
	d.Omcasts = data11.Omcasts
	d.Iqdrops = data11.Iqdrops
	d.Oqdrops = data11.Oqdrops
	d.Noproto = data11.Noproto
	d.Hwassist = data11.Hwassist

	d.fillEpochLastChange(data11)
}

func (d *IfData) copyFromV10(data10 *ifData10) {
	d.Type = data10.Type
	d.Physical = data10.Physical
	d.Addrlen = data10.Addrlen
	d.Hdrlen = data10.Hdrlen
	d.Link_state = data10.Link_state
	d.Vhid = data10.Vhid
	d.Datalen = uint16(data10.Datalen)
	d.Mtu = uint32(data10.Mtu)
	d.Metric = uint32(data10.Metric)
	d.Baudrate = uint64(data10.Baudrate)
	d.Ipackets = uint64(data10.Ipackets)
	d.Ierrors = uint64(data10.Ierrors)
	d.Opackets = uint64(data10.Opackets)
	d.Oerrors = uint64(data10.Oerrors)
	d.Collisions = uint64(data10.Collisions)
	d.Ibytes = uint64(data10.Ibytes)
	d.Obytes = uint64(data10.Obytes)
	d.Imcasts = uint64(data10.Imcasts)
	d.Omcasts = uint64(data10.Omcasts)
	d.Iqdrops = uint64(data10.Iqdrops)
	d.Oqdrops = uint64(data10.Oqdrops)
	d.Noproto = uint64(data10.Noproto)
	d.Hwassist = uint64(data10.Hwassist)

	d.Epoch = data10.Epoch
	d.Lastchange = data10.Lastchange
}

func (any *anyMessage) parseInterfaceAddrMessage(b []byte) *InterfaceAddrMessage {
	p := (*IfaMsghdr)(unsafe.Pointer(any))

	h := IfaMsghdr{
		Msglen:   p.Msglen,
		Version:  p.Version,
		Type:     p.Type,
		Addrs:    p.Addrs,
		Flags:    p.Flags,
		Index:    p.Index,
		Len:      p.Len,
		Data_off: p.Data_off,
		Metric:   p.Metric,
	}

	switch {
	case freebsdVersion >= 1100011:
		// FreeBSD 11 uses a new struct if_data layout
		// See https://svnweb.freebsd.org/base?view=revision&revision=263102
		data11 := *(*ifData11Raw)(unsafe.Pointer(&b[p.Data_off:p.Len][0]))
		h.Data.copyFromV11Raw(&data11)
	case freebsdVersion >= 1001000:
		// FreeBSD 10.1 and newer
		data10 := *(*ifData10)(unsafe.Pointer(&b[p.Data_off:p.Len][0]))
		h.Data.copyFromV10(&data10)
	case freebsdVersion >= 903000:
		// TODO
	}
	return &InterfaceAddrMessage{Header: h, Data: b[p.Len:any.Msglen]}
}

// InterfaceAnnounceMessage represents a routing message containing
// network interface arrival and departure information.
type InterfaceAnnounceMessage struct {
	Header IfAnnounceMsghdr
}

func (m *InterfaceAnnounceMessage) sockaddr() ([]Sockaddr, error) { return nil, nil }

// InterfaceMulticastAddrMessage represents a routing message
// containing network interface address entries.
type InterfaceMulticastAddrMessage struct {
	Header IfmaMsghdr
	Data   []byte
}

func (m *InterfaceMulticastAddrMessage) sockaddr() ([]Sockaddr, error) {
	var sas [RTAX_MAX]Sockaddr
	b := m.Data[:]
	for i := uint(0); i < RTAX_MAX && len(b) >= minRoutingSockaddrLen; i++ {
		if m.Header.Addrs&(1<<i) == 0 {
			continue
		}
		rsa := (*RawSockaddr)(unsafe.Pointer(&b[0]))
		switch rsa.Family {
		case AF_LINK:
			sa, err := parseSockaddrLink(b)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[rsaAlignOf(int(rsa.Len)):]
		case AF_INET, AF_INET6:
			sa, err := parseSockaddrInet(b, rsa.Family)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[rsaAlignOf(int(rsa.Len)):]
		default:
			sa, l, err := parseLinkLayerAddr(b)
			if err != nil {
				return nil, err
			}
			sas[i] = sa
			b = b[l:]
		}
	}
	return sas[:], nil
}
