// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Netlink sockets and messages

package syscall

import (
	"unsafe"
)

// Round the length of a netlink message up to align it properly.
func nlmAlignOf(msglen int) int {
	return (msglen + NLMSG_ALIGNTO - 1) & ^(NLMSG_ALIGNTO - 1)
}

// Round the length of a netlink route attribute up to align it
// properly.
func rtaAlignOf(attrlen int) int {
	return (attrlen + RTA_ALIGNTO - 1) & ^(RTA_ALIGNTO - 1)
}

// NetlinkRouteRequest represents the request message to receive
// routing and link states from the kernel.
type NetlinkRouteRequest struct {
	Header NlMsghdr
	Data   RtGenmsg
}

func (rr *NetlinkRouteRequest) toWireFormat() []byte {
	b := make([]byte, rr.Header.Len)
	*(*uint32)(unsafe.Pointer(&b[0:4][0])) = rr.Header.Len
	*(*uint16)(unsafe.Pointer(&b[4:6][0])) = rr.Header.Type
	*(*uint16)(unsafe.Pointer(&b[6:8][0])) = rr.Header.Flags
	*(*uint32)(unsafe.Pointer(&b[8:12][0])) = rr.Header.Seq
	*(*uint32)(unsafe.Pointer(&b[12:16][0])) = rr.Header.Pid
	b[16] = byte(rr.Data.Family)
	return b
}

func newNetlinkRouteRequest(proto, seq, family int) []byte {
	rr := &NetlinkRouteRequest{}
	rr.Header.Len = uint32(NLMSG_HDRLEN + SizeofRtGenmsg)
	rr.Header.Type = uint16(proto)
	rr.Header.Flags = NLM_F_DUMP | NLM_F_REQUEST
	rr.Header.Seq = uint32(seq)
	rr.Data.Family = uint8(family)
	return rr.toWireFormat()
}

// NetlinkRIB returns routing information base, as known as RIB,
// which consists of network facility information, states and
// parameters.
func NetlinkRIB(proto, family int) ([]byte, error) {
	var (
		lsanl SockaddrNetlink
		tab   []byte
	)

	s, e := Socket(AF_NETLINK, SOCK_RAW, 0)
	if e != nil {
		return nil, e
	}
	defer Close(s)

	lsanl.Family = AF_NETLINK
	e = Bind(s, &lsanl)
	if e != nil {
		return nil, e
	}

	seq := 1
	wb := newNetlinkRouteRequest(proto, seq, family)
	e = Sendto(s, wb, 0, &lsanl)
	if e != nil {
		return nil, e
	}

	for {
		var (
			rb  []byte
			nr  int
			lsa Sockaddr
		)

		rb = make([]byte, Getpagesize())
		nr, _, e = Recvfrom(s, rb, 0)
		if e != nil {
			return nil, e
		}
		if nr < NLMSG_HDRLEN {
			return nil, EINVAL
		}
		rb = rb[:nr]
		tab = append(tab, rb...)

		msgs, _ := ParseNetlinkMessage(rb)
		for _, m := range msgs {
			if lsa, e = Getsockname(s); e != nil {
				return nil, e
			}
			switch v := lsa.(type) {
			case *SockaddrNetlink:
				if m.Header.Seq != uint32(seq) || m.Header.Pid != v.Pid {
					return nil, EINVAL
				}
			default:
				return nil, EINVAL
			}
			if m.Header.Type == NLMSG_DONE {
				goto done
			}
			if m.Header.Type == NLMSG_ERROR {
				return nil, EINVAL
			}
		}
	}

done:
	return tab, nil
}

// NetlinkMessage represents the netlink message.
type NetlinkMessage struct {
	Header NlMsghdr
	Data   []byte
}

// ParseNetlinkMessage parses buf as netlink messages and returns
// the slice containing the NetlinkMessage structs.
func ParseNetlinkMessage(buf []byte) ([]NetlinkMessage, error) {
	var (
		h    *NlMsghdr
		dbuf []byte
		dlen int
		e    error
		msgs []NetlinkMessage
	)

	for len(buf) >= NLMSG_HDRLEN {
		h, dbuf, dlen, e = netlinkMessageHeaderAndData(buf)
		if e != nil {
			break
		}
		m := NetlinkMessage{}
		m.Header = *h
		m.Data = dbuf[:int(h.Len)-NLMSG_HDRLEN]
		msgs = append(msgs, m)
		buf = buf[dlen:]
	}

	return msgs, e
}

func netlinkMessageHeaderAndData(buf []byte) (*NlMsghdr, []byte, int, error) {
	h := (*NlMsghdr)(unsafe.Pointer(&buf[0]))
	if int(h.Len) < NLMSG_HDRLEN || int(h.Len) > len(buf) {
		return nil, nil, 0, EINVAL
	}
	return h, buf[NLMSG_HDRLEN:], nlmAlignOf(int(h.Len)), nil
}

// NetlinkRouteAttr represents the netlink route attribute.
type NetlinkRouteAttr struct {
	Attr  RtAttr
	Value []byte
}

// ParseNetlinkRouteAttr parses msg's payload as netlink route
// attributes and returns the slice containing the NetlinkRouteAttr
// structs.
func ParseNetlinkRouteAttr(msg *NetlinkMessage) ([]NetlinkRouteAttr, error) {
	var (
		buf   []byte
		a     *RtAttr
		alen  int
		vbuf  []byte
		e     error
		attrs []NetlinkRouteAttr
	)

	switch msg.Header.Type {
	case RTM_NEWLINK, RTM_DELLINK:
		buf = msg.Data[SizeofIfInfomsg:]
	case RTM_NEWADDR, RTM_DELADDR:
		buf = msg.Data[SizeofIfAddrmsg:]
	case RTM_NEWROUTE, RTM_DELROUTE:
		buf = msg.Data[SizeofRtMsg:]
	default:
		return nil, EINVAL
	}

	for len(buf) >= SizeofRtAttr {
		a, vbuf, alen, e = netlinkRouteAttrAndValue(buf)
		if e != nil {
			break
		}
		ra := NetlinkRouteAttr{}
		ra.Attr = *a
		ra.Value = vbuf[:int(a.Len)-SizeofRtAttr]
		attrs = append(attrs, ra)
		buf = buf[alen:]
	}

	return attrs, nil
}

func netlinkRouteAttrAndValue(buf []byte) (*RtAttr, []byte, int, error) {
	h := (*RtAttr)(unsafe.Pointer(&buf[0]))
	if int(h.Len) < SizeofRtAttr || int(h.Len) > len(buf) {
		return nil, nil, 0, EINVAL
	}
	return h, buf[SizeofRtAttr:], rtaAlignOf(int(h.Len)), nil
}
