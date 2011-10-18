// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"big"
	"bytes"
	"io"
	"os"
	"reflect"
)

// These are SSH message type numbers. They are scattered around several
// documents but many were taken from
// http://www.iana.org/assignments/ssh-parameters/ssh-parameters.xml#ssh-parameters-1
const (
	msgDisconnect     = 1
	msgIgnore         = 2
	msgUnimplemented  = 3
	msgDebug          = 4
	msgServiceRequest = 5
	msgServiceAccept  = 6

	msgKexInit = 20
	msgNewKeys = 21

	msgKexDHInit  = 30
	msgKexDHReply = 31

	msgUserAuthRequest  = 50
	msgUserAuthFailure  = 51
	msgUserAuthSuccess  = 52
	msgUserAuthBanner   = 53
	msgUserAuthPubKeyOk = 60

	msgGlobalRequest  = 80
	msgRequestSuccess = 81
	msgRequestFailure = 82

	msgChannelOpen         = 90
	msgChannelOpenConfirm  = 91
	msgChannelOpenFailure  = 92
	msgChannelWindowAdjust = 93
	msgChannelData         = 94
	msgChannelExtendedData = 95
	msgChannelEOF          = 96
	msgChannelClose        = 97
	msgChannelRequest      = 98
	msgChannelSuccess      = 99
	msgChannelFailure      = 100
)

// SSH messages:
//
// These structures mirror the wire format of the corresponding SSH messages.
// They are marshaled using reflection with the marshal and unmarshal functions
// in this file. The only wrinkle is that a final member of type []byte with a
// ssh tag of "rest" receives the remainder of a packet when unmarshaling.

// See RFC 4253, section 11.1.
type disconnectMsg struct {
	Reason   uint32
	Message  string
	Language string
}

// See RFC 4253, section 7.1.
type kexInitMsg struct {
	Cookie                  [16]byte
	KexAlgos                []string
	ServerHostKeyAlgos      []string
	CiphersClientServer     []string
	CiphersServerClient     []string
	MACsClientServer        []string
	MACsServerClient        []string
	CompressionClientServer []string
	CompressionServerClient []string
	LanguagesClientServer   []string
	LanguagesServerClient   []string
	FirstKexFollows         bool
	Reserved                uint32
}

// See RFC 4253, section 8.
type kexDHInitMsg struct {
	X *big.Int
}

type kexDHReplyMsg struct {
	HostKey   []byte
	Y         *big.Int
	Signature []byte
}

// See RFC 4253, section 10.
type serviceRequestMsg struct {
	Service string
}

// See RFC 4253, section 10.
type serviceAcceptMsg struct {
	Service string
}

// See RFC 4252, section 5.
type userAuthRequestMsg struct {
	User    string
	Service string
	Method  string
	Payload []byte `ssh:"rest"`
}

// See RFC 4252, section 5.1
type userAuthFailureMsg struct {
	Methods        []string
	PartialSuccess bool
}

// See RFC 4254, section 5.1.
type channelOpenMsg struct {
	ChanType         string
	PeersId          uint32
	PeersWindow      uint32
	MaxPacketSize    uint32
	TypeSpecificData []byte `ssh:"rest"`
}

// See RFC 4254, section 5.1.
type channelOpenConfirmMsg struct {
	PeersId          uint32
	MyId             uint32
	MyWindow         uint32
	MaxPacketSize    uint32
	TypeSpecificData []byte `ssh:"rest"`
}

// See RFC 4254, section 5.1.
type channelOpenFailureMsg struct {
	PeersId  uint32
	Reason   uint32
	Message  string
	Language string
}

// See RFC 4254, section 5.2.
type channelData struct {
	PeersId uint32
	Payload []byte `ssh:"rest"`
}

// See RFC 4254, section 5.2.
type channelExtendedData struct {
	PeersId  uint32
	Datatype uint32
	Data     string
}

type channelRequestMsg struct {
	PeersId             uint32
	Request             string
	WantReply           bool
	RequestSpecificData []byte `ssh:"rest"`
}

// See RFC 4254, section 5.4.
type channelRequestSuccessMsg struct {
	PeersId uint32
}

// See RFC 4254, section 5.4.
type channelRequestFailureMsg struct {
	PeersId uint32
}

// See RFC 4254, section 5.3
type channelCloseMsg struct {
	PeersId uint32
}

// See RFC 4254, section 5.3
type channelEOFMsg struct {
	PeersId uint32
}

// See RFC 4254, section 4
type globalRequestMsg struct {
	Type      string
	WantReply bool
}

// See RFC 4254, section 5.2
type windowAdjustMsg struct {
	PeersId         uint32
	AdditionalBytes uint32
}

// See RFC 4252, section 7
type userAuthPubKeyOkMsg struct {
	Algo   string
	PubKey string
}

// unmarshal parses the SSH wire data in packet into out using reflection.
// expectedType is the expected SSH message type. It either returns nil on
// success, or a ParseError or UnexpectedMessageError on error.
func unmarshal(out interface{}, packet []byte, expectedType uint8) os.Error {
	if len(packet) == 0 {
		return ParseError{expectedType}
	}
	if packet[0] != expectedType {
		return UnexpectedMessageError{expectedType, packet[0]}
	}
	packet = packet[1:]

	v := reflect.ValueOf(out).Elem()
	structType := v.Type()
	var ok bool
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		t := field.Type()
		switch t.Kind() {
		case reflect.Bool:
			if len(packet) < 1 {
				return ParseError{expectedType}
			}
			field.SetBool(packet[0] != 0)
			packet = packet[1:]
		case reflect.Array:
			if t.Elem().Kind() != reflect.Uint8 {
				panic("array of non-uint8")
			}
			if len(packet) < t.Len() {
				return ParseError{expectedType}
			}
			for j := 0; j < t.Len(); j++ {
				field.Index(j).Set(reflect.ValueOf(packet[j]))
			}
			packet = packet[t.Len():]
		case reflect.Uint32:
			var u32 uint32
			if u32, packet, ok = parseUint32(packet); !ok {
				return ParseError{expectedType}
			}
			field.SetUint(uint64(u32))
		case reflect.String:
			var s []byte
			if s, packet, ok = parseString(packet); !ok {
				return ParseError{expectedType}
			}
			field.SetString(string(s))
		case reflect.Slice:
			switch t.Elem().Kind() {
			case reflect.Uint8:
				if structType.Field(i).Tag.Get("ssh") == "rest" {
					field.Set(reflect.ValueOf(packet))
					packet = nil
				} else {
					var s []byte
					if s, packet, ok = parseString(packet); !ok {
						return ParseError{expectedType}
					}
					field.Set(reflect.ValueOf(s))
				}
			case reflect.String:
				var nl []string
				if nl, packet, ok = parseNameList(packet); !ok {
					return ParseError{expectedType}
				}
				field.Set(reflect.ValueOf(nl))
			default:
				panic("slice of unknown type")
			}
		case reflect.Ptr:
			if t == bigIntType {
				var n *big.Int
				if n, packet, ok = parseInt(packet); !ok {
					return ParseError{expectedType}
				}
				field.Set(reflect.ValueOf(n))
			} else {
				panic("pointer to unknown type")
			}
		default:
			panic("unknown type")
		}
	}

	if len(packet) != 0 {
		return ParseError{expectedType}
	}

	return nil
}

// marshal serializes the message in msg, using the given message type.
func marshal(msgType uint8, msg interface{}) []byte {
	var out []byte
	out = append(out, msgType)

	v := reflect.ValueOf(msg)
	structType := v.Type()
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		t := field.Type()
		switch t.Kind() {
		case reflect.Bool:
			var v uint8
			if field.Bool() {
				v = 1
			}
			out = append(out, v)
		case reflect.Array:
			if t.Elem().Kind() != reflect.Uint8 {
				panic("array of non-uint8")
			}
			for j := 0; j < t.Len(); j++ {
				out = append(out, byte(field.Index(j).Uint()))
			}
		case reflect.Uint32:
			u32 := uint32(field.Uint())
			out = append(out, byte(u32>>24))
			out = append(out, byte(u32>>16))
			out = append(out, byte(u32>>8))
			out = append(out, byte(u32))
		case reflect.String:
			s := field.String()
			out = append(out, byte(len(s)>>24))
			out = append(out, byte(len(s)>>16))
			out = append(out, byte(len(s)>>8))
			out = append(out, byte(len(s)))
			out = append(out, s...)
		case reflect.Slice:
			switch t.Elem().Kind() {
			case reflect.Uint8:
				length := field.Len()
				if structType.Field(i).Tag.Get("ssh") != "rest" {
					out = append(out, byte(length>>24))
					out = append(out, byte(length>>16))
					out = append(out, byte(length>>8))
					out = append(out, byte(length))
				}
				for j := 0; j < length; j++ {
					out = append(out, byte(field.Index(j).Uint()))
				}
			case reflect.String:
				var length int
				for j := 0; j < field.Len(); j++ {
					if j != 0 {
						length++ /* comma */
					}
					length += len(field.Index(j).String())
				}

				out = append(out, byte(length>>24))
				out = append(out, byte(length>>16))
				out = append(out, byte(length>>8))
				out = append(out, byte(length))
				for j := 0; j < field.Len(); j++ {
					if j != 0 {
						out = append(out, ',')
					}
					out = append(out, field.Index(j).String()...)
				}
			default:
				panic("slice of unknown type")
			}
		case reflect.Ptr:
			if t == bigIntType {
				var n *big.Int
				nValue := reflect.ValueOf(&n)
				nValue.Elem().Set(field)
				needed := intLength(n)
				oldLength := len(out)

				if cap(out)-len(out) < needed {
					newOut := make([]byte, len(out), 2*(len(out)+needed))
					copy(newOut, out)
					out = newOut
				}
				out = out[:oldLength+needed]
				marshalInt(out[oldLength:], n)
			} else {
				panic("pointer to unknown type")
			}
		}
	}

	return out
}

var bigOne = big.NewInt(1)

func parseString(in []byte) (out, rest []byte, ok bool) {
	if len(in) < 4 {
		return
	}
	length := uint32(in[0])<<24 | uint32(in[1])<<16 | uint32(in[2])<<8 | uint32(in[3])
	if uint32(len(in)) < 4+length {
		return
	}
	out = in[4 : 4+length]
	rest = in[4+length:]
	ok = true
	return
}

var comma = []byte{','}

func parseNameList(in []byte) (out []string, rest []byte, ok bool) {
	contents, rest, ok := parseString(in)
	if !ok {
		return
	}
	if len(contents) == 0 {
		return
	}
	parts := bytes.Split(contents, comma)
	out = make([]string, len(parts))
	for i, part := range parts {
		out[i] = string(part)
	}
	return
}

func parseInt(in []byte) (out *big.Int, rest []byte, ok bool) {
	contents, rest, ok := parseString(in)
	if !ok {
		return
	}
	out = new(big.Int)

	if len(contents) > 0 && contents[0]&0x80 == 0x80 {
		// This is a negative number
		notBytes := make([]byte, len(contents))
		for i := range notBytes {
			notBytes[i] = ^contents[i]
		}
		out.SetBytes(notBytes)
		out.Add(out, bigOne)
		out.Neg(out)
	} else {
		// Positive number
		out.SetBytes(contents)
	}
	ok = true
	return
}

func parseUint32(in []byte) (out uint32, rest []byte, ok bool) {
	if len(in) < 4 {
		return
	}
	out = uint32(in[0])<<24 | uint32(in[1])<<16 | uint32(in[2])<<8 | uint32(in[3])
	rest = in[4:]
	ok = true
	return
}

const maxPacketSize = 36000

func nameListLength(namelist []string) int {
	length := 4 /* uint32 length prefix */
	for i, name := range namelist {
		if i != 0 {
			length++ /* comma */
		}
		length += len(name)
	}
	return length
}

func intLength(n *big.Int) int {
	length := 4 /* length bytes */
	if n.Sign() < 0 {
		nMinus1 := new(big.Int).Neg(n)
		nMinus1.Sub(nMinus1, bigOne)
		bitLen := nMinus1.BitLen()
		if bitLen%8 == 0 {
			// The number will need 0xff padding
			length++
		}
		length += (bitLen + 7) / 8
	} else if n.Sign() == 0 {
		// A zero is the zero length string
	} else {
		bitLen := n.BitLen()
		if bitLen%8 == 0 {
			// The number will need 0x00 padding
			length++
		}
		length += (bitLen + 7) / 8
	}

	return length
}

func marshalInt(to []byte, n *big.Int) []byte {
	lengthBytes := to
	to = to[4:]
	length := 0

	if n.Sign() < 0 {
		// A negative number has to be converted to two's-complement
		// form. So we'll subtract 1 and invert. If the
		// most-significant-bit isn't set then we'll need to pad the
		// beginning with 0xff in order to keep the number negative.
		nMinus1 := new(big.Int).Neg(n)
		nMinus1.Sub(nMinus1, bigOne)
		bytes := nMinus1.Bytes()
		for i := range bytes {
			bytes[i] ^= 0xff
		}
		if len(bytes) == 0 || bytes[0]&0x80 == 0 {
			to[0] = 0xff
			to = to[1:]
			length++
		}
		nBytes := copy(to, bytes)
		to = to[nBytes:]
		length += nBytes
	} else if n.Sign() == 0 {
		// A zero is the zero length string
	} else {
		bytes := n.Bytes()
		if len(bytes) > 0 && bytes[0]&0x80 != 0 {
			// We'll have to pad this with a 0x00 in order to
			// stop it looking like a negative number.
			to[0] = 0
			to = to[1:]
			length++
		}
		nBytes := copy(to, bytes)
		to = to[nBytes:]
		length += nBytes
	}

	lengthBytes[0] = byte(length >> 24)
	lengthBytes[1] = byte(length >> 16)
	lengthBytes[2] = byte(length >> 8)
	lengthBytes[3] = byte(length)
	return to
}

func writeInt(w io.Writer, n *big.Int) {
	length := intLength(n)
	buf := make([]byte, length)
	marshalInt(buf, n)
	w.Write(buf)
}

func writeString(w io.Writer, s []byte) {
	var lengthBytes [4]byte
	lengthBytes[0] = byte(len(s) >> 24)
	lengthBytes[1] = byte(len(s) >> 16)
	lengthBytes[2] = byte(len(s) >> 8)
	lengthBytes[3] = byte(len(s))
	w.Write(lengthBytes[:])
	w.Write(s)
}

func stringLength(s []byte) int {
	return 4 + len(s)
}

func marshalString(to []byte, s []byte) []byte {
	to[0] = byte(len(s) >> 24)
	to[1] = byte(len(s) >> 16)
	to[2] = byte(len(s) >> 8)
	to[3] = byte(len(s))
	to = to[4:]
	copy(to, s)
	return to[len(s):]
}

var bigIntType = reflect.TypeOf((*big.Int)(nil))

// Decode a packet into it's corresponding message.
func decode(packet []byte) interface{} {
	var msg interface{}
	switch packet[0] {
	case msgDisconnect:
		msg = new(disconnectMsg)
	case msgServiceRequest:
		msg = new(serviceRequestMsg)
	case msgServiceAccept:
		msg = new(serviceAcceptMsg)
	case msgKexInit:
		msg = new(kexInitMsg)
	case msgKexDHInit:
		msg = new(kexDHInitMsg)
	case msgKexDHReply:
		msg = new(kexDHReplyMsg)
	case msgUserAuthRequest:
		msg = new(userAuthRequestMsg)
	case msgUserAuthFailure:
		msg = new(userAuthFailureMsg)
	case msgUserAuthPubKeyOk:
		msg = new(userAuthPubKeyOkMsg)
	case msgGlobalRequest:
		msg = new(globalRequestMsg)
	case msgRequestSuccess:
		msg = new(channelRequestSuccessMsg)
	case msgRequestFailure:
		msg = new(channelRequestFailureMsg)
	case msgChannelOpen:
		msg = new(channelOpenMsg)
	case msgChannelOpenConfirm:
		msg = new(channelOpenConfirmMsg)
	case msgChannelOpenFailure:
		msg = new(channelOpenFailureMsg)
	case msgChannelWindowAdjust:
		msg = new(windowAdjustMsg)
	case msgChannelData:
		msg = new(channelData)
	case msgChannelExtendedData:
		msg = new(channelExtendedData)
	case msgChannelEOF:
		msg = new(channelEOFMsg)
	case msgChannelClose:
		msg = new(channelCloseMsg)
	case msgChannelRequest:
		msg = new(channelRequestMsg)
	case msgChannelSuccess:
		msg = new(channelRequestSuccessMsg)
	case msgChannelFailure:
		msg = new(channelRequestFailureMsg)
	default:
		return UnexpectedMessageError{0, packet[0]}
	}
	if err := unmarshal(msg, packet, packet[0]); err != nil {
		return err
	}
	return msg
}
