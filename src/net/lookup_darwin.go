// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package net

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"golang.org/x/net/dns/dnsmessage"
)

func resSearch(ctx context.Context, hostname string, rtype, class int32) ([]IPAddr, error) {

	var byteHostname = []byte(hostname)
	var responseBuffer = [512]byte{}

	retcode := res_search(&byteHostname[0], class, rtype, &responseBuffer[0], 512)
	if retcode < 0 {
		return nil, errors.New("could not complete domain resolution")
	}

	msg := &dnsmessage.Message{}
	err := msg.Unpack(responseBuffer[:])
	if err != nil {
		return nil, fmt.Errorf("could not parse dns response: %s", err.Error())
	}

	// parse received answers
	var dnsParser dnsmessage.Parser

	if _, err := dnsParser.Start(responseBuffer); err != nil {
		return nil, err
	}

	var answers []IPAddr
	for {
		h, err := dnsParser.AnswerHeader()
		if err == dnsmessage.ErrSectionDone {
			break
		}
		if err != nil {
			return nil, err
		}

		if (h.Type != dnsmessage.TypeA && h.Type != dnsmessage.TypeAAAA) ||
			h.Class != dnsmessage.ClassINET {
			continue
		}

		if !strings.EqualFold(h.Name.String(), hostname) {
			if err := p.SkipAnswer(); err != nil {
				return nil, err
			}
			continue
		}

		switch h.Type {
		case dnsmessage.TypeA:
			r, err := dnsParser.AResource()
			if err != nil {
				return nil, err
			}
			answers = append(answers, IPAddr{IP: r.A[:]})
		case dnsmessage.TypeAAAA:
			r, err := dnsParser.AAAAResource()
			if err != nil {
				return nil, err
			}
			answers = append(answers, IPAddr{IP: r.AAAA[:]})
		}
	}
	return answers, nil
}

//go:nosplit
//go:cgo_unsafe_args
func res_search(name *byte, class int32, rtype int32, answer *byte, anslen int32) int32 {
	return libcCall(unsafe.Pointer(funcPC(res_search_trampoline)), unsafe.Pointer(&name))
}
func res_search_trampoline()

//go:cgo_import_dynamic libc_res_search res_search "/usr/lib/libSystem.B.dylib"
