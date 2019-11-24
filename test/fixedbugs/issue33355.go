// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code failed on arm64 in the register allocator.
// See issue 33355.

package server

import (
	"bytes"
	"sync"
)

type client struct {
	junk [4]int
	mu   sync.Mutex
	srv  *Server
	gw   *gateway
	msgb [100]byte
}

type gateway struct {
	cfg    *gatewayCfg
	outsim *sync.Map
}

type gatewayCfg struct {
	replyPfx []byte
}

type Account struct {
	Name string
}

type Server struct {
	gateway *srvGateway
}

type srvGateway struct {
	outo     []*client
}

type subscription struct {
	queue   []byte
	client  *client
}

type outsie struct {
	ni    map[string]struct{}
	sl    *Sublist
	qsubs int
}

type Sublist struct {
}

type SublistResult struct {
	psubs []*subscription
	qsubs [][]*subscription
}

var subPool = &sync.Pool{}

func (c *client) sendMsgToGateways(acc *Account, msg, subject, reply []byte, qgroups [][]byte) {
	var gws []*client
	gw := c.srv.gateway
	for i := 0; i < len(gw.outo); i++ {
		gws = append(gws, gw.outo[i])
	}
	var (
		subj       = string(subject)
		queuesa    = [512]byte{}
		queues     = queuesa[:0]
		mreply     []byte
		dstPfx     []byte
		checkReply = len(reply) > 0
	)

	sub := subPool.Get().(*subscription)

	if subjectStartsWithGatewayReplyPrefix(subject) {
		dstPfx = subject[:8]
	}
	for i := 0; i < len(gws); i++ {
		gwc := gws[i]
		if dstPfx != nil {
			gwc.mu.Lock()
			ok := bytes.Equal(dstPfx, gwc.gw.cfg.replyPfx)
			gwc.mu.Unlock()
			if !ok {
				continue
			}
		} else {
			qr := gwc.gatewayInterest(acc.Name, subj)
			queues = queuesa[:0]
			for i := 0; i < len(qr.qsubs); i++ {
				qsubs := qr.qsubs[i]
				queue := qsubs[0].queue
				add := true
				for _, qn := range qgroups {
					if bytes.Equal(queue, qn) {
						add = false
						break
					}
				}
				if add {
					qgroups = append(qgroups, queue)
				}
			}
			if len(queues) == 0 {
				continue
			}
		}
		if checkReply {
			checkReply = false
			mreply = reply
		}
		mh := c.msgb[:10]
		mh = append(mh, subject...)
		if len(queues) > 0 {
			mh = append(mh, mreply...)
			mh = append(mh, queues...)
		}
		sub.client = gwc
	}
	subPool.Put(sub)
}

func subjectStartsWithGatewayReplyPrefix(subj []byte) bool {
	return len(subj) > 8 && string(subj[:4]) == "foob"
}

func (c *client) gatewayInterest(acc, subj string) *SublistResult {
	ei, _ := c.gw.outsim.Load(acc)
	var r *SublistResult
	e := ei.(*outsie)
	r = e.sl.Match(subj)
	return r
}

func (s *Sublist) Match(subject string) *SublistResult {
	return nil
}

