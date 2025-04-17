// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diameter

type Runnable interface {
	Run()
}

// RunnableFunc is converter which converts function to Runnable interface
type RunnableFunc func()

// Run is Runnable.Run
func (r RunnableFunc) Run() {
	r()
}

type Executor interface {
	ExecuteUnsafe(runnable Runnable)
}

type Promise[T any] interface {
	Future() Future[T]
	Success(value T) bool
	Failure(err error) bool
	IsCompleted() bool
	Complete(result Try[T]) bool
}

type Future[T any] interface {
	OnFailure(cb func(err error), ctx ...Executor)
	OnSuccess(cb func(success T), ctx ...Executor)
	Foreach(f func(v T), ctx ...Executor)
	OnComplete(cb func(try Try[T]), ctx ...Executor)
	IsCompleted() bool
	//	Value() Option[Try[T]]
	Failed() Future[error]
	Recover(f func(err error) T, ctx ...Executor) Future[T]
	RecoverWith(f func(err error) Future[T], ctx ...Executor) Future[T]
}

type Try[T any] struct {
	v   *T
	err error
}

func (r Try[T]) IsSuccess() bool {
	return r.v != nil
}

type ByteBuffer struct {
	pos       int
	buf       []byte
	underflow error
}

// InboundHandler is extends of uclient.NetInboundHandler
type InboundHandler interface {
	OriginHost() string
	OriginRealm() string
}

type transactionID struct {
	hopID uint32
	endID uint32
}

type roundTripper struct {
	promise map[transactionID]Promise[*ByteBuffer]
	host    string
	realm   string
}

func (r *roundTripper) OriginHost() string {
	return r.host
}
func (r *roundTripper) OriginRealm() string {
	return r.realm
}

func NewInboundHandler(host string, realm string, productName string) InboundHandler {
	ret := &roundTripper{promise: make(map[transactionID]Promise[*ByteBuffer]), host: host, realm: realm}

	return ret
}
