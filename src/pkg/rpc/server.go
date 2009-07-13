// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

import (
	"gob";
	"log";
	"io";
	"os";
	"reflect";
	"strings";
	"sync";
	"unicode";
	"utf8";
)

// Precompute the reflect type for os.Error.  Can't use os.Error directly
// because Typeof takes an empty interface value.  This is annoying.
var unusedError *os.Error;
var typeOfOsError = reflect.Typeof(unusedError).(*reflect.PtrType).Elem()

type methodType struct {
	method	reflect.Method;
	argType	*reflect.PtrType;
	replyType	*reflect.PtrType;
}

type service struct {
	name	string;	// name of service
	rcvr	reflect.Value;	// receiver of methods for the service
	typ	reflect.Type;	// type of the receiver
	method	map[string] *methodType;	// registered methods
}

// Request is a header written before every RPC call.
type Request struct {
	ServiceMethod	string;
	Seq	uint64;
}

// Response is a header written before every RPC return.
type Response struct {
	ServiceMethod	string;
	Seq	uint64;
	Error	string;
}

// Server represents the set of services available to an RPC client.
// The zero type for Server is ready to have services added.
type Server struct {
	serviceMap	map[string] *service;
}

// Is this a publicly vislble - upper case - name?
func isPublic(name string) bool {
	rune, wid_ := utf8.DecodeRuneInString(name);
	return unicode.IsUpper(rune)
}

// Add publishes in the server the set of methods of the
// recevier value that satisfy the following conditions:
//	- public method
//	- two arguments, both pointers to structs
//	- one return value of type os.Error
// It returns an error if the receiver is not suitable.
func (server *Server) Add(rcvr interface{}) os.Error {
	if server.serviceMap == nil {
		server.serviceMap = make(map[string] *service);
	}
	s := new(service);
	s.typ = reflect.Typeof(rcvr);
	s.rcvr = reflect.NewValue(rcvr);
	path_, sname := reflect.Indirect(s.rcvr).Type().Name();
	if sname == "" {
		log.Exit("rpc: no service name for type", s.typ.String())
	}
	if !isPublic(sname) {
		s := "rpc server.Add: type " + sname + " is not public";
		log.Stderr(s);
		return os.ErrorString(s);
	}
	s.name = sname;
	s.method = make(map[string] *methodType);

	// Install the methods
	for m := 0; m < s.typ.NumMethod(); m++ {
		method := s.typ.Method(m);
		mtype := method.Type;
		mname := method.Name;
		if !isPublic(mname) {
			continue
		}
		// Method needs three ins: receiver, *args, *reply.
		// The args and reply must be structs until gobs are more general.
		if mtype.NumIn() != 3 {
			log.Stderr("method", mname, "has wrong number of ins:", mtype.NumIn());
			continue;
		}
		argType, ok := mtype.In(1).(*reflect.PtrType);
		if !ok {
			log.Stderr(mname, "arg type not a pointer:", argType.String());
			continue;
		}
		if _, ok := argType.Elem().(*reflect.StructType); !ok {
			log.Stderr(mname, "arg type not a pointer to a struct:", argType.String());
			continue;
		}
		replyType, ok := mtype.In(2).(*reflect.PtrType);
		if !ok {
			log.Stderr(mname, "reply type not a pointer:", replyType.String());
			continue;
		}
		if _, ok := replyType.Elem().(*reflect.StructType); !ok {
			log.Stderr(mname, "reply type not a pointer to a struct:", replyType.String());
			continue;
		}
		// Method needs one out: os.Error.
		if mtype.NumOut() != 1 {
			log.Stderr("method", mname, "has wrong number of outs:", mtype.NumOut());
			continue;
		}
		if returnType := mtype.Out(0); returnType != typeOfOsError {
			log.Stderr("method", mname, "returns", returnType.String(), "not os.Error");
			continue;
		}
		s.method[mname] = &methodType{method, argType, replyType};
	}

	if len(s.method) == 0 {
		s := "rpc server.Add: type " + sname + " has no public methods of suitable type";
		log.Stderr(s);
		return os.ErrorString(s);
	}
	server.serviceMap[s.name] = s;
	return nil;
}

func _new(t *reflect.PtrType) *reflect.PtrValue {
	v := reflect.MakeZero(t).(*reflect.PtrValue);
	v.PointTo(reflect.MakeZero(t.Elem()));
	return v;
}

func (s *service) call(sending *sync.Mutex, function *reflect.FuncValue, req *Request, argv, replyv reflect.Value, enc *gob.Encoder) {
	// Invoke the method, providing a new value for the reply.
	returnValues := function.Call([]reflect.Value{s.rcvr, argv, replyv});
	// The return value for the method is an os.Error.
	err := returnValues[0].Interface();
	resp := new(Response);
	if err != nil {
		resp.Error = err.(os.Error).String();
	}
	// Encode the response header
	sending.Lock();
	resp.ServiceMethod = req.ServiceMethod;
	resp.Seq = req.Seq;
	enc.Encode(resp);
	// Encode the reply value.
	enc.Encode(replyv.Interface());
	sending.Unlock();
}

func (server *Server) serve(conn io.ReadWriteCloser) {
	dec := gob.NewDecoder(conn);
	enc := gob.NewEncoder(conn);
	sending := new(sync.Mutex);
	for {
		// Grab the request header.
		req := new(Request);
		err := dec.Decode(req);
		if err != nil {
			panicln("can't handle decode error yet", err.String());
		}
		serviceMethod := strings.Split(req.ServiceMethod, ".", 0);
		if len(serviceMethod) != 2 {
			panicln("service/Method request ill-formed:", req.ServiceMethod);
		}
		// Look up the request.
		service, ok := server.serviceMap[serviceMethod[0]];
		if !ok {
			panicln("can't find service", serviceMethod[0]);
		}
		mtype, ok := service.method[serviceMethod[1]];
		if !ok {
			panicln("can't find method", serviceMethod[1]);
		}
		method := mtype.method;
		// Decode the argument value.
		argv := _new(mtype.argType);
		err = dec.Decode(argv.Interface());
		if err != nil {
			panicln("can't handle payload decode error yet", err.String());
		}
		go service.call(sending, method.Func, req, argv, _new(mtype.replyType), enc);
	}
}

// Serve runs the server on the connection.
func (server *Server) Serve(conn io.ReadWriteCloser) {
	go server.serve(conn)
}
