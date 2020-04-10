package protocol

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
)

type loggingStream struct {
	stream jsonrpc2.Stream
	logMu  sync.Mutex
	log    io.Writer
}

// LoggingStream returns a stream that does LSP protocol logging too
func LoggingStream(str jsonrpc2.Stream, w io.Writer) jsonrpc2.Stream {
	return &loggingStream{stream: str, log: w}
}

func (s *loggingStream) Read(ctx context.Context) ([]byte, int64, error) {
	data, count, err := s.stream.Read(ctx)
	if err == nil {
		s.logMu.Lock()
		defer s.logMu.Unlock()
		logIn(s.log, data)
	}
	return data, count, err
}

func (s *loggingStream) Write(ctx context.Context, data []byte) (int64, error) {
	s.logMu.Lock()
	defer s.logMu.Unlock()
	logOut(s.log, data)
	count, err := s.stream.Write(ctx, data)
	return count, err
}

// wireCombined has all the fields of both Request and Response.
// We can decode this and then work out which it is.
type wireCombined struct {
	VersionTag interface{}      `json:"jsonrpc"`
	ID         *jsonrpc2.ID     `json:"id,omitempty"`
	Method     string           `json:"method"`
	Params     *json.RawMessage `json:"params,omitempty"`
	Result     *json.RawMessage `json:"result,omitempty"`
	Error      *wireError       `json:"error,omitempty"`
}

type wireError struct {
	Code    int64            `json:"code"`
	Message string           `json:"message"`
	Data    *json.RawMessage `json:"data"`
}

type req struct {
	method string
	start  time.Time
}

type mapped struct {
	mu          sync.Mutex
	clientCalls map[string]req
	serverCalls map[string]req
}

var maps = &mapped{
	sync.Mutex{},
	make(map[string]req),
	make(map[string]req),
}

// these 4 methods are each used exactly once, but it seemed
// better to have the encapsulation rather than ad hoc mutex
// code in 4 places
func (m *mapped) client(id string, del bool) req {
	m.mu.Lock()
	defer m.mu.Unlock()
	v := m.clientCalls[id]
	if del {
		delete(m.clientCalls, id)
	}
	return v
}

func (m *mapped) server(id string, del bool) req {
	m.mu.Lock()
	defer m.mu.Unlock()
	v := m.serverCalls[id]
	if del {
		delete(m.serverCalls, id)
	}
	return v
}

func (m *mapped) setClient(id string, r req) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.clientCalls[id] = r
}

func (m *mapped) setServer(id string, r req) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.serverCalls[id] = r
}

const eor = "\r\n\r\n\r\n"

func logCommon(outfd io.Writer, data []byte) (*wireCombined, time.Time, string) {
	if outfd == nil {
		return nil, time.Time{}, ""
	}
	var v wireCombined
	err := json.Unmarshal(data, &v)
	if err != nil {
		fmt.Fprintf(outfd, "Unmarshal %v\n", err)
		panic(err) // do better
	}
	tm := time.Now()
	tmfmt := tm.Format("15:04:05.000 PM")
	return &v, tm, tmfmt
}

// logOut and logIn could be combined. "received"<->"Sending", serverCalls<->clientCalls
// but it wouldn't be a lot shorter or clearer and "shutdown" is a special case

// Writing a message to the client, log it
func logOut(outfd io.Writer, data []byte) {
	v, tm, tmfmt := logCommon(outfd, data)
	if v == nil {
		return
	}
	id := fmt.Sprint(v.ID)
	if v.Error != nil {
		fmt.Fprintf(outfd, "[Error - %s] Received #%s %s%s", tmfmt, id, v.Error.Message, eor)
		return
	}
	buf := strings.Builder{}
	fmt.Fprintf(&buf, "[Trace - %s] ", tmfmt) // common beginning
	if v.ID != nil && v.Method != "" && v.Params != nil {
		fmt.Fprintf(&buf, "Received request '%s - (%s)'.\n", v.Method, id)
		fmt.Fprintf(&buf, "Params: %s%s", *v.Params, eor)
		maps.setServer(id, req{method: v.Method, start: tm})
	} else if v.ID != nil && v.Method == "" && v.Params == nil {
		cc := maps.client(id, true)
		elapsed := tm.Sub(cc.start)
		fmt.Fprintf(&buf, "Received response '%s - (%s)' in %dms.\n",
			cc.method, id, elapsed/time.Millisecond)
		if v.Result == nil {
			fmt.Fprintf(&buf, "Result: {}%s", eor)
		} else {
			fmt.Fprintf(&buf, "Result: %s%s", string(*v.Result), eor)
		}
	} else if v.ID == nil && v.Method != "" && v.Params != nil {
		p := "null"
		if v.Params != nil {
			p = string(*v.Params)
		}
		fmt.Fprintf(&buf, "Received notification '%s'.\n", v.Method)
		fmt.Fprintf(&buf, "Params: %s%s", p, eor)
	} else { // for completeness, as it should never happen
		buf = strings.Builder{} // undo common Trace
		fmt.Fprintf(&buf, "[Error - %s] on write ID?%v method:%q Params:%v Result:%v Error:%v%s",
			tmfmt, v.ID != nil, v.Method, v.Params != nil,
			v.Result != nil, v.Error != nil, eor)
		p := "null"
		if v.Params != nil {
			p = string(*v.Params)
		}
		r := "null"
		if v.Result != nil {
			r = string(*v.Result)
		}
		fmt.Fprintf(&buf, "%s\n%s\n%s%s", p, r, v.Error.Message, eor)
	}
	outfd.Write([]byte(buf.String()))
}

// Got a message from the client, log it
func logIn(outfd io.Writer, data []byte) {
	v, tm, tmfmt := logCommon(outfd, data)
	if v == nil {
		return
	}
	id := fmt.Sprint(v.ID)
	// ID Method Params => Sending request
	// ID !Method Result(might be null, but !Params) => Sending response (could we get an Error?)
	// !ID Method Params => Sending notification
	if v.Error != nil { // does this ever happen?
		fmt.Fprintf(outfd, "[Error - %s] Sent #%s %s%s", tmfmt, id, v.Error.Message, eor)
		return
	}
	buf := strings.Builder{}
	fmt.Fprintf(&buf, "[Trace - %s] ", tmfmt) // common beginning
	if v.ID != nil && v.Method != "" && (v.Params != nil || v.Method == "shutdown") {
		fmt.Fprintf(&buf, "Sending request '%s - (%s)'.\n", v.Method, id)
		x := "{}"
		if v.Params != nil {
			x = string(*v.Params)
		}
		fmt.Fprintf(&buf, "Params: %s%s", x, eor)
		maps.setClient(id, req{method: v.Method, start: tm})
	} else if v.ID != nil && v.Method == "" && v.Params == nil {
		sc := maps.server(id, true)
		elapsed := tm.Sub(sc.start)
		fmt.Fprintf(&buf, "Sending response '%s - (%s)' took %dms.\n",
			sc.method, id, elapsed/time.Millisecond)
		if v.Result == nil {
			fmt.Fprintf(&buf, "Result: {}%s", eor)
		} else {
			fmt.Fprintf(&buf, "Result: %s%s", string(*v.Result), eor)
		}
	} else if v.ID == nil && v.Method != "" {
		p := "null"
		if v.Params != nil {
			p = string(*v.Params)
		}
		fmt.Fprintf(&buf, "Sending notification '%s'.\n", v.Method)
		fmt.Fprintf(&buf, "Params: %s%s", p, eor)
	} else { // for completeness, as it should never happen
		buf = strings.Builder{} // undo common Trace
		fmt.Fprintf(&buf, "[Error - %s] on read ID?%v method:%q Params:%v Result:%v Error:%v%s",
			tmfmt, v.ID != nil, v.Method, v.Params != nil,
			v.Result != nil, v.Error != nil, eor)
		p := "null"
		if v.Params != nil {
			p = string(*v.Params)
		}
		r := "null"
		if v.Result != nil {
			r = string(*v.Result)
		}
		fmt.Fprintf(&buf, "%s\n%s\n%s%s", p, r, v.Error.Message, eor)
	}
	outfd.Write([]byte(buf.String()))
}
