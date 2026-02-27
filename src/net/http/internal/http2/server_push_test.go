// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2_test

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strconv"
	"testing"
	"testing/synctest"
	"time"

	. "net/http/internal/http2"
)

func TestServer_Push_Success(t *testing.T) { synctestTest(t, testServer_Push_Success) }
func testServer_Push_Success(t testing.TB) {
	const (
		mainBody   = "<html>index page</html>"
		pushedBody = "<html>pushed page</html>"
		userAgent  = "testagent"
		cookie     = "testcookie"
	)

	var stURL string
	checkPromisedReq := func(r *http.Request, wantMethod string, wantH http.Header) error {
		if got, want := r.Method, wantMethod; got != want {
			return fmt.Errorf("promised Req.Method=%q, want %q", got, want)
		}
		if got, want := r.Header, wantH; !reflect.DeepEqual(got, want) {
			return fmt.Errorf("promised Req.Header=%q, want %q", got, want)
		}
		if got, want := "https://"+r.Host, stURL; got != want {
			return fmt.Errorf("promised Req.Host=%q, want %q", got, want)
		}
		if r.Body == nil {
			return fmt.Errorf("nil Body")
		}
		if buf, err := io.ReadAll(r.Body); err != nil || len(buf) != 0 {
			return fmt.Errorf("ReadAll(Body)=%q,%v, want '',nil", buf, err)
		}
		return nil
	}

	errc := make(chan error, 3)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.RequestURI() {
		case "/":
			// Push "/pushed?get" as a GET request, using an absolute URL.
			opt := &http.PushOptions{
				Header: http.Header{
					"User-Agent": {userAgent},
				},
			}
			if err := w.(http.Pusher).Push(stURL+"/pushed?get", opt); err != nil {
				errc <- fmt.Errorf("error pushing /pushed?get: %v", err)
				return
			}
			// Push "/pushed?head" as a HEAD request, using a path.
			opt = &http.PushOptions{
				Method: "HEAD",
				Header: http.Header{
					"User-Agent": {userAgent},
					"Cookie":     {cookie},
				},
			}
			if err := w.(http.Pusher).Push("/pushed?head", opt); err != nil {
				errc <- fmt.Errorf("error pushing /pushed?head: %v", err)
				return
			}
			w.Header().Set("Content-Type", "text/html")
			w.Header().Set("Content-Length", strconv.Itoa(len(mainBody)))
			w.WriteHeader(200)
			io.WriteString(w, mainBody)
			errc <- nil

		case "/pushed?get":
			wantH := http.Header{}
			wantH.Set("User-Agent", userAgent)
			if err := checkPromisedReq(r, "GET", wantH); err != nil {
				errc <- fmt.Errorf("/pushed?get: %v", err)
				return
			}
			w.Header().Set("Content-Type", "text/html")
			w.Header().Set("Content-Length", strconv.Itoa(len(pushedBody)))
			w.WriteHeader(200)
			io.WriteString(w, pushedBody)
			errc <- nil

		case "/pushed?head":
			wantH := http.Header{}
			wantH.Set("User-Agent", userAgent)
			wantH.Set("Cookie", cookie)
			if err := checkPromisedReq(r, "HEAD", wantH); err != nil {
				errc <- fmt.Errorf("/pushed?head: %v", err)
				return
			}
			w.WriteHeader(204)
			errc <- nil

		default:
			errc <- fmt.Errorf("unknown RequestURL %q", r.URL.RequestURI())
		}
	})
	stURL = "https://" + st.authority()

	// Send one request, which should push two responses.
	st.greet()
	getSlash(st)
	for k := 0; k < 3; k++ {
		select {
		case <-time.After(2 * time.Second):
			t.Errorf("timeout waiting for handler %d to finish", k)
		case err := <-errc:
			if err != nil {
				t.Fatal(err)
			}
		}
	}

	checkPushPromise := func(f Frame, promiseID uint32, wantH [][2]string) error {
		pp, ok := f.(*PushPromiseFrame)
		if !ok {
			return fmt.Errorf("got a %T; want *PushPromiseFrame", f)
		}
		if !pp.HeadersEnded() {
			return fmt.Errorf("want END_HEADERS flag in PushPromiseFrame")
		}
		if got, want := pp.PromiseID, promiseID; got != want {
			return fmt.Errorf("got PromiseID %v; want %v", got, want)
		}
		gotH := st.decodeHeader(pp.HeaderBlockFragment())
		if !reflect.DeepEqual(gotH, wantH) {
			return fmt.Errorf("got promised headers %v; want %v", gotH, wantH)
		}
		return nil
	}
	checkHeaders := func(f Frame, wantH [][2]string) error {
		hf, ok := f.(*HeadersFrame)
		if !ok {
			return fmt.Errorf("got a %T; want *HeadersFrame", f)
		}
		gotH := st.decodeHeader(hf.HeaderBlockFragment())
		if !reflect.DeepEqual(gotH, wantH) {
			return fmt.Errorf("got response headers %v; want %v", gotH, wantH)
		}
		return nil
	}
	checkData := func(f Frame, wantData string) error {
		df, ok := f.(*DataFrame)
		if !ok {
			return fmt.Errorf("got a %T; want *DataFrame", f)
		}
		if gotData := string(df.Data()); gotData != wantData {
			return fmt.Errorf("got response data %q; want %q", gotData, wantData)
		}
		return nil
	}

	// Stream 1 has 2 PUSH_PROMISE + HEADERS + DATA
	// Stream 2 has HEADERS + DATA
	// Stream 4 has HEADERS
	expected := map[uint32][]func(Frame) error{
		1: {
			func(f Frame) error {
				return checkPushPromise(f, 2, [][2]string{
					{":method", "GET"},
					{":scheme", "https"},
					{":authority", st.authority()},
					{":path", "/pushed?get"},
					{"user-agent", userAgent},
				})
			},
			func(f Frame) error {
				return checkPushPromise(f, 4, [][2]string{
					{":method", "HEAD"},
					{":scheme", "https"},
					{":authority", st.authority()},
					{":path", "/pushed?head"},
					{"cookie", cookie},
					{"user-agent", userAgent},
				})
			},
			func(f Frame) error {
				return checkHeaders(f, [][2]string{
					{":status", "200"},
					{"content-type", "text/html"},
					{"content-length", strconv.Itoa(len(mainBody))},
				})
			},
			func(f Frame) error {
				return checkData(f, mainBody)
			},
		},
		2: {
			func(f Frame) error {
				return checkHeaders(f, [][2]string{
					{":status", "200"},
					{"content-type", "text/html"},
					{"content-length", strconv.Itoa(len(pushedBody))},
				})
			},
			func(f Frame) error {
				return checkData(f, pushedBody)
			},
		},
		4: {
			func(f Frame) error {
				return checkHeaders(f, [][2]string{
					{":status", "204"},
				})
			},
		},
	}

	consumed := map[uint32]int{}
	for k := 0; len(expected) > 0; k++ {
		f := st.readFrame()
		if f == nil {
			for id, left := range expected {
				t.Errorf("stream %d: missing %d frames", id, len(left))
			}
			break
		}
		id := f.Header().StreamID
		label := fmt.Sprintf("stream %d, frame %d", id, consumed[id])
		if len(expected[id]) == 0 {
			t.Fatalf("%s: unexpected frame %#+v", label, f)
		}
		check := expected[id][0]
		expected[id] = expected[id][1:]
		if len(expected[id]) == 0 {
			delete(expected, id)
		}
		if err := check(f); err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		consumed[id]++
	}
}

func TestServer_Push_SuccessNoRace(t *testing.T) { synctestTest(t, testServer_Push_SuccessNoRace) }
func testServer_Push_SuccessNoRace(t testing.TB) {
	// Regression test for issue #18326. Ensure the request handler can mutate
	// pushed request headers without racing with the PUSH_PROMISE write.
	errc := make(chan error, 2)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.RequestURI() {
		case "/":
			opt := &http.PushOptions{
				Header: http.Header{"User-Agent": {"testagent"}},
			}
			if err := w.(http.Pusher).Push("/pushed", opt); err != nil {
				errc <- fmt.Errorf("error pushing: %v", err)
				return
			}
			w.WriteHeader(200)
			errc <- nil

		case "/pushed":
			// Update request header, ensure there is no race.
			r.Header.Set("User-Agent", "newagent")
			r.Header.Set("Cookie", "cookie")
			w.WriteHeader(200)
			errc <- nil

		default:
			errc <- fmt.Errorf("unknown RequestURL %q", r.URL.RequestURI())
		}
	})

	// Send one request, which should push one response.
	st.greet()
	getSlash(st)
	for k := 0; k < 2; k++ {
		select {
		case <-time.After(2 * time.Second):
			t.Errorf("timeout waiting for handler %d to finish", k)
		case err := <-errc:
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestServer_Push_RejectRecursivePush(t *testing.T) {
	synctestTest(t, testServer_Push_RejectRecursivePush)
}
func testServer_Push_RejectRecursivePush(t testing.TB) {
	// Expect two requests, but might get three if there's a bug and the second push succeeds.
	errc := make(chan error, 3)
	handler := func(w http.ResponseWriter, r *http.Request) error {
		baseURL := "https://" + r.Host
		switch r.URL.Path {
		case "/":
			if err := w.(http.Pusher).Push(baseURL+"/push1", nil); err != nil {
				return fmt.Errorf("first Push()=%v, want nil", err)
			}
			return nil

		case "/push1":
			if got, want := w.(http.Pusher).Push(baseURL+"/push2", nil), ErrRecursivePush; got != want {
				return fmt.Errorf("Push()=%v, want %v", got, want)
			}
			return nil

		default:
			return fmt.Errorf("unexpected path: %q", r.URL.Path)
		}
	}
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		errc <- handler(w, r)
	})
	defer st.Close()
	st.greet()
	getSlash(st)
	if err := <-errc; err != nil {
		t.Errorf("First request failed: %v", err)
	}
	if err := <-errc; err != nil {
		t.Errorf("Second request failed: %v", err)
	}
}

func testServer_Push_RejectSingleRequest(t *testing.T, doPush func(http.Pusher, *http.Request) error, settings ...Setting) {
	synctestTest(t, func(t testing.TB) {
		testServer_Push_RejectSingleRequest_Bubble(t, doPush, settings...)
	})
}
func testServer_Push_RejectSingleRequest_Bubble(t testing.TB, doPush func(http.Pusher, *http.Request) error, settings ...Setting) {
	// Expect one request, but might get two if there's a bug and the push succeeds.
	errc := make(chan error, 2)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		errc <- doPush(w.(http.Pusher), r)
	})
	defer st.Close()
	st.greet()
	if err := st.fr.WriteSettings(settings...); err != nil {
		st.t.Fatalf("WriteSettings: %v", err)
	}
	st.wantSettingsAck()
	getSlash(st)
	if err := <-errc; err != nil {
		t.Error(err)
	}
	// Should not get a PUSH_PROMISE frame.
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: true,
	})
}

func TestServer_Push_RejectIfDisabled(t *testing.T) {
	testServer_Push_RejectSingleRequest(t,
		func(p http.Pusher, r *http.Request) error {
			if got, want := p.Push("https://"+r.Host+"/pushed", nil), http.ErrNotSupported; got != want {
				return fmt.Errorf("Push()=%v, want %v", got, want)
			}
			return nil
		},
		Setting{SettingEnablePush, 0})
}

func TestServer_Push_RejectWhenNoConcurrentStreams(t *testing.T) {
	testServer_Push_RejectSingleRequest(t,
		func(p http.Pusher, r *http.Request) error {
			if got, want := p.Push("https://"+r.Host+"/pushed", nil), ErrPushLimitReached; got != want {
				return fmt.Errorf("Push()=%v, want %v", got, want)
			}
			return nil
		},
		Setting{SettingMaxConcurrentStreams, 0})
}

func TestServer_Push_RejectWrongScheme(t *testing.T) {
	testServer_Push_RejectSingleRequest(t,
		func(p http.Pusher, r *http.Request) error {
			if err := p.Push("http://"+r.Host+"/pushed", nil); err == nil {
				return errors.New("Push() should have failed (push target URL is http)")
			}
			return nil
		})
}

func TestServer_Push_RejectMissingHost(t *testing.T) {
	testServer_Push_RejectSingleRequest(t,
		func(p http.Pusher, r *http.Request) error {
			if err := p.Push("https:pushed", nil); err == nil {
				return errors.New("Push() should have failed (push target URL missing host)")
			}
			return nil
		})
}

func TestServer_Push_RejectRelativePath(t *testing.T) {
	testServer_Push_RejectSingleRequest(t,
		func(p http.Pusher, r *http.Request) error {
			if err := p.Push("../test", nil); err == nil {
				return errors.New("Push() should have failed (push target is a relative path)")
			}
			return nil
		})
}

func TestServer_Push_RejectForbiddenMethod(t *testing.T) {
	testServer_Push_RejectSingleRequest(t,
		func(p http.Pusher, r *http.Request) error {
			if err := p.Push("https://"+r.Host+"/pushed", &http.PushOptions{Method: "POST"}); err == nil {
				return errors.New("Push() should have failed (cannot promise a POST)")
			}
			return nil
		})
}

func TestServer_Push_RejectForbiddenHeader(t *testing.T) {
	testServer_Push_RejectSingleRequest(t,
		func(p http.Pusher, r *http.Request) error {
			header := http.Header{
				"Content-Length":   {"10"},
				"Content-Encoding": {"gzip"},
				"Trailer":          {"Foo"},
				"Te":               {"trailers"},
				"Host":             {"test.com"},
				":authority":       {"test.com"},
			}
			if err := p.Push("https://"+r.Host+"/pushed", &http.PushOptions{Header: header}); err == nil {
				return errors.New("Push() should have failed (forbidden headers)")
			}
			return nil
		})
}

func TestServer_Push_StateTransitions(t *testing.T) {
	synctestTest(t, testServer_Push_StateTransitions)
}
func testServer_Push_StateTransitions(t testing.TB) {
	const body = "foo"

	gotPromise := make(chan bool)
	finishedPush := make(chan bool)

	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.RequestURI() {
		case "/":
			if err := w.(http.Pusher).Push("/pushed", nil); err != nil {
				t.Errorf("Push error: %v", err)
			}
			// Don't finish this request until the push finishes so we don't
			// nondeterministically interleave output frames with the push.
			<-finishedPush
		case "/pushed":
			<-gotPromise
		}
		w.Header().Set("Content-Type", "text/html")
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(200)
		io.WriteString(w, body)
	})
	defer st.Close()

	st.greet()
	if st.streamExists(2) {
		t.Fatal("stream 2 should be empty")
	}
	if got, want := st.streamState(2), StateIdle; got != want {
		t.Fatalf("streamState(2)=%v, want %v", got, want)
	}
	getSlash(st)
	// After the PUSH_PROMISE is sent, the stream should be stateHalfClosedRemote.
	_ = readFrame[*PushPromiseFrame](t, st)
	if got, want := st.streamState(2), StateHalfClosedRemote; got != want {
		t.Fatalf("streamState(2)=%v, want %v", got, want)
	}
	// We stall the HTTP handler for "/pushed" until the above check. If we don't
	// stall the handler, then the handler might write HEADERS and DATA and finish
	// the stream before we check st.streamState(2) -- should that happen, we'll
	// see stateClosed and fail the above check.
	close(gotPromise)
	st.wantHeaders(wantHeader{
		streamID:  2,
		endStream: false,
	})
	if got, want := st.streamState(2), StateClosed; got != want {
		t.Fatalf("streamState(2)=%v, want %v", got, want)
	}
	close(finishedPush)
}

func TestServer_Push_RejectAfterGoAway(t *testing.T) {
	synctestTest(t, testServer_Push_RejectAfterGoAway)
}
func testServer_Push_RejectAfterGoAway(t testing.TB) {
	ready := make(chan struct{})
	errc := make(chan error, 2)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		<-ready
		if got, want := w.(http.Pusher).Push("https://"+r.Host+"/pushed", nil), http.ErrNotSupported; got != want {
			errc <- fmt.Errorf("Push()=%v, want %v", got, want)
		}
		errc <- nil
	})
	defer st.Close()
	st.greet()
	getSlash(st)

	// Send GOAWAY and wait for it to be processed.
	st.fr.WriteGoAway(1, ErrCodeNo, nil)
	synctest.Wait()
	close(ready)
	if err := <-errc; err != nil {
		t.Error(err)
	}
}

func TestServer_Push_Underflow(t *testing.T) { synctestTest(t, testServer_Push_Underflow) }
func testServer_Push_Underflow(t testing.TB) {
	// Test for #63511: Send several requests which generate PUSH_PROMISE responses,
	// verify they all complete successfully.
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.RequestURI() {
		case "/":
			opt := &http.PushOptions{
				Header: http.Header{"User-Agent": {"testagent"}},
			}
			if err := w.(http.Pusher).Push("/pushed", opt); err != nil {
				t.Errorf("error pushing: %v", err)
			}
			w.WriteHeader(200)
		case "/pushed":
			r.Header.Set("User-Agent", "newagent")
			r.Header.Set("Cookie", "cookie")
			w.WriteHeader(200)
		default:
			t.Errorf("unknown RequestURL %q", r.URL.RequestURI())
		}
	})
	// Send several requests.
	st.greet()
	const numRequests = 4
	for i := 0; i < numRequests; i++ {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      uint32(1 + i*2), // clients send odd numbers
			BlockFragment: st.encodeHeader(),
			EndStream:     true,
			EndHeaders:    true,
		})
	}
	// Each request should result in one PUSH_PROMISE and two responses.
	numPushPromises := 0
	numHeaders := 0
	for numHeaders < numRequests*2 || numPushPromises < numRequests {
		f := st.readFrame()
		if f == nil {
			st.t.Fatal("conn is idle, want frame")
		}
		switch f := f.(type) {
		case *HeadersFrame:
			if !f.Flags.Has(FlagHeadersEndStream) {
				t.Fatalf("got HEADERS frame with no END_STREAM, expected END_STREAM: %v", f)
			}
			numHeaders++
		case *PushPromiseFrame:
			numPushPromises++
		}
	}
}
