// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"encoding/json"
)

// nullResult is the JSON result of a request the proxy declines to map.
var nullResult = json.RawMessage("null")

// clientRequest handles one request from the client.
func (p *Proxy) clientRequest(ctx context.Context, m *message) {
	switch m.Method {
	case "initialize":
		p.initialize(m)
	case "shutdown":
		p.mu.Lock()
		p.shutdown = true
		p.mu.Unlock()
		p.send(p.server, m)
	default:
		p.request(ctx, m)
	}
}

// initialize forwards the initialize request with the position encodings
// removed from the client's capabilities, so gopls speaks UTF-16 to the
// proxy, the encoding the proxy converts for the client.
func (p *Proxy) initialize(m *message) {
	if params, err := decode(m.Params); err == nil {
		dropPositionEncodings(params)
		if raw, err := encode(params); err == nil {
			m.Params = raw
		}
	}
	p.addPending(m.ID, pending{method: m.Method})
	p.send(p.server, m)
}

// dropPositionEncodings removes capabilities.general.positionEncodings
// from decoded initialize params.
func dropPositionEncodings(params any) {
	root, _ := params.(map[string]any)
	caps, _ := root["capabilities"].(map[string]any)
	general, _ := caps["general"].(map[string]any)
	delete(general, "positionEncodings")
}

// dropDiagnosticProvider removes capabilities.diagnosticProvider from a
// decoded initialize result: the proxy merges compiler diagnostics into
// published diagnostics, so the client must not pull them from gopls.
func dropDiagnosticProvider(result any) {
	root, _ := result.(map[string]any)
	caps, _ := root["capabilities"].(map[string]any)
	delete(caps, "diagnosticProvider")
}

// requestPath returns the path of the document that decoded request
// params name, or "" when they name none.
func requestPath(params any) string {
	root, _ := params.(map[string]any)
	td, _ := root["textDocument"].(map[string]any)
	uri, _ := td["uri"].(string)
	path, err := uriPath(uri)
	if err != nil {
		return ""
	}
	return path
}

// target returns the lowering a request on path is mapped with: none for
// a Go document. It reports false when the request cannot be answered, a
// loGo/s document that has no lowering made from its current text.
func (p *Proxy) target(path string) (*lowered, bool) {
	if !isLogosPath(path) {
		return nil, true
	}
	lw := p.tables.Load().bySrc[path]
	return lw, lw != nil && lw.fresh
}

// request forwards one client request to gopls with its URIs and
// positions mapped to the lowered documents. It first recompiles the
// packages whose documents changed, so the request sees every edit the
// client made before it. A request on a loGo/s document with no current
// lowering gets a null result, except for textDocument/codeAction, which
// answers the compiler autofix candidates of the source itself.
func (p *Proxy) request(ctx context.Context, m *message) {
	params, err := decode(m.Params)
	if err != nil {
		p.send(p.server, m)
		return
	}
	path := requestPath(params)
	p.work.Lock()
	defer p.work.Unlock()
	p.refreshLocked(ctx)
	lw, ok := p.target(path)
	if !ok {
		// A codeAction request names no lowered document when the
		// source does not compile, yet the compiler diagnostics of
		// the source itself may still carry autofix candidates. Answer
		// those directly, without the gopls mapping there is nothing
		// to map. Every other position request gets a null result,
		// never a position computed from an older text.
		if m.Method == "textDocument/codeAction" && isLogosPath(path) {
			p.respond(m.ID, p.finishResult(pending{method: m.Method, path: path, params: m.Params}, []any{}))
			return
		}
		p.send(p.client, &message{ID: m.ID, Result: nullResult})
		return
	}
	tb := p.tables.Load()
	out := &message{ID: m.ID, Method: m.Method, Params: m.Params}
	if params != nil {
		if out.Params, err = encode(tb.toLowered().rewrite(params, lw)); err != nil {
			p.send(p.client, &message{ID: m.ID, Result: nullResult})
			return
		}
	}
	pend := pending{method: m.Method, path: path, lw: lw, tb: tb, params: m.Params}
	if m.Method == "textDocument/codeAction" {
		pend.fixVersion, pend.fixHas = p.docVersion(path)
	}
	p.addPending(m.ID, pend)
	p.send(p.server, out)
}

// fromServer handles one message from gopls.
func (p *Proxy) fromServer(ctx context.Context, m *message) {
	switch {
	case m.isResponse():
		p.serverResponse(ctx, m)
	case !m.isRequest() && m.Method == "textDocument/publishDiagnostics":
		p.serverDiagnostics(m)
	default:
		p.forwardToClient(m)
	}
}

// forwardToClient sends a notification or request of gopls to the client
// with the lowered documents it names mapped back to their sources.
func (p *Proxy) forwardToClient(m *message) {
	if params, err := decode(m.Params); err == nil && params != nil {
		if raw, err := encode(p.tables.Load().toSource().rewrite(params, nil)); err == nil {
			m.Params = raw
		}
	}
	p.send(p.client, m)
}

// serverResponse forwards the response of gopls to the client, with the
// result mapped through the lowerings the request was mapped with. An
// error response, and a response to no request the proxy tracked, go
// through unchanged.
func (p *Proxy) serverResponse(ctx context.Context, m *message) {
	p.mu.Lock()
	pend, ok := p.pending[string(m.ID)]
	p.mu.Unlock()
	result, err := decode(m.Result)
	if !ok || len(m.Error) > 0 || err != nil {
		p.send(p.client, m)
		p.finish(m.ID)
		return
	}
	result = p.finishResult(pend, pend.tb.toSource().rewrite(prepareResult(pend, result), pend.lw))
	if p.opts.Hints != nil && servesHints(pend.method) && p.spawnHints(ctx, m.ID, pend, result) {
		return
	}
	p.respond(m.ID, result)
	p.finish(m.ID)
}

// finishResult applies the per-method changes to a mapped result.
func (p *Proxy) finishResult(pend pending, result any) any {
	switch pend.method {
	case "initialize":
		dropDiagnosticProvider(result)
	case "textDocument/codeAction":
		return p.mergeCodeActions(pend, result)
	}
	return result
}

// respond sends a successful response to the client.
func (p *Proxy) respond(id json.RawMessage, result any) {
	raw, err := encode(result)
	if err != nil {
		p.logf("encoding a response: %v", err)
		raw = nullResult
	}
	p.send(p.client, &message{ID: id, Result: raw})
}

// serverDiagnostics records the diagnostics gopls published for a
// document, and publishes the merged diagnostics of the client document
// they belong to.
func (p *Proxy) serverDiagnostics(m *message) {
	params, err := decode(m.Params)
	root, _ := params.(map[string]any)
	uri, _ := root["uri"].(string)
	if err != nil || uri == "" {
		p.send(p.client, m)
		return
	}
	items, _ := json.Marshal(root["diagnostics"])
	version, hasVer := number(root["version"])
	p.mu.Lock()
	p.srvDiags[uri] = serverDiags{version: version, hasVer: hasVer, raw: items}
	p.mu.Unlock()
	tb := p.tables.Load()
	if lw := tb.lookup(tb.byVirt, uri); lw != nil {
		uri = lw.srcURI()
	}
	p.publish(uri)
}
