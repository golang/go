// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import "sync"

type tableEntry struct {
	name  string
	value string
}

// staticTableEntry returns the static table entry with the given index.
func staticTableEntry(index int64) (tableEntry, error) {
	if index >= int64(len(staticTableEntries)) {
		return tableEntry{}, errQPACKDecompressionFailed
	}
	return staticTableEntries[index], nil
}

func initStaticTableMaps() {
	staticTableByName = make(map[string]int)
	staticTableByNameValue = make(map[tableEntry]int)
	for i, ent := range staticTableEntries {
		if _, ok := staticTableByName[ent.name]; !ok {
			staticTableByName[ent.name] = i
		}
		staticTableByNameValue[ent] = i
	}
}

var (
	staticTableOnce        sync.Once
	staticTableByName      map[string]int
	staticTableByNameValue map[tableEntry]int
)

// https://www.rfc-editor.org/rfc/rfc9204.html#appendix-A
//
// Note that this is different from the HTTP/2 static table.
var staticTableEntries = [...]tableEntry{
	0:  {":authority", ""},
	1:  {":path", "/"},
	2:  {"age", "0"},
	3:  {"content-disposition", ""},
	4:  {"content-length", "0"},
	5:  {"cookie", ""},
	6:  {"date", ""},
	7:  {"etag", ""},
	8:  {"if-modified-since", ""},
	9:  {"if-none-match", ""},
	10: {"last-modified", ""},
	11: {"link", ""},
	12: {"location", ""},
	13: {"referer", ""},
	14: {"set-cookie", ""},
	15: {":method", "CONNECT"},
	16: {":method", "DELETE"},
	17: {":method", "GET"},
	18: {":method", "HEAD"},
	19: {":method", "OPTIONS"},
	20: {":method", "POST"},
	21: {":method", "PUT"},
	22: {":scheme", "http"},
	23: {":scheme", "https"},
	24: {":status", "103"},
	25: {":status", "200"},
	26: {":status", "304"},
	27: {":status", "404"},
	28: {":status", "503"},
	29: {"accept", "*/*"},
	30: {"accept", "application/dns-message"},
	31: {"accept-encoding", "gzip, deflate, br"},
	32: {"accept-ranges", "bytes"},
	33: {"access-control-allow-headers", "cache-control"},
	34: {"access-control-allow-headers", "content-type"},
	35: {"access-control-allow-origin", "*"},
	36: {"cache-control", "max-age=0"},
	37: {"cache-control", "max-age=2592000"},
	38: {"cache-control", "max-age=604800"},
	39: {"cache-control", "no-cache"},
	40: {"cache-control", "no-store"},
	41: {"cache-control", "public, max-age=31536000"},
	42: {"content-encoding", "br"},
	43: {"content-encoding", "gzip"},
	44: {"content-type", "application/dns-message"},
	45: {"content-type", "application/javascript"},
	46: {"content-type", "application/json"},
	47: {"content-type", "application/x-www-form-urlencoded"},
	48: {"content-type", "image/gif"},
	49: {"content-type", "image/jpeg"},
	50: {"content-type", "image/png"},
	51: {"content-type", "text/css"},
	52: {"content-type", "text/html; charset=utf-8"},
	53: {"content-type", "text/plain"},
	54: {"content-type", "text/plain;charset=utf-8"},
	55: {"range", "bytes=0-"},
	56: {"strict-transport-security", "max-age=31536000"},
	57: {"strict-transport-security", "max-age=31536000; includesubdomains"},
	58: {"strict-transport-security", "max-age=31536000; includesubdomains; preload"},
	59: {"vary", "accept-encoding"},
	60: {"vary", "origin"},
	61: {"x-content-type-options", "nosniff"},
	62: {"x-xss-protection", "1; mode=block"},
	63: {":status", "100"},
	64: {":status", "204"},
	65: {":status", "206"},
	66: {":status", "302"},
	67: {":status", "400"},
	68: {":status", "403"},
	69: {":status", "421"},
	70: {":status", "425"},
	71: {":status", "500"},
	72: {"accept-language", ""},
	73: {"access-control-allow-credentials", "FALSE"},
	74: {"access-control-allow-credentials", "TRUE"},
	75: {"access-control-allow-headers", "*"},
	76: {"access-control-allow-methods", "get"},
	77: {"access-control-allow-methods", "get, post, options"},
	78: {"access-control-allow-methods", "options"},
	79: {"access-control-expose-headers", "content-length"},
	80: {"access-control-request-headers", "content-type"},
	81: {"access-control-request-method", "get"},
	82: {"access-control-request-method", "post"},
	83: {"alt-svc", "clear"},
	84: {"authorization", ""},
	85: {"content-security-policy", "script-src 'none'; object-src 'none'; base-uri 'none'"},
	86: {"early-data", "1"},
	87: {"expect-ct", ""},
	88: {"forwarded", ""},
	89: {"if-range", ""},
	90: {"origin", ""},
	91: {"purpose", "prefetch"},
	92: {"server", ""},
	93: {"timing-allow-origin", "*"},
	94: {"upgrade-insecure-requests", "1"},
	95: {"user-agent", ""},
	96: {"x-forwarded-for", ""},
	97: {"x-frame-options", "deny"},
	98: {"x-frame-options", "sameorigin"},
}
