// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

// HTTP status codes as registered with IANA.
// See: https://www.iana.org/assignments/http-status-codes/http-status-codes.xhtml
const (
	// StatusContinue 100 - RFC 7231, 6.2.1
	StatusContinue = 100

	// StatusSwitchingProtocols 101 - RFC 7231, 6.2.2
	StatusSwitchingProtocols = 101

	// StatusProcessing 102 - RFC 2518, 10.1
	StatusProcessing = 102

	// StatusEarlyHints 103 - RFC 8297
	StatusEarlyHints = 103

	// StatusOK 200 - RFC 7231, 6.3.1
	StatusOK = 200

	// StatusCreated 201 - RFC 7231, 6.3.2
	StatusCreated = 201

	// StatusAccepted 202 - RFC 7231, 6.3.3
	StatusAccepted = 202

	// StatusNonAuthoritativeInfo 203 - RFC 7231, 6.3.4
	StatusNonAuthoritativeInfo = 203

	// StatusNoContent 204 - RFC 7231, 6.3.5
	StatusNoContent = 204

	// StatusResetContent 205 - RFC 7231, 6.3.6
	StatusResetContent = 205

	// StatusPartialContent 206 - RFC 7233, 4.1
	StatusPartialContent = 206

	// Status 207 - RFC 4918, 11.1
	StatusMultiStatus = 207

	// StatusAlreadyReported 208 - RFC 5842, 7.1
	StatusAlreadyReported = 208

	// StatusIMUsed 226 - RFC 3229, 10.4.1
	StatusIMUsed = 226

	// StatusMultipleChoices 300 - RFC 7231, 6.4.1
	StatusMultipleChoices = 300

	// StatusMovedPermanently 301 - RFC 7231, 6.4.2
	StatusMovedPermanently = 301

	// StatusFound 302 - RFC 7231, 6.4.3
	StatusFound = 302

	// StatusSeeOther 303 - RFC 7231, 6.4.4
	StatusSeeOther = 303

	// StatusNotModified 304 - RFC 7232, 4.1
	StatusNotModified = 304

	// StatusUseProxy 305 - RFC 7231, 6.4.5
	StatusUseProxy = 305

	// RFC 7231, 6.4.6 (Unused)
	_ = 306

	// StatusTemporaryRedirect 307 - RFC 7231, 6.4.7
	StatusTemporaryRedirect = 307

	// StatusPermanentRedirect 308 - RFC 7538, 3
	StatusPermanentRedirect = 308

	// StatusBadRequest 400 - RFC 7231, 6.5.1
	StatusBadRequest = 400

	// StatusUnauthorized 401 - RFC 7235, 3.1
	StatusUnauthorized = 401

	// StatusPaymentRequired 402 - RFC 7231, 6.5.2
	StatusPaymentRequired = 402

	// StatusForbidden 403 - RFC 7231, 6.5.3
	StatusForbidden = 403

	// StatusNotFound 404 - RFC 7231, 6.5.4
	StatusNotFound = 404

	// StatusMethodNotAllowed 405 - RFC 7231, 6.5.5
	StatusMethodNotAllowed = 405

	// StatusNotAcceptable 406 - RFC 7231, 6.5.6
	StatusNotAcceptable = 406

	// StatusProxyAuthRequired 407 - RFC 7235, 3.2
	StatusProxyAuthRequired = 407

	// StatusRequestTimeout 408 - RFC 7231, 6.5.7
	StatusRequestTimeout = 408

	// StatusConflict 409 - RFC 7231, 6.5.8
	StatusConflict = 409

	// StatusGone 410 - RFC 7231, 6.5.9
	StatusGone = 410

	// StatusLengthRequired 411 - RFC 7231, 6.5.10
	StatusLengthRequired = 411

	// StatusPreconditionFailed 412 - RFC 7232, 4.2
	StatusPreconditionFailed = 412

	// StatusRequestEntityTooLarge 413 - RFC 7231, 6.5.11
	StatusRequestEntityTooLarge = 413

	// StatusRequestURITooLong 414 - RFC 7231, 6.5.12
	StatusRequestURITooLong = 414

	// StatusUnsupportedMediaType 415 - RFC 7231, 6.5.13
	StatusUnsupportedMediaType = 415

	// StatusRequestedRangeNotSatisfiable 416 - RFC 7233, 4.4
	StatusRequestedRangeNotSatisfiable = 416

	// StatusExpectationFailed 417 - RFC 7231, 6.5.14
	StatusExpectationFailed = 417

	// StatusTeapot 418 - RFC 7168, 2.3.3
	StatusTeapot = 418

	// StatusMisdirectedRequest 421 - RFC 7540, 9.1.2
	StatusMisdirectedRequest = 421

	// StatusUnprocessableEntity 422 - RFC 4918, 11.2
	StatusUnprocessableEntity = 422

	// StatusLocked 423 - RFC 4918, 11.3
	StatusLocked = 423

	// StatusFailedDependency 424 - RFC 4918, 11.4
	StatusFailedDependency = 424

	// StatusTooEarly 425 - RFC 8470, 5.2.
	StatusTooEarly = 425

	// StatusUpgradeRequired 426 - RFC 7231, 6.5.15
	StatusUpgradeRequired = 426

	// StatusPreconditionRequired 428 - RFC 6585, 3
	StatusPreconditionRequired = 428

	// StatusTooManyRequests 429 - RFC 6585, 4
	StatusTooManyRequests = 429

	// StatusRequestHeaderFieldsTooLarge 431 - RFC 6585, 5
	StatusRequestHeaderFieldsTooLarge = 431

	// StatusUnavailableForLegalReasons 451 - RFC 7725, 3
	StatusUnavailableForLegalReasons = 451

	// StatusInternalServerError 500 - RFC 7231, 6.6.1
	StatusInternalServerError = 500

	// StatusNotImplemented 501 - RFC 7231, 6.6.2
	StatusNotImplemented = 501

	// StatusBadGateway 502 - RFC 7231, 6.6.3
	StatusBadGateway = 502

	// StatusServiceUnavailable 503 - RFC 7231, 6.6.4
	StatusServiceUnavailable = 503

	// StatusGatewayTimeout 504 - RFC 7231, 6.6.5
	StatusGatewayTimeout = 504

	// StatusHTTPVersionNotSupported 505 - RFC 7231, 6.6.6
	StatusHTTPVersionNotSupported = 505

	// StatusVariantAlsoNegotiates 506 - RFC 2295, 8.1
	StatusVariantAlsoNegotiates = 506

	// StatusInsufficientStorage 507 - RFC 4918, 11.5
	StatusInsufficientStorage = 507

	// StatusLoopDetected 508 - RFC 5842, 7.2
	StatusLoopDetected = 508

	// StatusNotExtended 510 - RFC 2774, 7
	StatusNotExtended = 510

	// StatusNetworkAuthenticationRequired 511 - RFC 6585, 6
	StatusNetworkAuthenticationRequired = 511
)

var statusText = map[int]string{
	StatusContinue:           "Continue",
	StatusSwitchingProtocols: "Switching Protocols",
	StatusProcessing:         "Processing",
	StatusEarlyHints:         "Early Hints",

	StatusOK:                   "OK",
	StatusCreated:              "Created",
	StatusAccepted:             "Accepted",
	StatusNonAuthoritativeInfo: "Non-Authoritative Information",
	StatusNoContent:            "No Content",
	StatusResetContent:         "Reset Content",
	StatusPartialContent:       "Partial Content",
	StatusMultiStatus:          "Multi-Status",
	StatusAlreadyReported:      "Already Reported",
	StatusIMUsed:               "IM Used",

	StatusMultipleChoices:   "Multiple Choices",
	StatusMovedPermanently:  "Moved Permanently",
	StatusFound:             "Found",
	StatusSeeOther:          "See Other",
	StatusNotModified:       "Not Modified",
	StatusUseProxy:          "Use Proxy",
	StatusTemporaryRedirect: "Temporary Redirect",
	StatusPermanentRedirect: "Permanent Redirect",

	StatusBadRequest:                   "Bad Request",
	StatusUnauthorized:                 "Unauthorized",
	StatusPaymentRequired:              "Payment Required",
	StatusForbidden:                    "Forbidden",
	StatusNotFound:                     "Not Found",
	StatusMethodNotAllowed:             "Method Not Allowed",
	StatusNotAcceptable:                "Not Acceptable",
	StatusProxyAuthRequired:            "Proxy Authentication Required",
	StatusRequestTimeout:               "Request Timeout",
	StatusConflict:                     "Conflict",
	StatusGone:                         "Gone",
	StatusLengthRequired:               "Length Required",
	StatusPreconditionFailed:           "Precondition Failed",
	StatusRequestEntityTooLarge:        "Request Entity Too Large",
	StatusRequestURITooLong:            "Request URI Too Long",
	StatusUnsupportedMediaType:         "Unsupported Media Type",
	StatusRequestedRangeNotSatisfiable: "Requested Range Not Satisfiable",
	StatusExpectationFailed:            "Expectation Failed",
	StatusTeapot:                       "I'm a teapot",
	StatusMisdirectedRequest:           "Misdirected Request",
	StatusUnprocessableEntity:          "Unprocessable Entity",
	StatusLocked:                       "Locked",
	StatusFailedDependency:             "Failed Dependency",
	StatusTooEarly:                     "Too Early",
	StatusUpgradeRequired:              "Upgrade Required",
	StatusPreconditionRequired:         "Precondition Required",
	StatusTooManyRequests:              "Too Many Requests",
	StatusRequestHeaderFieldsTooLarge:  "Request Header Fields Too Large",
	StatusUnavailableForLegalReasons:   "Unavailable For Legal Reasons",

	StatusInternalServerError:           "Internal Server Error",
	StatusNotImplemented:                "Not Implemented",
	StatusBadGateway:                    "Bad Gateway",
	StatusServiceUnavailable:            "Service Unavailable",
	StatusGatewayTimeout:                "Gateway Timeout",
	StatusHTTPVersionNotSupported:       "HTTP Version Not Supported",
	StatusVariantAlsoNegotiates:         "Variant Also Negotiates",
	StatusInsufficientStorage:           "Insufficient Storage",
	StatusLoopDetected:                  "Loop Detected",
	StatusNotExtended:                   "Not Extended",
	StatusNetworkAuthenticationRequired: "Network Authentication Required",
}

// StatusText returns a text for the HTTP status code. It returns the empty
// string if the code is unknown.
func StatusText(code int) string {
	return statusText[code]
}
