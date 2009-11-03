// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

type alertLevel int
type alertType int

const (
	alertLevelWarning	alertLevel	= 1;
	alertLevelError		alertLevel	= 2;
)

const (
	alertCloseNotify		alertType	= 0;
	alertUnexpectedMessage		alertType	= 10;
	alertBadRecordMAC		alertType	= 20;
	alertDecryptionFailed		alertType	= 21;
	alertRecordOverflow		alertType	= 22;
	alertDecompressionFailure	alertType	= 30;
	alertHandshakeFailure		alertType	= 40;
	alertBadCertificate		alertType	= 42;
	alertUnsupportedCertificate	alertType	= 43;
	alertCertificateRevoked		alertType	= 44;
	alertCertificateExpired		alertType	= 45;
	alertCertificateUnknown		alertType	= 46;
	alertIllegalParameter		alertType	= 47;
	alertUnknownCA			alertType	= 48;
	alertAccessDenied		alertType	= 49;
	alertDecodeError		alertType	= 50;
	alertDecryptError		alertType	= 51;
	alertProtocolVersion		alertType	= 70;
	alertInsufficientSecurity	alertType	= 71;
	alertInternalError		alertType	= 80;
	alertUserCanceled		alertType	= 90;
	alertNoRenegotiation		alertType	= 100;
)

type alert struct {
	level	alertLevel;
	error	alertType;
}
