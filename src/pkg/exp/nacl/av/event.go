// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NaCl GUI events.
// Clients do not have raw access to the event stream
// (only filtered through the lens of package draw)
// but perhaps they will.

package av

import (
	"bytes";
	"debug/binary";
	"exp/draw";
	"log";
	"os";
	"time";
)

// An eventType identifies the type of a Native Client Event.
type eventType uint8;
const (
	eventActive = 1+iota;
	eventExpose;
	eventKeyDown;
	eventKeyUp;
	eventMouseMotion;
	eventMouseButtonDown;
	eventMouseButtonUp;
	eventQuit;
	eventUnsupported;
)

// A key represents a key on a keyboard.
type key uint16
const (
	keyUnknown      = 0;
	keyFirst        = 0;
	keyBackspace    = 8;
	keyTab          = 9;
	keyClear        = 12;
	keyReturn       = 13;
	keyPause        = 19;
	keyEscape       = 27;
	keySpace        = 32;
	keyExclaim      = 33;
	keyQuotedbl     = 34;
	keyHash         = 35;
	keyDollar       = 36;
	keyAmpersand    = 38;
	keyQuote        = 39;
	keyLeftparen    = 40;
	keyRightparen   = 41;
	keyAsterisk     = 42;
	keyPlus         = 43;
	keyComma        = 44;
	keyMinus        = 45;
	keyPeriod       = 46;
	keySlash        = 47;
	key0            = 48;
	key1            = 49;
	key2            = 50;
	key3            = 51;
	key4            = 52;
	key5            = 53;
	key6            = 54;
	key7            = 55;
	key8            = 56;
	key9            = 57;
	keyColon        = 58;
	keySemicolon    = 59;
	keyLess         = 60;
	keyEquals       = 61;
	keyGreater      = 62;
	keyQuestion     = 63;
	keyAt           = 64;
	keyLeftbracket  = 91;
	keyBackslash    = 92;
	keyRightbracket = 93;
	keyCaret        = 94;
	keyUnderscore   = 95;
	keyBackquote    = 96;
	keyA            = 97;
	keyB            = 98;
	keyC            = 99;
	keyD            = 100;
	keyE            = 101;
	keyF            = 102;
	keyG            = 103;
	keyH            = 104;
	keyI            = 105;
	keyJ            = 106;
	keyK            = 107;
	keyL            = 108;
	keyM            = 109;
	keyN            = 110;
	keyO            = 111;
	keyP            = 112;
	keyQ            = 113;
	keyR            = 114;
	keyS            = 115;
	keyT            = 116;
	keyU            = 117;
	keyV            = 118;
	keyW            = 119;
	keyX            = 120;
	keyY            = 121;
	keyZ            = 122;
	keyDelete       = 127;
	keyWorld0      = 160;
	keyWorld1      = 161;
	keyWorld2      = 162;
	keyWorld3      = 163;
	keyWorld4      = 164;
	keyWorld5      = 165;
	keyWorld6      = 166;
	keyWorld7      = 167;
	keyWorld8      = 168;
	keyWorld9      = 169;
	keyWorld10     = 170;
	keyWorld11     = 171;
	keyWorld12     = 172;
	keyWorld13     = 173;
	keyWorld14     = 174;
	keyWorld15     = 175;
	keyWorld16     = 176;
	keyWorld17     = 177;
	keyWorld18     = 178;
	keyWorld19     = 179;
	keyWorld20     = 180;
	keyWorld21     = 181;
	keyWorld22     = 182;
	keyWorld23     = 183;
	keyWorld24     = 184;
	keyWorld25     = 185;
	keyWorld26     = 186;
	keyWorld27     = 187;
	keyWorld28     = 188;
	keyWorld29     = 189;
	keyWorld30     = 190;
	keyWorld31     = 191;
	keyWorld32     = 192;
	keyWorld33     = 193;
	keyWorld34     = 194;
	keyWorld35     = 195;
	keyWorld36     = 196;
	keyWorld37     = 197;
	keyWorld38     = 198;
	keyWorld39     = 199;
	keyWorld40     = 200;
	keyWorld41     = 201;
	keyWorld42     = 202;
	keyWorld43     = 203;
	keyWorld44     = 204;
	keyWorld45     = 205;
	keyWorld46     = 206;
	keyWorld47     = 207;
	keyWorld48     = 208;
	keyWorld49     = 209;
	keyWorld50     = 210;
	keyWorld51     = 211;
	keyWorld52     = 212;
	keyWorld53     = 213;
	keyWorld54     = 214;
	keyWorld55     = 215;
	keyWorld56     = 216;
	keyWorld57     = 217;
	keyWorld58     = 218;
	keyWorld59     = 219;
	keyWorld60     = 220;
	keyWorld61     = 221;
	keyWorld62     = 222;
	keyWorld63     = 223;
	keyWorld64     = 224;
	keyWorld65     = 225;
	keyWorld66     = 226;
	keyWorld67     = 227;
	keyWorld68     = 228;
	keyWorld69     = 229;
	keyWorld70     = 230;
	keyWorld71     = 231;
	keyWorld72     = 232;
	keyWorld73     = 233;
	keyWorld74     = 234;
	keyWorld75     = 235;
	keyWorld76     = 236;
	keyWorld77     = 237;
	keyWorld78     = 238;
	keyWorld79     = 239;
	keyWorld80     = 240;
	keyWorld81     = 241;
	keyWorld82     = 242;
	keyWorld83     = 243;
	keyWorld84     = 244;
	keyWorld85     = 245;
	keyWorld86     = 246;
	keyWorld87     = 247;
	keyWorld88     = 248;
	keyWorld89     = 249;
	keyWorld90     = 250;
	keyWorld91     = 251;
	keyWorld92     = 252;
	keyWorld93     = 253;
	keyWorld94     = 254;
	keyWorld95     = 255;

	// Numeric keypad
	keyKp0          = 256;
	keyKp1          = 257;
	keyKp2          = 258;
	keyKp3          = 259;
	keyKp4          = 260;
	keyKp5          = 261;
	keyKp6          = 262;
	keyKp7          = 263;
	keyKp8          = 264;
	keyKp9          = 265;
	keyKpPeriod    = 266;
	keyKpDivide    = 267;
	keyKpMultiply  = 268;
	keyKpMinus     = 269;
	keyKpPlus      = 270;
	keyKpEnter     = 271;
	keyKpEquals    = 272;

	// Arrow & insert/delete pad
	keyUp           = 273;
	keyDown         = 274;
	keyRight        = 275;
	keyLeft         = 276;
	keyInsert       = 277;
	keyHome         = 278;
	keyEnd          = 279;
	keyPageup       = 280;
	keyPagedown     = 281;

	// Function keys
	keyF1           = 282;
	keyF2           = 283;
	keyF3           = 284;
	keyF4           = 285;
	keyF5           = 286;
	keyF6           = 287;
	keyF7           = 288;
	keyF8           = 289;
	keyF9           = 290;
	keyF10          = 291;
	keyF11          = 292;
	keyF12          = 293;
	keyF13          = 294;
	keyF14          = 295;
	keyF15          = 296;

	// Modifier keys
	keyNumlock      = 300;
	keyCapslock     = 301;
	keyScrollock    = 302;
	keyRshift       = 303;
	keyLshift       = 304;
	keyRctrl        = 305;
	keyLctrl        = 306;
	keyRalt         = 307;
	keyLalt         = 308;
	keyRmeta        = 309;
	keyLmeta        = 310;
	keyLsuper       = 311;
	keyRsuper       = 312;
	keyMode         = 313;
	keyCompose      = 314;

	// Misc keys
	keyHelp         = 315;
	keyPrint        = 316;
	keySysreq       = 317;
	keyBreak        = 318;
	keyMenu         = 319;
	keyPower        = 320;
	keyEuro         = 321;
	keyUndo         = 322;

	// Add any other keys here
	keyLast
)

// A keymod is a set of bit flags
type keymod uint16
const (
	keymodNone  = 0x0000;
	keymodLshift= 0x0001;
	keymodRshift= 0x0002;
	keymodLctrl = 0x0040;
	keymodRctrl = 0x0080;
	keymodLalt  = 0x0100;
	keymodRalt  = 0x0200;
	keymodLmeta = 0x0400;
	keymodRmeta = 0x0800;
	keymodNum   = 0x1000;
	keymodCaps  = 0x2000;
	keymodMode  = 0x4000;
	keymodReserved = 0x8000
)

const (
	mouseButtonLeft = 1;
	mouseButtonMiddle = 2;
	mouseButtonRight = 3;
	mouseScrollUp = 4;
	mouseScrollDown = 5
)

const (
	mouseStateLeftButtonPressed = 1;
	mouseStateMiddleButtonPressed = 2;
	mouseStateRightButtonPressed = 4
)

const (
	activeMouse = 1;        //  mouse leaving/entering
	activeInputFocus = 2;  // input focus lost/restored
	activeApplication = 4   // application minimized/restored
)

const maxEventBytes = 64

type activeEvent struct {
	EventType eventType;
	Gain uint8;
	State uint8;
}

type exposeEvent struct {
	EventType eventType;
}

type keyboardEvent struct {
	EventType eventType;
	Device uint8;
	State uint8;
	Pad uint8;
	ScanCode uint8;
	Pad1 uint8;
	Key key;
	Mod keymod;
	Unicode uint16;
}

type mouseMotionEvent struct {
	EventType eventType;
	Device uint8;
	Buttons uint8;
	Pad uint8;
	X uint16;
	Y uint16;
	Xrel int16;
	Yrel int16;
}

type mouseButtonEvent struct {
	EventType eventType;
	Device uint8;
	Button uint8;
	State uint8;
	X uint16;
	Y uint16;
}

type quitEvent struct {
	EventType eventType;
}

type syncEvent struct {
}

type event interface {
}

type reader []byte
func (r *reader) Read(p []byte) (n int, err os.Error) {
	b := *r;
	if len(b) == 0 && len(p) > 0 {
		return 0, os.EOF;
	}
	n = bytes.Copy(p, b);
	*r = b[n:len(b)];
	return;
}

func (w *Window) readEvents() {
	buf := make([]byte, maxEventBytes);
	clean := false;
	var (
		ea *activeEvent;
		ee *exposeEvent;
		ke *keyboardEvent;
		mme *mouseMotionEvent;
		mbe *mouseButtonEvent;
		qe *quitEvent;
	)
	var m draw.Mouse;
	for {
		if err := videoPollEvent(buf); err != nil {
			if !clean {
				clean = w.resizec <- false;
			}
			time.Sleep(10e6);	// 10ms
			continue;
		}
		clean = false;
		var e event;
		switch buf[0] {
		default:
			log.Stdout("unsupported event type", buf[0]);
			continue;
		case eventActive:
			ea = new(activeEvent);
			e = ea;
		case eventExpose:
			ee = new(exposeEvent);
			e = ee;
		case eventKeyDown, eventKeyUp:
			ke = new(keyboardEvent);
			e = ke;
		case eventMouseMotion:
			mme = new(mouseMotionEvent);
			e = mme;
		case eventMouseButtonDown, eventMouseButtonUp:
			mbe = new(mouseButtonEvent);
			e = mbe;
		case eventQuit:
			qe = new(quitEvent);
			e = qe;
		}
		r := reader(buf);
		if err := binary.Read(&r, binary.LittleEndian, e); err != nil {
			log.Stdout("unpacking %T event: %s", e, err);
			continue;
		}
		// log.Stdoutf("%#v\n", e);
		switch buf[0] {
		case eventExpose:
			w.resizec <- true
		case eventKeyDown:
			w.kbdc <- int(ke.Key);
		case eventKeyUp:
			w.kbdc <- -int(ke.Key);
		case eventMouseMotion:
			m.X = int(mme.X);
			m.Y = int(mme.Y);
			m.Buttons = int(mme.Buttons);
			m.Nsec = time.Nanoseconds();
			_ = w.mousec <- m;
		case eventMouseButtonDown:
			m.X = int(mbe.X);
			m.Y = int(mbe.Y);
			// TODO(rsc): Remove uint cast once 8g bug is fixed.
			m.Buttons |= 1<<uint(mbe.Button-1);
			m.Nsec = time.Nanoseconds();
			_ = w.mousec <- m;
		case eventMouseButtonUp:
			m.X = int(mbe.X);
			m.Y = int(mbe.Y);
			// TODO(rsc): Remove uint cast once 8g bug is fixed.
			m.Buttons &^= 1<<uint(mbe.Button-1);
			m.Nsec = time.Nanoseconds();
			_ = w.mousec <- m;
		case eventQuit:
			w.quitc <- true;
		}
	}
}
